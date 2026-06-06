import os
import random

import imageio
import numpy as np
from torch.utils.data import Dataset
import dataset.imutils as imutils


def parse_disaster_event(data_name):
    """Return the disaster-event id from a BRIGHT split entry."""
    name = str(data_name).replace("\\", "/")
    if "/" in name:
        return name.split("/", 1)[0]
    if "_" in name:
        return name.rsplit("_", 1)[0]
    return name


def img_loader(path):
    img = np.array(imageio.imread(path), np.float32)
    return img


class MultimodalDamageAssessmentDatset(Dataset):
    def __init__(
        self,
        dataset_path,
        data_list,
        crop_size,
        max_iters=None,
        type='train',
        data_loader=img_loader,
        suffix='.tif',
        use_dacutmix=False,
        dacutmix_prob=0.0,
        damage_class_ids=(2, 3),
        dacutmix_min_damage_pixels=200,
        dacutmix_min_damage_ratio=0.05,
        dacutmix_patch_min_ratio=0.20,
        dacutmix_patch_max_ratio=0.50,
        dacutmix_box_tries=10,
        dacutmix_donor_tries=10,
        return_dacutmix_stats=False,
    ):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type
        self.suffix = suffix
        self.damage_class_ids = tuple(int(c) for c in damage_class_ids)
        self.use_dacutmix = bool(use_dacutmix)
        self.dacutmix_prob = float(dacutmix_prob)
        self.dacutmix_min_damage_pixels = int(dacutmix_min_damage_pixels)
        self.dacutmix_min_damage_ratio = float(dacutmix_min_damage_ratio)
        self.dacutmix_patch_min_ratio = float(dacutmix_patch_min_ratio)
        self.dacutmix_patch_max_ratio = float(dacutmix_patch_max_ratio)
        self.dacutmix_box_tries = int(dacutmix_box_tries)
        self.dacutmix_donor_tries = int(dacutmix_donor_tries)
        self.return_dacutmix_stats = bool(return_dacutmix_stats)

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size
        self.events = [parse_disaster_event(x) for x in self.data_list]
        self.event_to_indices = {}
        for i, event in enumerate(self.events):
            self.event_to_indices.setdefault(event, []).append(i)

        if len(self.event_to_indices) < 2:
            self.use_dacutmix = False
        self.last_dacutmix_info = None

    def _load_sample(self, index):
        data_name = self.data_list[index]
        pre_path = os.path.join(self.dataset_path, 'pre-event', data_name + '_pre_disaster' + self.suffix)
        post_path = os.path.join(self.dataset_path, 'post-event', data_name + '_post_disaster'  + self.suffix)
        label_path = os.path.join(self.dataset_path, 'target', data_name + '_building_damage'  + self.suffix)

        pre_img = self.loader(pre_path)[:, :, 0:3]
        post_img = self.loader(post_path)
        post_img = np.stack((post_img,) * 3, axis=-1)
        clf_label = self.loader(label_path)
        return pre_img, post_img, clf_label

    def __augment(self, aug, pre_img, post_img, label):
        if aug:
            pre_img, post_img, label = imutils.random_crop(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)
        return pre_img, post_img, label

    def __normalize_transpose(self, pre_img, post_img, label):
        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def __transforms(self, aug, pre_img, post_img, label):
        pre_img, post_img, label = self.__augment(aug, pre_img, post_img, label)
        return self.__normalize_transpose(pre_img, post_img, label)

    def _sample_donor_index(self, index):
        current_event = self.events[index]
        donor_events = [e for e in self.event_to_indices if e != current_event]
        if not donor_events:
            return None
        donor_event = random.choice(donor_events)
        return random.choice(self.event_to_indices[donor_event])

    def _random_patch_box(self, h, w):
        min_ratio = min(max(self.dacutmix_patch_min_ratio, 0.01), 1.0)
        max_ratio = min(max(self.dacutmix_patch_max_ratio, min_ratio), 1.0)
        min_h = max(1, int(round(h * min_ratio)))
        max_h = max(min_h, int(round(h * max_ratio)))
        min_w = max(1, int(round(w * min_ratio)))
        max_w = max(min_w, int(round(w * max_ratio)))
        patch_h = random.randint(min_h, max_h)
        patch_w = random.randint(min_w, max_w)
        y0 = random.randint(0, max(0, h - patch_h))
        x0 = random.randint(0, max(0, w - patch_w))
        return y0, y0 + patch_h, x0, x0 + patch_w

    def _find_damage_patch_box(self, label):
        h, w = label.shape
        for _ in range(max(1, self.dacutmix_box_tries)):
            y0, y1, x0, x1 = self._random_patch_box(h, w)
            patch = label[y0:y1, x0:x1]
            damage_pixels = np.isin(patch, self.damage_class_ids).sum()
            min_pixels = max(
                self.dacutmix_min_damage_pixels,
                int(round(patch.size * self.dacutmix_min_damage_ratio)),
            )
            if damage_pixels >= min_pixels:
                return y0, y1, x0, x1
        return None

    def _apply_dacutmix(self, index, pre_img, post_img, label):
        self.last_dacutmix_info = None
        for _ in range(max(1, self.dacutmix_donor_tries)):
            donor_idx = self._sample_donor_index(index)
            if donor_idx is None:
                return pre_img, post_img, label

            donor_pre, donor_post, donor_label = self._load_sample(donor_idx)
            donor_pre, donor_post, donor_label = self.__augment(
                True, donor_pre, donor_post, donor_label
            )
            box = self._find_damage_patch_box(donor_label)
            if box is None:
                continue

            y0, y1, x0, x1 = box
            pre_img = pre_img.copy()
            post_img = post_img.copy()
            label = label.copy()
            pre_img[y0:y1, x0:x1, :] = donor_pre[y0:y1, x0:x1, :]
            post_img[y0:y1, x0:x1, :] = donor_post[y0:y1, x0:x1, :]
            label[y0:y1, x0:x1] = donor_label[y0:y1, x0:x1]
            self.last_dacutmix_info = {
                "base_idx": index,
                "base_name": self.data_list[index],
                "base_event": self.events[index],
                "donor_idx": donor_idx,
                "donor_name": self.data_list[donor_idx],
                "donor_event": self.events[donor_idx],
                "box": box,
            }
            return pre_img, post_img, label

        return pre_img, post_img, label

    def __getitem__(self, index):
        self.last_dacutmix_info = None
        dacutmix_attempted = 0
        dacutmix_applied = 0
        pre_img, post_img, clf_label = self._load_sample(index)

        if 'train' in self.data_pro_type:
            pre_img, post_img, clf_label = self.__augment(True, pre_img, post_img, clf_label)
            if self.use_dacutmix and random.random() < self.dacutmix_prob:
                dacutmix_attempted = 1
                pre_img, post_img, clf_label = self._apply_dacutmix(
                    index, pre_img, post_img, clf_label
                )
                dacutmix_applied = 1 if self.last_dacutmix_info is not None else 0
            pre_img, post_img, clf_label = self.__normalize_transpose(pre_img, post_img, clf_label)
        else:
            pre_img, post_img, clf_label = self.__transforms(False, pre_img, post_img, clf_label)
            clf_label = np.asarray(clf_label)
        loc_label = clf_label.copy()
        loc_label[loc_label == 2] = 1
        loc_label[loc_label == 3] = 1

        data_idx = self.data_list[index]
        if self.return_dacutmix_stats:
            return (
                pre_img,
                post_img,
                loc_label,
                clf_label,
                data_idx,
                dacutmix_attempted,
                dacutmix_applied,
            )
        return pre_img, post_img, loc_label, clf_label, data_idx

    def __len__(self):
        return len(self.data_list)


class MultimodalDamageAssessmentDatset_Inference(Dataset):
    def __init__(self, dataset_path, data_list, data_loader=img_loader, suffix='.tif'):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.suffix = suffix

    def __transforms(self, pre_img, post_img):
        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img

    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path, 'pre-event', self.data_list[index] + '_pre_disaster' + self.suffix)
        post_path = os.path.join(self.dataset_path, 'post-event', self.data_list[index] + '_post_disaster'  + self.suffix)
        pre_img = self.loader(pre_path)[:,:,0:3] 
        post_img = self.loader(post_path)  
        
        # pre_img = np.stack((pre_img,)*3, axis=-1)
        post_img = np.stack((post_img,)*3, axis=-1) 
        
        pre_img, post_img = self.__transforms(pre_img, post_img)
    
        data_idx = self.data_list[index]
        return pre_img, post_img, data_idx
    
    def __len__(self):
        return len(self.data_list)
