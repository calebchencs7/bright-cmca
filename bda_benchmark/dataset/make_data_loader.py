import argparse
import os
import random

import imageio
import numpy as np
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
import cv2
import dataset.imutils as imutils
import matplotlib.pyplot as plt
from torch.utils import data
from PIL import Image


def img_loader(path):
    img = np.array(imageio.imread(path), np.float32)
    return img


def mask_loader(path):
    mask = np.array(imageio.imread(path), np.float32)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask



class MultimodalDamageAssessmentDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader, suffix='.tif'):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type
        self.suffix = suffix

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, label):
        if aug:
            pre_img, post_img, label = imutils.random_crop(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path, 'pre-event', self.data_list[index] + '_pre_disaster' + self.suffix)
        post_path = os.path.join(self.dataset_path, 'post-event', self.data_list[index] + '_post_disaster'  + self.suffix)
        label_path = os.path.join(self.dataset_path, 'target', self.data_list[index] + '_building_damage'  + self.suffix)
        pre_img = self.loader(pre_path)[:,:,0:3] 
        post_img = self.loader(post_path)  
        
        # pre_img = np.stack((pre_img,)*3, axis=-1)
        post_img = np.stack((post_img,)*3, axis=-1)
        clf_label = self.loader(label_path)
        

        if 'train' in self.data_pro_type:
            pre_img, post_img, clf_label = self.__transforms(True, pre_img, post_img, clf_label)
        else:
            pre_img, post_img, clf_label = self.__transforms(False, pre_img, post_img, clf_label)
            clf_label = np.asarray(clf_label)
        loc_label = clf_label.copy()
        loc_label[loc_label == 2] = 1
        loc_label[loc_label == 3] = 1

        data_idx = self.data_list[index]
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


class MultimodalDamageAssessmentDatsetWithSAM(Dataset):
    def __init__(
        self,
        dataset_path,
        data_list,
        crop_size,
        sam_mask_path,
        max_iters=None,
        type='train',
        data_loader=img_loader,
        mask_data_loader=mask_loader,
        suffix='.tif',
        sam_mask_suffix='.png',
        sam_mask_threshold=127
    ):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.mask_loader = mask_data_loader
        self.type = type
        self.data_pro_type = self.type
        self.suffix = suffix
        self.sam_mask_suffix = sam_mask_suffix
        self.sam_mask_threshold = sam_mask_threshold

        if sam_mask_path is None:
            raise ValueError("sam_mask_path must be provided when using MultimodalDamageAssessmentDatsetWithSAM.")
        self.sam_mask_path = sam_mask_path if os.path.isabs(sam_mask_path) else os.path.join(dataset_path, sam_mask_path)

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def _load_paths(self, idx):
        data_name = self.data_list[idx]
        pre_path = os.path.join(self.dataset_path, 'pre-event', data_name + '_pre_disaster' + self.suffix)
        post_path = os.path.join(self.dataset_path, 'post-event', data_name + '_post_disaster' + self.suffix)
        label_path = os.path.join(self.dataset_path, 'target', data_name + '_building_damage' + self.suffix)
        sam_mask_path = os.path.join(self.sam_mask_path, data_name + '_building_mask' + self.sam_mask_suffix)
        return data_name, pre_path, post_path, label_path, sam_mask_path

    @staticmethod
    def _crop_with_mask(pre_img, post_img, label, sam_mask, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
        h, w = label.shape

        H = max(crop_size, h)
        W = max(crop_size, w)

        pad_pre_image = np.zeros((H, W, 3), dtype=np.float32)
        pad_post_image = np.zeros((H, W, 3), dtype=np.float32)
        pad_label = np.ones((H, W), dtype=np.float32) * ignore_index
        pad_sam_mask = np.zeros((H, W), dtype=np.float32)

        pad_pre_image[:, :, 0] = mean_rgb[0]
        pad_pre_image[:, :, 1] = mean_rgb[1]
        pad_pre_image[:, :, 2] = mean_rgb[2]
        pad_post_image[:, :, 0] = mean_rgb[0]
        pad_post_image[:, :, 1] = mean_rgb[1]
        pad_post_image[:, :, 2] = mean_rgb[2]

        h_pad = int(np.random.randint(H - h + 1))
        w_pad = int(np.random.randint(W - w + 1))
        pad_pre_image[h_pad:(h_pad + h), w_pad:(w_pad + w), :] = pre_img
        pad_post_image[h_pad:(h_pad + h), w_pad:(w_pad + w), :] = post_img
        pad_label[h_pad:(h_pad + h), w_pad:(w_pad + w)] = label
        pad_sam_mask[h_pad:(h_pad + h), w_pad:(w_pad + w)] = sam_mask

        h_start, h_end, w_start, w_end = 0, crop_size, 0, crop_size
        for _ in range(10):
            h_start = random.randrange(0, H - crop_size + 1, 1)
            h_end = h_start + crop_size
            w_start = random.randrange(0, W - crop_size + 1, 1)
            w_end = w_start + crop_size

            temp_label = pad_label[h_start:h_end, w_start:w_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < 0.75:
                break

        pre_img = pad_pre_image[h_start:h_end, w_start:w_end, :]
        post_img = pad_post_image[h_start:h_end, w_start:w_end, :]
        label = pad_label[h_start:h_end, w_start:w_end]
        sam_mask = pad_sam_mask[h_start:h_end, w_start:w_end]
        return pre_img, post_img, label, sam_mask

    def _transforms(self, aug, pre_img, post_img, label, sam_mask):
        if aug:
            pre_img, post_img, label, sam_mask = self._crop_with_mask(pre_img, post_img, label, sam_mask, self.crop_size)
            if random.random() > 0.5:
                pre_img = np.fliplr(pre_img)
                post_img = np.fliplr(post_img)
                label = np.fliplr(label)
                sam_mask = np.fliplr(sam_mask)
            if random.random() > 0.5:
                pre_img = np.flipud(pre_img)
                post_img = np.flipud(post_img)
                label = np.flipud(label)
                sam_mask = np.flipud(sam_mask)
            k = random.randrange(3) + 1
            pre_img = np.rot90(pre_img, k).copy()
            post_img = np.rot90(post_img, k).copy()
            label = np.rot90(label, k).copy()
            sam_mask = np.rot90(sam_mask, k).copy()

        pre_img = imutils.normalize_img(pre_img)
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)
        post_img = np.transpose(post_img, (2, 0, 1))

        # Soft mask processing: instead of hard {0,1} binarization, produce a
        # [0,1] confidence map with smooth boundaries.  This gives the SGR module
        # gradient information at mask edges so it can learn to discount uncertain
        # regions rather than treating every SAM pixel with equal certainty.
        #
        # Pipeline:
        #   1. Normalize raw pixel values to [0, 1]
        #   2. Binarize at threshold to get the core mask
        #   3. Compute a signed distance transform from the boundary
        #   4. Apply a sigmoid to the distance → smooth [0, 1] transition
        #      (sigma controls transition width; 3px ≈ ~6px soft band)
        sam_binary = (sam_mask > self.sam_mask_threshold).astype(np.uint8)

        # Distance transform: positive inside, negative outside
        dist_inside = cv2.distanceTransform(sam_binary, cv2.DIST_L2, 5)
        dist_outside = cv2.distanceTransform(1 - sam_binary, cv2.DIST_L2, 5)
        signed_dist = dist_inside - dist_outside

        # Sigmoid softening (sigma=3 → ~6px transition band)
        sigma = 3.0
        sam_mask = 1.0 / (1.0 + np.exp(-signed_dist / sigma))
        sam_mask = sam_mask.astype(np.float32)

        sam_mask = np.expand_dims(sam_mask, axis=0)
        return pre_img, post_img, label, sam_mask

    def __getitem__(self, index):
        data_idx, pre_path, post_path, label_path, sam_mask_path = self._load_paths(index)

        if not os.path.exists(sam_mask_path):
            raise FileNotFoundError(f"SAM mask not found: {sam_mask_path}")

        pre_img = self.loader(pre_path)[:, :, 0:3]
        post_img = self.loader(post_path)
        post_img = np.stack((post_img,) * 3, axis=-1)
        clf_label = self.loader(label_path)
        sam_mask = self.mask_loader(sam_mask_path)

        if 'train' in self.data_pro_type:
            pre_img, post_img, clf_label, sam_mask = self._transforms(True, pre_img, post_img, clf_label, sam_mask)
        else:
            pre_img, post_img, clf_label, sam_mask = self._transforms(False, pre_img, post_img, clf_label, sam_mask)
            clf_label = np.asarray(clf_label)

        loc_label = clf_label.copy()
        loc_label[loc_label == 2] = 1
        loc_label[loc_label == 3] = 1
        return pre_img, post_img, sam_mask, loc_label, clf_label, data_idx

    def __len__(self):
        return len(self.data_list)
