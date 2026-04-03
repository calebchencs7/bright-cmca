import os
import sys
import argparse
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.make_data_loader import MultimodalDamageAssessmentDatset, MultimodalDamageAssessmentDatsetWithSAM
from model.UNet import UNet
from model.SiamCRNN import SiamCRNN
from datetime import datetime

from util_func.metrics import Evaluator
import util_func.lovasz_loss as L
from util_func.training_curve import TrainingCurveRecorder


class Trainer(object):
    """
    Trainer class that encapsulates model, optimizer, and data loading.
    It can train the model and evaluate its performance on a holdout set.
    """

    def __init__(self, args):
        self.args = args
        self.device = self._resolve_device(args.device)
        print(f'Using device: {self.device}')

        self.evaluator_loc = Evaluator(num_class=2)
        self.evaluator_clf = Evaluator(num_class=4)
        self.evaluator_total = Evaluator(num_class=4)

        self.use_sam_mask = args.sam_mask_dir is not None and args.sam_mask_dir != ""
        self.sam_mode = args.sam_mode.lower()
        self.hybrid_min_area_ratio = args.hybrid_min_area_ratio
        self.hybrid_max_area_ratio = args.hybrid_max_area_ratio

        if self.sam_mode not in ['soft', 'hard', 'hybrid']:
            raise ValueError(f'Unsupported sam_mode: {self.sam_mode}')
        if self.sam_mode == 'hybrid' and self.hybrid_min_area_ratio >= self.hybrid_max_area_ratio:
            raise ValueError('hybrid_min_area_ratio must be smaller than hybrid_max_area_ratio.')

        if self.use_sam_mask:
            print(f'SAM guidance enabled. mode={self.sam_mode}')
        else:
            print('SAM guidance disabled (no sam_mask_dir provided).')

        in_channels = 7 if self.use_sam_mask else 6
        self.deep_model = UNet(in_channels=in_channels, num_classes=4)
        # self.deep_model = SiamCRNN()
        self.deep_model = self.deep_model.to(self.device)

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_save_path = os.path.join(args.model_param_path, args.dataset, args.model_type + '_' + now_str)
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        self.curve_recorder = TrainingCurveRecorder(self.model_save_path)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=self.device)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.optim = optim.AdamW(
            self.deep_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        self.class_weights = None
        if args.class_weights is not None:
            weights = [float(x) for x in args.class_weights.split(',')]
            if len(weights) != 4:
                raise ValueError("class_weights must have 4 comma-separated values (for 4 classes).")
            self.class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

    @staticmethod
    def _resolve_device(device_arg):
        if device_arg == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            return torch.device('cpu')

        if device_arg == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is not available on this machine.')
            return torch.device('cuda')

        if device_arg == 'mps':
            if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                raise RuntimeError('MPS is not available on this machine.')
            return torch.device('mps')

        return torch.device('cpu')

    @staticmethod
    def _safe_hmean(scores, eps=1e-6):
        scores = np.asarray(scores, dtype=np.float32)
        scores = scores[np.isfinite(scores)]
        if scores.size == 0:
            return 0.0
        scores = np.where(scores <= 0, eps, scores)
        return len(scores) / np.sum(1.0 / scores)

    def _create_dataset(self, dataset_path, data_name_list, crop_size, max_iters, data_type):
        if self.use_sam_mask:
            return MultimodalDamageAssessmentDatsetWithSAM(
                dataset_path=dataset_path,
                data_list=data_name_list,
                crop_size=crop_size,
                sam_mask_path=self.args.sam_mask_dir,
                max_iters=max_iters,
                type=data_type,
                suffix='.tif',
                sam_mask_suffix=self.args.sam_mask_suffix,
                sam_mask_threshold=self.args.sam_mask_threshold
            )

        return MultimodalDamageAssessmentDatset(
            dataset_path=dataset_path,
            data_list=data_name_list,
            crop_size=crop_size,
            max_iters=max_iters,
            type=data_type,
            suffix='.tif'
        )

    def _resolve_guidance_mask(self, sam_mask):
        if sam_mask is None:
            return None

        binary_mask = sam_mask.squeeze(1) >= 0.5
        if self.sam_mode == 'soft':
            return None
        if self.sam_mode == 'hard':
            return binary_mask

        area_ratio = binary_mask.float().mean(dim=(1, 2))
        use_hard = (area_ratio >= self.hybrid_min_area_ratio) & (area_ratio <= self.hybrid_max_area_ratio)
        if torch.all(use_hard):
            return binary_mask

        effective_mask = torch.ones_like(binary_mask, dtype=torch.bool)
        effective_mask[use_hard] = binary_mask[use_hard]
        return effective_mask

    def _prepare_batch(self, data):
        if self.use_sam_mask:
            pre_change_imgs, post_change_imgs, sam_mask, labels_loc, labels_clf, _ = data
            sam_mask = sam_mask.to(self.device).float()
        else:
            pre_change_imgs, post_change_imgs, labels_loc, labels_clf, _ = data
            sam_mask = None

        pre_change_imgs = pre_change_imgs.to(self.device)
        post_change_imgs = post_change_imgs.to(self.device)
        labels_loc = labels_loc.to(self.device).long()
        labels_clf = labels_clf.to(self.device).long()

        if sam_mask is not None:
            input_data = torch.cat([pre_change_imgs, post_change_imgs, sam_mask], dim=1)
            guidance_mask = self._resolve_guidance_mask(sam_mask)
            if guidance_mask is not None:
                labels_for_loss = labels_clf.clone()
                labels_for_loss[~guidance_mask] = 255
            else:
                labels_for_loss = labels_clf
        else:
            input_data = torch.cat([pre_change_imgs, post_change_imgs], dim=1)
            labels_for_loss = labels_clf
            guidance_mask = None

        return input_data, labels_loc, labels_clf, labels_for_loss, sam_mask, guidance_mask

    def training(self):
        best_mIoU = 0.0
        best_round = []

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        train_dataset = self._create_dataset(
            dataset_path=self.args.train_dataset_path,
            data_name_list=self.args.train_data_name_list,
            crop_size=self.args.crop_size,
            max_iters=self.args.max_iters,
            data_type='train'
        )
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=False
        )

        elem_num = len(train_data_loader)
        train_enumerator = enumerate(train_data_loader)

        for _ in tqdm(range(elem_num)):
            itera, data = train_enumerator.__next__()
            input_data, labels_loc, labels_clf, labels_for_loss, _, _ = self._prepare_batch(data)

            valid_labels_clf = (labels_for_loss != 255).any().item()
            if not valid_labels_clf:
                continue

            output_clf = self.deep_model(input_data)

            self.optim.zero_grad()
            ce_loss_clf = F.cross_entropy(output_clf, labels_for_loss, ignore_index=255, weight=self.class_weights)
            lovasz_loss_clf = L.lovasz_softmax(F.softmax(output_clf, dim=1), labels_for_loss, ignore=255)
            final_loss = ce_loss_clf + 0.75 * lovasz_loss_clf

            final_loss.backward()
            self.optim.step()

            if (itera + 1) % 10 == 0:
                print(f'iter is {itera + 1}, classification loss is {final_loss.item()}')
                self.curve_recorder.add_train_loss(itera + 1, final_loss.item())

                if (itera + 1) % 500 == 0:
                    self.deep_model.eval()
                    loc_f1_score_val, harmonic_mean_f1_val, final_OA_val, mIoU_val, IoU_of_each_class_val = self.validation()
                    loc_f1_score_test, harmonic_mean_f1_test, final_OA_test, mIoU_test, IoU_of_each_class_test = self.test()

                    self.curve_recorder.add_eval_metrics(itera + 1, 'val', final_OA_val * 100, mIoU_val * 100)
                    self.curve_recorder.add_eval_metrics(itera + 1, 'test', final_OA_test * 100, mIoU_test * 100)

                    if mIoU_val > best_mIoU:
                        torch.save(self.deep_model.state_dict(), os.path.join(self.model_save_path, 'best_model.pth'))
                        best_mIoU = mIoU_val
                        best_round = {
                            'best iter': itera + 1,
                            'loc f1 (val)': loc_f1_score_val * 100,
                            'clf f1 (val)': harmonic_mean_f1_val * 100,
                            'OA (val)': final_OA_val * 100,
                            'mIoU (val)': mIoU_val * 100,
                            'sub class IoU (val)': IoU_of_each_class_val * 100,
                            'loc f1 (test)': loc_f1_score_test * 100,
                            'clf f1 (test)': harmonic_mean_f1_test * 100,
                            'OA (test)': final_OA_test * 100,
                            'mIoU (test)': mIoU_test * 100,
                            'sub class IoU (test)': IoU_of_each_class_test * 100
                        }
                    self.deep_model.train()

        self.curve_recorder.save()
        print(f'Training curve files are saved in {self.model_save_path}')
        print('The accuracy of the best round is ', best_round)

    def validation(self):
        print('---------starting validation-----------')
        self.evaluator_total.reset()
        self.evaluator_loc.reset()
        self.evaluator_clf.reset()

        val_dataset = self._create_dataset(
            dataset_path=self.args.val_dataset_path,
            data_name_list=self.args.val_data_name_list,
            crop_size=1024,
            max_iters=None,
            data_type='test'
        )
        val_data_loader = DataLoader(val_dataset, batch_size=self.args.eval_batch_size, num_workers=1, drop_last=False)

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        with torch.no_grad():
            for _, data in enumerate(val_data_loader):
                input_data, labels_loc, labels_clf, _, _, guidance_mask = self._prepare_batch(data)
                output_clf = self.deep_model(input_data)

                labels_loc = labels_loc.cpu().numpy()
                output_clf = output_clf.data.cpu().numpy()
                output_clf = np.argmax(output_clf, axis=1)

                if guidance_mask is not None:
                    output_clf[~guidance_mask.cpu().numpy()] = 0

                labels_clf = labels_clf.cpu().numpy()
                output_loc = output_clf.copy()
                output_loc[output_loc > 0] = 1

                self.evaluator_loc.add_batch(labels_loc, output_loc)
                output_clf_damage_part = output_clf[labels_loc > 0]
                labels_clf_damage_part = labels_clf[labels_loc > 0]
                self.evaluator_clf.add_batch(labels_clf_damage_part, output_clf_damage_part)
                self.evaluator_total.add_batch(labels_clf, output_clf)

        loc_f1_score = self.evaluator_loc.Pixel_F1_score()
        damage_f1_score = self.evaluator_clf.Damage_F1_score()
        harmonic_mean_f1 = self._safe_hmean(damage_f1_score)
        final_OA = self.evaluator_total.Pixel_Accuracy()
        IoU_of_each_class = self.evaluator_total.Intersection_over_Union()
        mIoU = self.evaluator_total.Mean_Intersection_over_Union()
        print(f'OA is {100 * final_OA}, mIoU is {100 * mIoU}, sub class IoU is {100 * IoU_of_each_class}')
        return loc_f1_score, harmonic_mean_f1, final_OA, mIoU, IoU_of_each_class

    def test(self):
        print('---------starting testing-----------')
        self.evaluator_total.reset()
        self.evaluator_loc.reset()
        self.evaluator_clf.reset()

        test_dataset = self._create_dataset(
            dataset_path=self.args.test_dataset_path,
            data_name_list=self.args.test_data_name_list,
            crop_size=1024,
            max_iters=None,
            data_type='test'
        )
        test_data_loader = DataLoader(test_dataset, batch_size=self.args.eval_batch_size, num_workers=1, drop_last=False)

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        with torch.no_grad():
            for _, data in enumerate(test_data_loader):
                input_data, labels_loc, labels_clf, _, _, guidance_mask = self._prepare_batch(data)
                output_clf = self.deep_model(input_data)

                labels_loc = labels_loc.cpu().numpy()
                output_clf = output_clf.data.cpu().numpy()
                output_clf = np.argmax(output_clf, axis=1)

                if guidance_mask is not None:
                    output_clf[~guidance_mask.cpu().numpy()] = 0

                labels_clf = labels_clf.cpu().numpy()
                output_loc = output_clf.copy()
                output_loc[output_loc > 0] = 1

                self.evaluator_loc.add_batch(labels_loc, output_loc)
                output_clf_damage_part = output_clf[labels_loc > 0]
                labels_clf_damage_part = labels_clf[labels_loc > 0]
                self.evaluator_clf.add_batch(labels_clf_damage_part, output_clf_damage_part)
                self.evaluator_total.add_batch(labels_clf, output_clf)

        loc_f1_score = self.evaluator_loc.Pixel_F1_score()
        damage_f1_score = self.evaluator_clf.Damage_F1_score()
        harmonic_mean_f1 = self._safe_hmean(damage_f1_score)
        final_OA = self.evaluator_total.Pixel_Accuracy()
        IoU_of_each_class = self.evaluator_total.Intersection_over_Union()
        mIoU = self.evaluator_total.Mean_Intersection_over_Union()
        print(f'OA is {100 * final_OA}, mIoU is {100 * mIoU}, sub class IoU is {100 * IoU_of_each_class}')
        return loc_f1_score, harmonic_mean_f1, final_OA, mIoU, IoU_of_each_class


def main():
    parser = argparse.ArgumentParser(description="Training on BRIGHT dataset")

    parser.add_argument('--dataset', type=str, default='BRIGHT')
    parser.add_argument('--train_dataset_path', type=str)
    parser.add_argument('--train_data_list_path', type=str)
    parser.add_argument('--val_dataset_path', type=str)
    parser.add_argument('--val_data_list_path', type=str)
    parser.add_argument('--test_dataset_path', type=str)
    parser.add_argument('--test_data_list_path', type=str)

    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--crop_size', type=int)

    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--val_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)

    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--model_param_path', type=str, default='/home/songjian/project/BRIGHT/dfc25_benchmark/saved_weights')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--num_workers', type=int)

    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Training device. Use auto to prefer CUDA, then MPS, then CPU.')
    parser.add_argument('--class_weights', type=str, default=None,
                        help='Comma-separated weights for 4 classes, e.g. 1,1,2,2')

    parser.add_argument('--sam_mask_dir', type=str, default=None,
                        help='Directory of SAM building masks. File naming: <tile_id>_building_mask<sam_mask_suffix>.')
    parser.add_argument('--sam_mask_suffix', type=str, default='.png',
                        help='Suffix of SAM mask files.')
    parser.add_argument('--sam_mask_threshold', type=int, default=127,
                        help='Threshold for binarizing SAM masks.')

    parser.add_argument('--sam_mode', type=str, default='soft', choices=['soft', 'hard', 'hybrid'],
                        help='soft=feature-only guidance, hard=mask constraint, hybrid=area-based fallback.')
    parser.add_argument('--hybrid_min_area_ratio', type=float, default=0.001,
                        help='Hybrid lower bound of mask area ratio to apply hard mask.')
    parser.add_argument('--hybrid_max_area_ratio', type=float, default=0.6,
                        help='Hybrid upper bound of mask area ratio to apply hard mask.')

    args = parser.parse_args()

    with open(args.train_data_list_path, "r") as f:
        args.train_data_name_list = [data_name.strip() for data_name in f]

    with open(args.val_data_list_path, "r") as f:
        args.val_data_name_list = [data_name.strip() for data_name in f]

    with open(args.test_data_list_path, "r") as f:
        args.test_data_name_list = [data_name.strip() for data_name in f]

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()
