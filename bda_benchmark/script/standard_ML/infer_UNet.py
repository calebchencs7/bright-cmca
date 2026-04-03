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


from PIL import Image

class Inference:
    def __init__(self, args):
        self.device = self._resolve_device(args.device)
        print(f'Using device: {self.device}')
        self.model_path = args.model_path
        self.output_dir = args.output_dir
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
        # config = get_config(args)
        num_classes = 4
        # Load dataset
        if self.use_sam_mask:
            dataset = MultimodalDamageAssessmentDatsetWithSAM(
                dataset_path=args.test_dataset_path,
                data_list=args.test_data_list,
                crop_size=1024,
                sam_mask_path=args.sam_mask_dir,
                max_iters=None,
                type='test',
                suffix='.tif',
                sam_mask_suffix=args.sam_mask_suffix,
                sam_mask_threshold=args.sam_mask_threshold
            )
        else:
            dataset = MultimodalDamageAssessmentDatset(args.test_dataset_path, args.test_data_list, 1024, None, 'test', suffix='.tif')
        self.test_loader = DataLoader(dataset, batch_size=1, num_workers=1, drop_last=False)
        
        # Load model
        in_channels = 7 if self.use_sam_mask else 6
        self.model = UNet(in_channels=in_channels, num_classes=num_classes)
        # self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        self.color_map = {
            0: (255, 255, 255),       # No damage - black
            1: (70, 181, 121),     # Minor damage - green
            2: (228, 189, 139),   # Major damage - yellow
            3: (182, 70, 69)      # Destroyed - red
        }
        # Overall evaluator
        self.evaluator = Evaluator(num_class=num_classes)
        self.single_evaluator = Evaluator(num_class=num_classes)
        self.evaluator_clf = Evaluator(num_class=num_classes)

        # Disaster-type-specific evaluators
        self.disaster_type_evaluator_dict = {event: Evaluator(num_class=num_classes) for event in self.get_disaster_types()}
        self.disaster_event_evaluator_dict = {event: Evaluator(num_class=num_classes) for event in self.get_disaster_events()}

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        if not os.path.exists(os.path.join(self.output_dir, 'original')):
            os.makedirs(os.path.join(self.output_dir, 'original'))
        
        if not os.path.exists(os.path.join(self.output_dir, 'colored')):
            os.makedirs(os.path.join(self.output_dir, 'colored'))

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

    def _resolve_guidance_mask(self, sam_mask):
        if sam_mask is None:
            return None

        binary_mask = sam_mask.squeeze(1) >= 0.5
        if self.sam_mode == 'soft':
            return None
        if self.sam_mode == 'hard':
            return binary_mask

        # Hybrid: use hard mask only for samples whose mask area is within a normal range.
        area_ratio = binary_mask.float().mean(dim=(1, 2))
        use_hard = (area_ratio >= self.hybrid_min_area_ratio) & (area_ratio <= self.hybrid_max_area_ratio)
        if torch.all(use_hard):
            return binary_mask

        effective_mask = torch.ones_like(binary_mask, dtype=torch.bool)
        effective_mask[use_hard] = binary_mask[use_hard]
        return effective_mask

    def get_disaster_events(self):
        """Returns a list of disaster events based on filename prefixes."""
        return [
            "turkey-earthquake", "hawaii-wildfire", "morocco-earthquake",
            "haiti-earthquake", "la_palma-volcano", "congo-volcano",
            "beirut-explosion", "bata-explosion", "libya-flood", 
            "noto-earthquake", "marshall-wildfire", "ukraine-conflict", "myanmar-hurricane", "mexico-hurricane"
        ]
    
    def get_disaster_types(self):
        """Returns a list of disaster events based on filename prefixes."""
        return [
            "earthquake", "wildfire", "volcano", "explosion", "flood", 
            "conflict", "hurricane"
        ]

    def apply_tta_inference(self, model, pre_change_imgs, post_change_imgs):
        """
        Performs test-time augmentations (TTA) on the input images and
        fuses the resulting logits. Returns fused logits for damage classification.
        
        Args:
            model (nn.Module): your model in eval mode
            pre_change_imgs (Tensor): shape [B, C, H, W]
            post_change_imgs (Tensor): shape [B, C, H, W]
        
        Returns:
            Tensor: fused logits with shape [B, num_damage_classes, H, W]
        """
        # Collect logits from each transform
        logits_collection = []
        
        # 1) No transform
        output_clf = model(torch.cat([pre_change_imgs, post_change_imgs], dim=1))  # output_clf is [B, num_damage_classes, H, W]
        logits_collection.append(output_clf)

        # 2) Horizontal flip
        output_clf_hf = model(torch.cat([pre_change_imgs.flip(dims=[3]), post_change_imgs.flip(dims=[3])], dim=1))
        # Unflip the output back
        output_clf_hf = output_clf_hf.flip(dims=[3])
        logits_collection.append(output_clf_hf)

        # 3) Vertical flip
        output_clf_vf = model(torch.cat([pre_change_imgs.flip(dims=[2]), post_change_imgs.flip(dims=[2])], dim=1))
        # Unflip the output
        output_clf_vf = output_clf_vf.flip(dims=[2])
        logits_collection.append(output_clf_vf)

        # 4) 90-degree rotation
        pre_90 = torch.rot90(pre_change_imgs, 1, dims=(2, 3))
        post_90 = torch.rot90(post_change_imgs, 1, dims=(2, 3))
        output_clf_90 = model(torch.cat([pre_90, post_90], dim=1))
        output_clf_90 = torch.rot90(output_clf_90, 3, dims=(2, 3))
        logits_collection.append(output_clf_90)

        # 5) 180-degree rotation
        pre_180 = torch.rot90(pre_change_imgs, 2, dims=(2, 3))
        post_180 = torch.rot90(post_change_imgs, 2, dims=(2, 3))
        output_clf_180 = model(torch.cat([pre_180, post_180], dim=1))
        output_clf_180 = torch.rot90(output_clf_180, 2, dims=(2, 3))
        logits_collection.append(output_clf_180)

        # 6) 270-degree rotation
        pre_270 = torch.rot90(pre_change_imgs, 3, dims=(2, 3))
        post_270 = torch.rot90(post_change_imgs, 3, dims=(2, 3))
        output_clf_270 = model(torch.cat([pre_270, post_270], dim=1))
        output_clf_270 = torch.rot90(output_clf_270, 1, dims=(2, 3))
        logits_collection.append(output_clf_270)

        fused_logits = torch.mean(torch.stack(logits_collection, dim=0), dim=0)
        return fused_logits

    def run_inference(self):
        print('Starting inference...')
        self.evaluator.reset()
        
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.test_loader)):
                self.single_evaluator.reset()
                if self.use_sam_mask:
                    pre_change_imgs, post_change_imgs, sam_mask, labels_loc, labels_clf, file_name = data
                    sam_mask = sam_mask.to(self.device).float()
                else:
                    pre_change_imgs, post_change_imgs, labels_loc, labels_clf, file_name = data
                    sam_mask = None
                guidance_mask = self._resolve_guidance_mask(sam_mask)

                pre_change_imgs = pre_change_imgs.to(self.device)
                post_change_imgs = post_change_imgs.to(self.device)
                file_name = file_name[0]
                
                if sam_mask is not None:
                    input_data = torch.cat([pre_change_imgs, post_change_imgs, sam_mask], dim=1)
                else:
                    input_data = torch.cat([pre_change_imgs, post_change_imgs], dim=1)
                output_clf = self.model(input_data)

                output = torch.argmax(output_clf, dim=1).cpu().numpy().astype(np.uint8)
                if guidance_mask is not None:
                    output[~guidance_mask.cpu().numpy()] = 0
                output = output.squeeze(0)

                self.save_colored_map(output, file_name)
                self.save_original_map(output, file_name)

                labels_clf = labels_clf.squeeze().cpu().numpy()
                labels_loc = labels_loc.squeeze().cpu().numpy()

                damage_mask = labels_loc > 0
                if damage_mask.any():
                    output_clf_damage_part = output[damage_mask]
                    labels_clf_damage_part = labels_clf[damage_mask]
                    self.evaluator_clf.add_batch(labels_clf_damage_part, output_clf_damage_part)

                self.evaluator.add_batch(labels_clf, output)
                self.single_evaluator.add_batch(labels_clf, output)
                print(f'{file_name}: {self.single_evaluator.Mean_Intersection_over_Union()}')

                for disaster_type in self.disaster_type_evaluator_dict.keys():
                    if disaster_type in file_name:
                        self.disaster_type_evaluator_dict[disaster_type].add_batch(labels_clf, output)
                        break
                
                for event in self.disaster_event_evaluator_dict.keys():
                    if event in file_name:
                        self.disaster_event_evaluator_dict[event].add_batch(labels_clf, output)
                        break

        self.compute_and_print_overall_metrics()
        self.compute_and_print_disaster_event_metrics()
        self.compute_and_print_disaster_type_metrics()

    def save_original_map(self, prediction, file_name):
        output_path = os.path.join(self.output_dir, 'original', file_name + '_building_damage.png')
        Image.fromarray(prediction).save(output_path)

    def save_colored_map(self, prediction, file_name):
        color_map_img = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
        for cls, color in self.color_map.items():
            color_map_img[prediction == cls] = color
        output_path = os.path.join(self.output_dir, 'colored', file_name + '_building_damage.png')
        Image.fromarray(color_map_img).save(output_path)

    def compute_and_print_overall_metrics(self):
        pixel_accuracy = self.evaluator.Pixel_Accuracy()
        mean_iou = self.evaluator.Mean_Intersection_over_Union()
        
        print("\nOverall Metrics:")
        print(f'Pixel Accuracy: {pixel_accuracy * 100:.2f}%')
        print(f'Mean IoU: {mean_iou * 100:.2f}%')
        print(f'IoU: {self.evaluator.Intersection_over_Union()}')
        damage_f1_score = self.evaluator_clf.Damage_F1_score()
        harmonic_mean_f1 = self._safe_hmean(damage_f1_score) * 100
        print(f'F1 Score: {harmonic_mean_f1:.2f}%')

    def compute_and_print_disaster_type_metrics(self):
        print("\nPer-Disaster Type mIoU:")
        average_mIoU = 0
        for disaster_type, evaluator in self.disaster_type_evaluator_dict.items():
            mean_iou = evaluator.Mean_Intersection_over_Union()
            iou_per_class = evaluator.Intersection_over_Union()
            average_mIoU += mean_iou
            print(f"{disaster_type}: mIoU = {mean_iou * 100:.2f}%, IoU = {iou_per_class * 100}")
        print(f"Average mIoU = {average_mIoU / 7 * 100:.2f}%")
    
    def compute_and_print_disaster_event_metrics(self):
        print("\nPer-Event Type mIoU:")
        average_mIoU = 0
        for event, evaluator in self.disaster_event_evaluator_dict.items():
            mean_iou = evaluator.Mean_Intersection_over_Union()
            iou_per_class = evaluator.Intersection_over_Union()
            average_mIoU += mean_iou
            print(f"{event}: mIoU = {mean_iou * 100:.2f}%, IoU = {iou_per_class * 100}")
        print(f"Average mIoU = {average_mIoU / 14 * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference on BRIGHT")
    parser.add_argument('--model_path', type=str, default='BRIGHT')
    parser.add_argument('--test_dataset_path', type=str)
    parser.add_argument('--test_data_list_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Inference device. Use auto to prefer CUDA, then MPS, then CPU.')
    parser.add_argument('--sam_mask_dir', type=str, default=None,
                        help='Directory of SAM building masks. File naming: <tile_id>_building_mask<sam_mask_suffix>.')
    parser.add_argument('--sam_mask_suffix', type=str, default='.png',
                        help='Suffix of SAM mask files.')
    parser.add_argument('--sam_mask_threshold', type=int, default=127,
                        help='Threshold for binarizing SAM masks.')
    parser.add_argument('--sam_mode', type=str, default='hard', choices=['soft', 'hard', 'hybrid'],
                        help='How SAM masks are used: soft=as feature only, hard=mask constraint, hybrid=auto fallback by area.')
    parser.add_argument('--hybrid_min_area_ratio', type=float, default=0.001,
                        help='Hybrid lower bound of mask area ratio to apply hard mask.')
    parser.add_argument('--hybrid_max_area_ratio', type=float, default=0.6,
                        help='Hybrid upper bound of mask area ratio to apply hard mask.')

    args = parser.parse_args()
    
    with open(args.test_data_list_path, "r") as f:
        test_data_list = [data_name.strip() for data_name in f]
    args.test_data_list = test_data_list

    inference = Inference(args)
    inference.run_inference()
