# basicsr\data\paired_png16_dataset.py
from basicsr.data.paired_image_dataset import PairedImageDataset
from basicsr.data.transforms import paired_random_crop, augment
import cv2, numpy as np, torch


class Paired16BitDataset(PairedImageDataset):
    def __getitem__(self, index):
        scale = self.opt['scale']
        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']

        img_lq = cv2.imread(lq_path, cv2.IMREAD_UNCHANGED)
        assert img_lq is not None, f"Cannot read: {lq_path}"
        assert img_lq.dtype == np.uint16, f"Expect uint16 PNG, got {img_lq.dtype} at {lq_path}"
        img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0

        img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        assert img_gt is not None, f"Cannot read GT jpg: {gt_path}"
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.opt['phase'] == 'train':
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, self.opt['gt_size'], scale, gt_path)
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])

        img_gt = torch.from_numpy(img_gt.transpose(2, 0, 1).copy()).float()
        img_lq = torch.from_numpy(img_lq.transpose(2, 0, 1).copy()).float()

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}