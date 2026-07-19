# basicsr\models\nafnet_jpeg_model.py
import torch
from collections import OrderedDict
from torch.cuda.amp import autocast, GradScaler
from basicsr.models.image_restoration_model import ImageRestorationModel
from DiffJPEG.DiffJPEG import DiffJPEG


class NAFNetJPEGModel(ImageRestorationModel):
    def __init__(self, opt):
        super().__init__(opt)
        jpeg_opt = opt.get('diffjpeg', {})
        gt_size = opt['datasets']['train']['gt_size']
        self.diffjpeg = DiffJPEG(
            height=gt_size, width=gt_size,
            differentiable=True, quality=jpeg_opt.get('quality', 95)
        ).to(self.device)

        self.use_amp = opt['train'].get('use_amp', True)
        self.use_grad_checkpoint = opt['train'].get('use_grad_checkpoint', False)
        self.scaler = GradScaler(enabled=self.use_amp)
        self.grad_clip_norm = opt['train'].get('grad_clip_norm', 0.01)

    def optimize_parameters(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()

        with autocast(enabled=self.use_amp):
            if self.use_grad_checkpoint:
                preds = torch.utils.checkpoint.checkpoint(self.net_g, self.lq, use_reentrant=False)
            else:
                preds = self.net_g(self.lq)
            if not isinstance(preds, list):
                preds = [preds]
            self.output = preds[-1]
            enhanced = torch.clamp(self.output, 0.0, 1.0)

        pred_jpeg = self.diffjpeg(enhanced.float())

        l_total = 0
        loss_dict = OrderedDict()
        if self.cri_pix:
            l_pix = self.cri_pix(pred_jpeg, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        self.scaler.scale(l_total).backward()

        if self.grad_clip_norm > 0:
            self.scaler.unscale_(self.optimizer_g)
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.grad_clip_norm)

        self.scaler.step(self.optimizer_g)
        self.scaler.update()
        self.log_dict = self.reduce_loss_dict(loss_dict)