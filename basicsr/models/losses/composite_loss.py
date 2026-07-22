import torch
import torch.nn as nn
import lpips
from pytorch_msssim import ssim as ssim_fn


class EvalAlignedLoss(nn.Module):
    def __init__(self, w_lpips=0.4, w_ssim=0.3, w_psnr=0.3,
                 psnr_norm=40.0, loss_weight=1.0):
        super().__init__()
        self.w_lpips = w_lpips
        self.w_ssim = w_ssim
        self.w_psnr = w_psnr
        self.psnr_norm = psnr_norm
        self.loss_weight = loss_weight
        self.lpips_fn = lpips.LPIPS(net='alex')
        self.lpips_fn.eval()
        for p in self.lpips_fn.parameters():
            p.requires_grad = False

    def _psnr_raw(self, pred, target, eps=1e-8):
        mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
        psnr = -10.0 * torch.log10(mse + eps)
        return psnr.mean()

    def forward(self, pred, target):
        pred = torch.clamp(pred, 0.0, 1.0)

        lpips_val = self.lpips_fn(pred * 2 - 1, target * 2 - 1).mean()
        ssim_val = ssim_fn(pred, target, data_range=1.0, size_average=True)
        psnr_raw = self._psnr_raw(pred, target)
        psnr_term = 1.0 - torch.clamp(psnr_raw / self.psnr_norm, 0.0, 1.5)

        loss = (self.w_lpips * lpips_val
                + self.w_ssim * (1 - ssim_val)
                + self.w_psnr * psnr_term)

        if torch.isnan(loss) or torch.isinf(loss):
            print("=== NaN/Inf DETECTED ===")
            print(f"lpips_val: {lpips_val.item()}")
            print(f"ssim_val: {ssim_val.item()}")
            print(f"psnr_raw: {psnr_raw.item()}")
            print(f"psnr_term: {psnr_term.item()}")
            print(f"pred range: [{pred.min().item()}, {pred.max().item()}]")
            print(f"target range: [{target.min().item()}, {target.max().item()}]")
            print(f"pred has nan: {torch.isnan(pred).any().item()}")
            print(f"target has nan: {torch.isnan(target).any().item()}")
            raise RuntimeError("Stopping to inspect NaN source")

        return self.loss_weight * loss