import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from pytorch_msssim import ssim as ssim_fn


# basicsr/models/losses/composite_loss.py — thêm class mới

class LaplacianLoss(nn.Module):
    """
    Phat L1 giua Laplacian(pred) va Laplacian(target) — ep model tai tao dung
    tan so cao (canh sac net, chi tiet manh) thay vi chi khop mau/cau truc tho.
    """
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer('kernel', kernel.view(1, 1, 3, 3))

    def _laplacian(self, img):
        # img: B x 3 x H x W, [0,1] -> ap dung tren tung kenh RGB rieng, khong gop grayscale
        b, c, h, w = img.shape
        img_flat = img.view(b * c, 1, h, w)
        lap = F.conv2d(img_flat, self.kernel, padding=1)
        return lap.view(b, c, h, w)

    def forward(self, pred, target):
        lap_pred = self._laplacian(pred)
        lap_target = self._laplacian(target)
        return (lap_pred - lap_target).abs().mean()

class EvalAlignedLoss(nn.Module):
    def __init__(self, w_lpips=0.4, w_ssim=0.3, w_psnr=0.3,
                 psnr_norm=40.0, loss_weight=1.0,
                 w_laplacian=0.0):   # mac dinh 0 de khong anh huong ckpt cu neu khong khai bao
        super().__init__()
        self.w_lpips = w_lpips
        self.w_ssim = w_ssim
        self.w_psnr = w_psnr
        self.psnr_norm = psnr_norm
        self.loss_weight = loss_weight
        self.lpips_fn = lpips.LPIPS(net='alex')

        self.w_laplacian = w_laplacian
        self.laplacian_loss_fn = LaplacianLoss() if w_laplacian > 0 else None

        self.last_score = None

    def _psnr_raw(self, pred, target, eps=1e-8):
        mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
        psnr = -10.0 * torch.log10(mse + eps)
        return psnr.mean()

    def forward(self, pred, target):
        pred = torch.clamp(pred, 0.0, 1.0)
        
        lpips_val = self.lpips_fn(pred * 2 - 1, target * 2 - 1).mean()
        ssim_val = ssim_fn(pred, target, data_range=1.0, size_average=True)
        psnr_raw = self._psnr_raw(pred, target)
        psnr_norm_val = torch.clamp(psnr_raw / self.psnr_norm, 0.0, 1.5)

        psnr_term = 1.0 - psnr_norm_val
        eval_loss = (self.w_lpips * lpips_val
                    + self.w_ssim * (1 - ssim_val)
                    + self.w_psnr * psnr_term)

        total_loss = eval_loss
        lap_val = 0.0
        if self.laplacian_loss_fn is not None:
            lap_loss = self.laplacian_loss_fn(pred, target)
            total_loss = total_loss + self.w_laplacian * lap_loss
            lap_val = lap_loss.item()

        with torch.no_grad():
            score = (self.w_lpips * (1 - lpips_val)
                    + self.w_ssim * ssim_val
                    + self.w_psnr * psnr_norm_val)
            self.last_score = score.item()
            self.last_laplacian = lap_val

            # === THEM: luu gia tri THO (chua nhan trong so) de so sanh do lon ===
            self.last_lpips_raw = lpips_val.item()
            self.last_ssim_raw = ssim_val.item()
            self.last_psnr_raw = psnr_raw.item()

        return self.loss_weight * total_loss