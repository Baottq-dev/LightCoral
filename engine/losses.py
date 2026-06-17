# engine/losses.py
# L_deg cho PG-DAM (module 4) + SCDetectionModel cong loss phu vao detect loss.

import torch
import torch.nn.functional as F

from ultralytics.nn.tasks import DetectionModel

from models.pg_dam import PGDAM_FiLM


def find_pgdam(model):
    """Tim instance PGDAM_FiLM dau tien trong model (None neu khong co)."""
    for m in model.modules():
        if isinstance(m, PGDAM_FiLM):
            return m
    return None


class DegradationLoss:
    """L_deg = w * SmoothL1(z_hat, z_gt).

    - z_hat: PGDAM_FiLM.last_z, duoc gan trong forward gan nhat.
    - z_gt:  batch["z_gt"] tu augment/physics_degradation (chuan hoa [0,1]^7).
    - Smooth-L1 thay vi L2 thuan: ben hon voi sample t_mean gan bien.
    """

    def __init__(self, weight=0.05):
        self.weight = float(weight)

    def __call__(self, det_model, batch):
        m = find_pgdam(det_model)
        if m is None or m.last_z is None:
            return None
        z_gt = batch.get("z_gt")
        if z_gt is None:
            return None  # vd: val loop khong co physics aug
        z_hat = m.last_z
        if z_gt.shape[0] != z_hat.shape[0]:
            return None  # an toan khi batch cuoi/val co kich thuoc khac
        z_gt = z_gt.to(device=z_hat.device, dtype=z_hat.dtype)
        return self.weight * F.smooth_l1_loss(z_hat, z_gt)


class SCDetectionModel(DetectionModel):
    """DetectionModel + L_deg.

    Ultralytics v8DetectionLoss tra ve (loss.sum() * batch_size, loss_items).
    De giu cung thang do, L_deg cung duoc nhan voi batch_size truoc khi cong.
    deg_loss duoc train.py gan khi --modules chua 4; mac dinh None (tat).
    """

    deg_loss = None  # type: DegradationLoss | None

    def loss(self, batch, preds=None):
        loss, loss_items = super().loss(batch, preds)
        if self.deg_loss is not None and self.training:
            extra = self.deg_loss(self, batch)
            if extra is not None:
                bs = batch["img"].shape[0]
                loss = loss + extra * bs
        return loss, loss_items