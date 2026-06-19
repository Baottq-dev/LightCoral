# benchmark/scoraldet/apt_loss.py
# Module (3b): APT loss - SCoralDet Eq 4-7.
#   - v8APTDetectionLoss: v8DetectionLoss + thay assigner = APTAssigner (Eq 2-3). [LUON BAT]
#   - v10APTDetectionLoss: goi 2 head YOLOv10 (one2many topk=10 + one2one topk=1).
#   - SoftClsLoss (Eq 5), SoftCenterConfLoss (Eq 6-7): module doc lap, co the bat them
#     (paper underspec cach nhung => mac dinh TAT de train on dinh; xem README de wiring).

import torch
import torch.nn as nn

from ultralytics.utils.loss import v8DetectionLoss

# ho tro chay ca khi import qua package (-m) lan flat (python benchmark/scoraldet/apt_loss.py)
try:
    from benchmark.scoraldet.apt_assigner import APTAssigner
except ImportError:
    from apt_assigner import APTAssigner


class SoftClsLoss(nn.Module):
    """Soft-label classification loss (Eq 5): L = (l_soft - p)^2 * CE(p, l_soft).

    l_soft: soft label (vi du = IoU cua anchor positive) thay nhan nhi phan.
    p     : xac suat du doan (sau sigmoid). Trong so (l_soft-p)^2 kieu quality-focal.
    """

    def __init__(self, reduction="none"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_logits, l_soft):
        p = pred_logits.sigmoid()
        ce = nn.functional.binary_cross_entropy_with_logits(pred_logits, l_soft, reduction="none")
        loss = (l_soft - p).pow(2) * ce
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "mean":
            return loss.mean()
        return loss


class SoftCenterConfLoss(nn.Module):
    """Soft center region confidence loss (Eq 6-7).

        D(a, b)  = sum_i (a_i - b_i)^n           (n=2 => Euclidean binh phuong)
        L_conf   = chi^(|x_pred - x_gt| - delta) + D(w_pred, w_gt) + D(h_pred, h_gt)

    Phat lech tam (so hang mu chi) + lech w/h. chi=10, delta=3 (paper Sec 4.2).
    Nhan toa do/kich thuoc da chuan hoa [0,1] de tranh tran so mu.
    """

    def __init__(self, chi=10.0, delta=3.0, n=2):
        super().__init__()
        self.chi, self.delta, self.n = float(chi), float(delta), int(n)

    def _D(self, a, b):
        return (a - b).abs().pow(self.n).sum(dim=-1)

    def forward(self, xy_pred, wh_pred, xy_gt, wh_gt):
        center_dist = (xy_pred - xy_gt).abs().sum(dim=-1)              # |x_pred - x_gt|
        center_term = torch.pow(self.chi, center_dist - self.delta)   # chi^(.-delta)
        return center_term + self._D(wh_pred[..., :1], wh_gt[..., :1]) \
            + self._D(wh_pred[..., 1:], wh_gt[..., 1:])


class v8APTDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss nhung assigner = APTAssigner (Eq 2-3). Loss box/cls/dfl giu nguyen
    co che Ultralytics de train on dinh; APT tac dong qua label assignment."""

    def __init__(self, model, tal_topk=10, power=2.0, thr=0.5):
        super().__init__(model, tal_topk=tal_topk)
        self.assigner = APTAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            power=power,
            thr=thr,
        )


class v10APTDetectionLoss:
    """Loss YOLOv10 dual-head dung APT cho ca one2many (topk=10) va one2one (topk=1).
    Tuong thich chu ky v10DetectionLoss cua Ultralytics: preds = {one2many, one2one}."""

    def __init__(self, model, power=2.0, thr=0.5):
        self.one2many = v8APTDetectionLoss(model, tal_topk=10, power=power, thr=thr)
        self.one2one = v8APTDetectionLoss(model, tal_topk=1, power=power, thr=thr)

    def __call__(self, preds, batch):
        # train: preds = {one2many, one2one}
        # val/eval: v10Detect tra ve tuple (inference_tensor, {one2many, one2one}) -> lay phan tu [1]
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        one2many = preds["one2many"]
        one2one = preds["one2one"]
        lm, lm_items = self.one2many(one2many, batch)
        lo, lo_items = self.one2one(one2one, batch)
        # cong element-wise (box/cls/dfl) giong E2EDetectLoss - KHONG cat (tranh 6 cot loss)
        return lm + lo, lm_items + lo_items


if __name__ == "__main__":
    # smoke test cac thanh phan loss (khong can model).
    logits = torch.randn(4, 6)
    lsoft = torch.rand(4, 6)
    assert SoftClsLoss()(logits, lsoft).shape == (4, 6)
    scl = SoftCenterConfLoss(chi=10, delta=3)
    xy = torch.rand(4, 2); wh = torch.rand(4, 2)
    out = scl(xy, wh, xy.clone(), wh.clone())
    assert torch.isfinite(out).all() and out.shape == (4,)
    print("OK SoftClsLoss + SoftCenterConfLoss")