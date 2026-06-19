# benchmark/scoraldet/apt_assigner.py
# Module (3a): APT (Adaptive Power Transformation) label assignment - SCoralDet Eq 2-3.
#   u_trans = u^p   neu u <  T   (nen IoU nho - giam dong gop box kem)
#            = u^(1/p) neu u >= T (khuech dai IoU lon - uu tien box tot)
#   t = s^alpha * u_trans^beta ; chon top-k anchor co t cao nhat lam positive.
# Ke thua TaskAlignedAssigner, CHI override get_box_metrics (it phu thuoc phien ban nhat).

import torch

from ultralytics.utils.tal import TaskAlignedAssigner


class APTAssigner(TaskAlignedAssigner):
    """TaskAlignedAssigner + bien doi power thich nghi tren IoU (Eq 2).

    Tham so them:
        power (p): so mu bien doi (paper mac dinh p=2).
        thr   (T): nguong chuyen che do nen/khuech dai (gia dinh 0.5 - paper khong publish).
    alpha/beta/topk/num_classes ke thua TaskAlignedAssigner (set boi loss).
    """

    def __init__(self, *args, power=2.0, thr=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.power = float(power)
        self.thr = float(thr)

    def _apt_transform(self, overlaps):
        """u_trans = u^p neu u<T, nguoc lai u^(1/p). overlaps da >=0."""
        u = overlaps.clamp_(0)
        p = self.power
        low = u.pow(p)              # nen IoU nho
        high = u.pow(1.0 / p)       # khuech dai IoU lon
        return torch.where(u < self.thr, low, high)

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Tai hien get_box_metrics cua TaskAlignedAssigner, chen u_trans truoc pow(beta)."""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros(
            [self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device
        )
        bbox_scores = torch.zeros(
            [self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device
        )

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        u_trans = self._apt_transform(overlaps)                       # <-- APT Eq 2
        align_metric = bbox_scores.pow(self.alpha) * u_trans.pow(self.beta)  # Eq 3
        return align_metric, overlaps


if __name__ == "__main__":
    # smoke test: bien doi power dung huong (nen u nho, giu/khuech dai u lon).
    a = APTAssigner(topk=10, num_classes=6, alpha=0.5, beta=6.0, power=2.0, thr=0.5)
    u = torch.tensor([0.1, 0.4, 0.5, 0.9])
    t = a._apt_transform(u)
    assert t[0] < u[0] and t[1] < u[1], "u<T phai bi nen"
    assert t[3] > u[3], "u>=T phai duoc khuech dai"
    print("OK APTAssigner transform:", [round(v, 3) for v in t.tolist()])