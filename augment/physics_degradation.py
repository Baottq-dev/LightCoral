# augment/physics_degradation.py
# Module (3): Physics-based underwater degradation augmentation (train-time).
# Ap dung o muc batch tensor (B,3,H,W) trong trainer.preprocess_batch.
# Tra ve (imgs_aug, z_gt) voi z_gt da chuan hoa [0,1]^7 cho L_deg cua PG-DAM.

import torch
import torch.nn as nn


class BatchPhysicsDegradation(nn.Module):
    """I_c = J_c * t_c + B_c * (1 - t_c),  t_c = exp(-beta_c * d(x, y)).

    - beta_c ~ U(beta_range[c]): R hap thu manh nhat (mat mau do truoc).
    - B_c ~ U(background_light), kenh B/G duoc nang nhe (anh nen xanh).
    - d(x, y): depth map gia lap (gradient doc / hang so / huong tam).
    - z_gt = [beta_R, beta_G, beta_B, B_R, B_G, B_B, t_mean] chuan hoa [0,1]
      theo CUNG range trong cfg/module_specs.yaml (khop Sigmoid cua LDE).
    - Anh khong augment: z_gt = [0,0,0,0,0,0,1] (khong suy giam, t=1).
    """

    Z_DIM = 7

    def __init__(
        self,
        prob=0.5,
        beta_range=None,
        background_light=(0.4, 0.9),
        depth_range=(0.5, 3.0),
        depth_mode="vertical_gradient",
        generator=None,   # torch.Generator de tai lap (seed tu utils/seed.py)
    ):
        super().__init__()
        self.prob = prob
        self.beta_range = beta_range or {"R": (0.8, 2.0), "G": (0.3, 1.0), "B": (0.2, 0.8)}
        self.bl_range = tuple(background_light)
        # Range HIEU DUNG cua B_c sau bien doi theo kenh (R *0.6; B clamp >=0.5).
        # Dung de chuan hoa z_gt cho dung - neu chuan theo bl_range goc thi z_R bi
        # am roi clamp ve 0 (mat thong tin giam sat cho LDE).
        _lo, _hi = self.bl_range
        self.bl_lo = (_lo * 0.6, _lo, max(_lo, 0.5))
        self.bl_hi = (_hi * 0.6, _hi, _hi)
        self.depth_range = tuple(depth_range)
        self.depth_mode = depth_mode
        self.gen = generator

    # ---------- helpers ----------
    def _u(self, lo, hi, *shape, device):
        r = torch.rand(*shape, device=device, generator=self.gen)
        return lo + (hi - lo) * r

    def _depth(self, b, h, w, device):
        d_max = self._u(*self.depth_range, b, 1, 1, 1, device=device)
        if self.depth_mode == "constant":
            return d_max.expand(b, 1, h, w)
        if self.depth_mode == "radial":
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, h, device=device),
                torch.linspace(-1, 1, w, device=device),
                indexing="ij",
            )
            r = (xx**2 + yy**2).sqrt().clamp(max=1.0)
            return d_max * r.view(1, 1, h, w)
        # mac dinh: vertical_gradient - cang xa day anh (xa camera) cang sau
        g = torch.linspace(0.3, 1.0, h, device=device).view(1, 1, h, 1)
        return d_max * g.expand(b, 1, h, w)

    def _normalize_z(self, beta, bl, t_mean):
        # min-max normalize ve [0,1] theo range cau hinh (khop voi LDE Sigmoid)
        lo_b = beta.new_tensor([self.beta_range["R"][0], self.beta_range["G"][0], self.beta_range["B"][0]])
        hi_b = beta.new_tensor([self.beta_range["R"][1], self.beta_range["G"][1], self.beta_range["B"][1]])
        zb = (beta - lo_b) / (hi_b - lo_b)
        lo_l = bl.new_tensor(self.bl_lo)
        hi_l = bl.new_tensor(self.bl_hi)
        zl = (bl - lo_l) / (hi_l - lo_l)                          # chuan theo range hieu dung per-kenh
        return torch.cat([zb, zl, t_mean], dim=1).clamp(0, 1)   # (B,7)

    # ---------- main ----------
    @torch.no_grad()
    def forward(self, imgs):
        """imgs: (B,3,H,W) float trong [0,1] (RGB). Tra ve (imgs_aug, z_gt)."""
        b, c, h, w = imgs.shape
        assert c == 3
        device = imgs.device

        # mask anh duoc augment
        apply = torch.rand(b, device=device, generator=self.gen) < self.prob  # (B,)

        # sample tham so vat ly per-image
        beta = torch.stack(
            [
                self._u(*self.beta_range["R"], b, device=device),
                self._u(*self.beta_range["G"], b, device=device),
                self._u(*self.beta_range["B"], b, device=device),
            ],
            dim=1,
        )                                                        # (B,3)
        bl = self._u(*self.bl_range, b, 3, device=device)        # (B,3)
        bl[:, 2] = bl[:, 2].clamp(min=0.5)                       # kenh B sang hon (nen xanh)
        bl[:, 0] = bl[:, 0] * 0.6                                # kenh R toi hon

        d = self._depth(b, h, w, device)                          # (B,1,H,W)
        t = torch.exp(-beta.view(b, 3, 1, 1) * d)                 # (B,3,H,W) transmission

        degraded = imgs * t + bl.view(b, 3, 1, 1) * (1.0 - t)

        m = apply.view(b, 1, 1, 1).float()
        out = m * degraded + (1.0 - m) * imgs

        # ---- z_gt ----
        t_mean = t.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)  # (B,1)
        z_aug = self._normalize_z(beta, bl, t_mean)                  # (B,7)
        z_clean = torch.zeros(b, self.Z_DIM, device=device)
        z_clean[:, 6] = 1.0                                          # t=1: khong suy giam
        z_gt = torch.where(apply.view(b, 1), z_aug, z_clean)
        return out, z_gt


def from_specs(specs: dict, generator=None) -> BatchPhysicsDegradation:
    """Tao tu cfg/module_specs.yaml -> modules.physics_aug."""
    p = specs["modules"]["physics_aug"]
    return BatchPhysicsDegradation(
        prob=p["prob"],
        beta_range={k: tuple(v) for k, v in p["beta_range"].items()},
        background_light=tuple(p["background_light"]),
        depth_range=tuple(p["depth_range"]),
        depth_mode=p.get("depth_mode", "vertical_gradient"),
        generator=generator,
    )