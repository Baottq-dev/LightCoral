# models/pg_dam.py
# Module (4): PG-DAM = LDE (uoc luong tham so suy giam) + FiLM modulation.
# Chen sau stem Conv (idx 0). Chu ky: __init__(c1, c2, z_dim, hidden); c1 == c2.

import copy

import torch
import torch.nn as nn


class PGDAM_FiLM(nn.Module):
    """Physics-Guided Degradation-Aware Modulation.

    LDE:  z_hat = sigmoid(FC2(SiLU(FC1(GAP(F)))))  trong [0,1]^z_dim
          z = (beta_R, beta_G, beta_B, B_R, B_G, B_B, t_mean) da chuan hoa
          theo range trong cfg/module_specs.yaml (giong z_gt cua module 3).
    FiLM: F' = gamma(z) * F + beta(z), voi gamma ~ 1 va beta ~ 0 luc khoi tao
          de khong pha vo dac trung pretrained (warm start an toan).

    z_hat duoc luu vao self.last_z de engine/losses.py tinh L_deg.
    """

    def __init__(self, c1, c2, z_dim=7, hidden=32):
        super().__init__()
        assert c1 == c2, "PG-DAM la modulation tai cho: c1 phai bang c2"
        self.z_dim = z_dim

        # ---- LDE: Latent Degradation Estimator ----
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.lde = nn.Sequential(
            nn.Linear(c1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, z_dim),
            nn.Sigmoid(),  # z_hat trong [0,1]^z_dim, cung thang do voi z_gt
        )

        # ---- FiLM generator: z -> (gamma, beta) ----
        self.film = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * c1),
        )
        # khoi tao lop cuoi ~ 0 => gamma ~ 1, beta ~ 0 (gan identity luc dau)
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

        self.last_z = None  # (B, z_dim) - duoc losses.py doc sau moi forward

    def forward(self, x):
        b, c, _, _ = x.shape
        z = self.lde(self.gap(x).flatten(1))          # (B, z_dim)
        self.last_z = z

        gb = self.film(z)                              # (B, 2C)
        gamma, beta = gb.chunk(2, dim=1)
        # gioi han bien do dieu bien de on dinh huan luyen
        gamma = 1.0 + torch.tanh(gamma)                # gamma trong (0, 2), ~1 luc dau
        beta = 0.5 * torch.tanh(beta)                  # beta trong (-0.5, 0.5), ~0 luc dau
        return x * gamma.view(b, c, 1, 1) + beta.view(b, c, 1, 1)

    # ---- giu last_z khoi deepcopy/pickle ----
    # last_z la tensor non-leaf (dinh graph autograd). ModelEMA goi deepcopy(model)
    # va torch.save pickle model => deepcopy/pickle tensor non-leaf se bao loi
    # "Only Tensors created explicitly by the user (graph leaves) support the
    # deepcopy protocol". Ta loai last_z khoi 2 co che nay; gradient cho L_deg
    # van con nguyen trong buoc forward thuc (engine/losses.py doc last_z ngay
    # sau forward, truoc khi deepcopy/save xay ra).
    def __deepcopy__(self, memo):
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            new.__dict__[k] = None if k == "last_z" else copy.deepcopy(v, memo)
        return new

    def __getstate__(self):
        state = self.__dict__.copy()
        state["last_z"] = None
        return state