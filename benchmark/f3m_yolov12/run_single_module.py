# run_single_module.py
"""
Chay RIENG LE tung module F3M (chi can torch, khong can Ultralytics).

    python run_single_module.py          # chay het
    python run_single_module.py f3m      # chi F3M (Neck)
    python run_single_module.py sa       # chi F3MWithSA (Stem)
    python run_single_module.py sub      # chi cac sub-block

Luu y: F3M/F3MWithSA suy ra so kenh tu dong o forward dau tien (lazy build),
nen khong truyen in_channels vao constructor; chi can dua tensor dung so kenh.
"""

import sys
import torch

from f3m_modules import (
    F3M, F3MWithSA,
    FrequencySeparator, FrequencyProjector, FrequencyFuser, SpatialAttentionModule,
)


def count(m):
    return sum(p.numel() for p in m.parameters())


def demo_f3m():
    print("\n=== F3M (Neck, lite: r=0.125, gate=False) ===")
    x = torch.randn(2, 1024, 20, 20)            # P5 feature map
    module = F3M(reduction_ratio=0.125, use_gate=False)
    module.eval()
    with torch.no_grad():
        y = module(x)                            # lazy build theo x.shape[1]=1024
    print("  in :", tuple(x.shape), "-> out:", tuple(y.shape), "| params:", count(module))
    assert y.shape == x.shape


def demo_f3m_sa():
    print("\n=== F3MWithSA (Stem, full: r=0.33, gate=True, k=7) ===")
    x = torch.randn(2, 64, 320, 320)            # feature map do phan giai cao
    module = F3MWithSA(reduction_ratio=0.33, use_gate=True, sa_kernel_size=7)
    module.eval()
    with torch.no_grad():
        y = module(x)
    print("  in :", tuple(x.shape), "-> out:", tuple(y.shape), "| params:", count(module))
    assert y.shape == x.shape


def demo_subblocks():
    print("\n=== Cac sub-block rieng le (eager, truyen so kenh truc tiep) ===")
    C = 256
    x = torch.randn(2, C, 80, 80)

    sep = FrequencySeparator()
    x_lf, x_hf = sep(x)
    print("  [Separate] x_lf:", tuple(x_lf.shape), " x_hf:", tuple(x_hf.shape))
    assert torch.allclose(x_lf + x_hf, x, atol=1e-5)
    print("            check x_lf + x_hf == x : OK")

    proj = FrequencyProjector(C, reduction_ratio=0.25)
    x_lf_p, x_hf_p = proj(x_lf, x_hf)
    print("  [Project ] C' =", proj.mid_channels, " -> ", tuple(x_lf_p.shape), "| params:", count(proj))

    fuse = FrequencyFuser(C, proj.mid_channels, use_gate=True)
    y = fuse(x, x_lf_p, x_hf_p)
    print("  [Fuse    ] out:", tuple(y.shape), "| params:", count(fuse))

    sa = SpatialAttentionModule(kernel_size=7)
    y_sa = sa(y)
    print("  [SpatialA] out:", tuple(y_sa.shape), "| params:", count(sa))
    assert y_sa.shape == x.shape


if __name__ == "__main__":
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    if arg in ("all", "f3m"):
        demo_f3m()
    if arg in ("all", "sa"):
        demo_f3m_sa()
    if arg in ("all", "sub"):
        demo_subblocks()
    print("\nDONE.")