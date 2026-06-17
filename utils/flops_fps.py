# utils/flops_fps.py
# Do Params / GFLOPs / FPS cho 1 to hop module.
#
#   python -m utils.flops_fps --modules 1,2,4,5 --imgsz 640 --device 0

import argparse
import time

import torch
import yaml

from engine.build_model import ROOT, build_model


@torch.no_grad()
def measure(model, imgsz=640, device="cpu", warmup=10, iters=50):
    net = model.model.to(device).eval()
    x = torch.randn(1, 3, imgsz, imgsz, device=device)

    n_params = sum(p.numel() for p in net.parameters())

    gflops = None
    try:
        from thop import profile
        macs, _ = profile(net, inputs=(x,), verbose=False)
        gflops = 2 * macs / 1e9   # 1 MAC = 2 FLOPs
    except Exception:
        pass  # thop khong bat buoc

    for _ in range(warmup):
        net(x)
    if device != "cpu":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        net(x)
    if device != "cpu":
        torch.cuda.synchronize()
    fps = iters / (time.perf_counter() - t0)

    return {
        "params_M": round(n_params / 1e6, 3),
        "gflops": round(gflops, 2) if gflops else None,
        "fps": round(fps, 1),
        "imgsz": imgsz,
        "device": str(device),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Do Params/GFLOPs/FPS")
    ap.add_argument("--modules", default="", help="vd: 1,2,4,5 (rong = baseline)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    specs = yaml.safe_load((ROOT / "cfg" / "module_specs.yaml").read_text())
    mods = [int(x) for x in args.modules.split(",") if x.strip()]
    dev = args.device if args.device == "cpu" else f"cuda:{args.device}"
    model = build_model(mods, specs, verbose=False)
    import json
    print(json.dumps(measure(model, args.imgsz, dev), indent=2))