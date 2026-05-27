"""
Script 2: Online Augmentation Library — Ultralytics-native
===========================================================

Library module — không chạy trực tiếp để train.
Được gọi từ scripts/3_train_baseline.py qua flag --custom_aug.

Có thể dùng --preview để xem trước các transform:
  python scripts/2_augmentations.py --preview <IMG_PATH>
  python scripts/2_augmentations.py --preview <IMG_PATH> --n_preview 12 --preview_save preview.jpg
  python scripts/2_augmentations.py --preview <IMG_PATH> --aug_groups underwater noise --aug_intensity strong

Cách hoạt động
──────────────
  CoralTrainer  (extends DetectionTrainer)
    └─ build_dataset() → CoralDataset  (extends YOLODataset)
         └─ build_transforms() →
              [ Ultralytics pipeline: Mosaic → Perspective → HSV → Flip ]
                       ↓
              [ CoralPipeline ← underwater transforms, in RAM, không đụng disk ]
                       ↓
              [ Format → tensor → GPU ]

Augmentation groups
───────────────────
  underwater  : color cast, light attenuation, backscatter, motion blur,
                caustic light, turbidity, chromatic aberration, depth blur
  noise       : Gaussian noise, salt-and-pepper, JPEG compression
  occlusion   : random erasing patches
  gamma       : gamma correction for exposure variation

Public API (import từ 3_train_baseline.py)
──────────────────────────────────────────
  CoralAugConfig   — cấu hình augmentation (groups, intensity, p_overrides)
  CoralPipeline    — wrapper chain transforms
  CoralDataset     — YOLODataset subclass có CoralPipeline
  CoralTrainer     — DetectionTrainer subclass dùng CoralDataset
  preview_augmentations(img_path, config, ...) — hiện grid ảnh preview

Tested with: ultralytics==8.4.52, torch 2.7.1
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Euphflfiaancora",
    "Favosites",
    "Platygyra",
    "Sarcophyton",
    "Sinularia",
    "WavingHand",
]

CLASS_COLORS = [  # BGR, for visualization
    (255, 100, 100),
    (100, 255, 100),
    (100, 100, 255),
    (255, 255, 100),
    (255, 100, 255),
    (100, 255, 255),
]

ALL_GROUPS = ["underwater", "noise", "occlusion", "gamma"]

# ───────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSFORM CLASSES
#  ─────────────────────────────────────────────────────────────────────────────
#  Protocol (same as Ultralytics internal transforms):
#    __call__(labels: dict) → labels: dict
#
#  Key fields used:
#    labels["img"]  → np.ndarray, shape (H, W, 3), dtype uint8, BGR
#                     Available after LetterBox/Mosaic; still numpy at this point.
#    Bounding boxes are NOT touched by color/noise transforms (no bbox adjustment
#    needed — only the pixel values change).
# ══════════════════════════════════════════════════════════════════════════════

class _BaseT:
    """Mixin: probabilistic skip."""
    def __init__(self, p: float):
        self.p = float(p)

    def _skip(self) -> bool:
        return random.random() > self.p

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p:.2f})"


# ── Group: underwater ─────────────────────────────────────────────────────────

class UnderwaterColorCast(_BaseT):
    """
    Simulate underwater color attenuation.
    Red light attenuates fastest with depth → image shifts toward blue/green/teal.
    Mode is sampled randomly per image.
    """
    def __init__(self, p: float = 0.60):
        super().__init__(p)

    def __call__(self, labels: dict) -> dict:
        if self._skip():
            return labels
        img = labels["img"].astype(np.float32)
        mode = random.choice(["blue_dominant", "green_dominant", "teal"])
        d = random.uniform(0.3, 0.9)          # depth factor

        if mode == "blue_dominant":
            img[:, :, 2] *= 1.0 - d * 0.70   # attenuate red strongly
            img[:, :, 1] *= 1.0 - d * 0.30   # attenuate green mildly
            img[:, :, 0]  = np.clip(img[:, :, 0] * 1.05, 0, 255)  # boost blue
        elif mode == "green_dominant":
            img[:, :, 2] *= 1.0 - d * 0.60
            img[:, :, 1]  = np.clip(img[:, :, 1] * 1.05, 0, 255)
        else:  # teal
            img[:, :, 2] *= 1.0 - d * 0.50
            img[:, :, 0]  = np.clip(img[:, :, 0] * 1.03, 0, 255)
            img[:, :, 1]  = np.clip(img[:, :, 1] * 1.03, 0, 255)

        labels["img"] = np.clip(img, 0, 255).astype(np.uint8)
        return labels


class LightAttenuation(_BaseT):
    """
    Depth-based light falloff vignette.
    Three modes: radial (point-source), top-down (gravity), corner (oblique sun).
    """
    def __init__(self, p: float = 0.50):
        super().__init__(p)

    def __call__(self, labels: dict) -> dict:
        if self._skip():
            return labels
        img = labels["img"]
        h, w = img.shape[:2]
        mode = random.choice(["radial", "top_down", "corner"])

        if mode == "radial":
            cx = random.uniform(0.3, 0.7) * w
            cy = random.uniform(0.3, 0.7) * h
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            mask = 1.0 - random.uniform(0.3, 0.7) * dist / np.sqrt(h**2 + w**2)

        elif mode == "top_down":
            col  = np.linspace(1.0, random.uniform(0.4, 0.8), h)
            mask = np.broadcast_to(col[:, None], (h, w)).copy()

        else:  # corner
            Y, X = np.mgrid[:h, :w].astype(np.float32)
            cx = w * random.uniform(0.05, 0.45)
            cy = h * random.uniform(0.05, 0.45)
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            mask = 1.0 - random.uniform(0.2, 0.5) * dist / np.sqrt(h**2 + w**2)

        mask = np.clip(mask, 0.2, 1.0).astype(np.float32)[:, :, None]
        labels["img"] = np.clip(img.astype(np.float32) * mask, 0, 255).astype(np.uint8)
        return labels


class Backscatter(_BaseT):
    """
    Suspended particles in water scatter flashlight/ambient light
    back toward the camera: appears as random bright speckles + uniform haze.
    """
    def __init__(self, p: float = 0.40):
        super().__init__(p)

    def __call__(self, labels: dict) -> dict:
        if self._skip():
            return labels
        img = labels["img"]
        h, w = img.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.float32)

        for _ in range(random.randint(50, 400)):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            r = random.randint(1, 4)
            s = random.uniform(0.5, 1.0)
            cv2.circle(overlay, (x, y), r, (s * 220, s * 220, s * 255), -1)

        overlay += random.uniform(0.0, 0.08) * 255   # uniform haze
        alpha = random.uniform(0.15, 0.40)
        labels["img"] = np.clip(
            img.astype(np.float32) * (1 - alpha) + overlay * alpha, 0, 255
        ).astype(np.uint8)
        return labels


class MotionBlur(_BaseT):
    """
    Directional blur from camera shake or water current.
    Kernel direction is sampled uniformly over 360°.
    """
    def __init__(self, p: float = 0.30, max_k: int = 11):
        super().__init__(p)
        self.max_k = max_k

    def __call__(self, labels: dict) -> dict:
        if self._skip():
            return labels
        k = random.choice(range(3, self.max_k + 1, 2))
        angle = random.uniform(0, 360)
        ker = np.zeros((k, k), dtype=np.float32)
        ker[k // 2, :] = 1.0 / k
        M   = cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1)
        ker = cv2.warpAffine(ker, M, (k, k))
        ker /= ker.sum() + 1e-8
        labels["img"] = cv2.filter2D(labels["img"], -1, ker)
        return labels


class CausticLight(_BaseT):
    """
    Rippling light from the water surface.
    Approximated as a superposition of sinusoidal waves (blue-tinted overlay).
    """
    def __init__(self, p: float = 0.40):
        super().__init__(p)

    def __call__(self, labels: dict) -> dict:
        if self._skip():
            return labels
        img = labels["img"]
        h, w = img.shape[:2]
        Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
        c = np.zeros((h, w), dtype=np.float32)

        for _ in range(random.randint(3, 6)):
            fx    = random.uniform(0.005, 0.030)
            fy    = random.uniform(0.005, 0.030)
            phase = random.uniform(0.0, 6.2832)
            c    += random.uniform(0.2, 0.6) * np.sin(fx * X + fy * Y + phase)

        c = (c - c.min()) / (c.max() - c.min() + 1e-8)
        strength = random.uniform(0.05, 0.25)
        rgb = np.stack([c * 0.80, c * 0.95, c], axis=-1)   # blue-tinted
        labels["img"] = np.clip(
            img.astype(np.float32) + strength * rgb * 255, 0, 255
        ).astype(np.uint8)
        return labels


class Turbidity(_BaseT):
    """
    Murky / cloudy water: blend the image with a soft colored fog layer.
    Fog color sampled from realistic underwater palettes (greenish / blueish / grey-green).
    """
    def __init__(self, p: float = 0.30):
        super().__init__(p)

    def __call__(self, labels: dict) -> dict:
        if self._skip():
            return labels
        img = labels["img"]
        h, w = img.shape[:2]
        fog_color = random.choice([(180, 200, 160), (160, 180, 200), (140, 160, 150)])
        fog   = np.full((h, w, 3), fog_color, dtype=np.float32)
        alpha = random.uniform(0.10, 0.40)
        labels["img"] = np.clip(
            img.astype(np.float32) * (1 - alpha) + fog * alpha, 0, 255
        ).astype(np.uint8)
        return labels


class ChromaticAberration(_BaseT):
    """
    Color channel fringing from water acting as a refractive lens.
    Red channel shifted one way, blue channel shifted the opposite way.
    """
    def __init__(self, p: float = 0.25, max_shift: int = 4):
        super().__init__(p)
        self.max_shift = max_shift

    def __call__(self, labels: dict) -> dict:
        if self._skip():
            return labels
        img = labels["img"]
        h, w = img.shape[:2]
        s = random.randint(1, self.max_shift)
        b, g, r = cv2.split(img)
        r = cv2.warpAffine(r, np.float32([[1, 0,  s], [0, 1,  s]]), (w, h))
        b = cv2.warpAffine(b, np.float32([[1, 0, -s], [0, 1, -s]]), (w, h))
        labels["img"] = cv2.merge([b, g, r])
        return labels


class UnderwaterBlur(_BaseT):
    """
    Non-uniform depth-zone blur: the upper portion of the frame (near the surface)
    is blurred more than the lower portion, simulating surface wave distortion.
    """
    def __init__(self, p: float = 0.30):
        super().__init__(p)

    def __call__(self, labels: dict) -> dict:
        if self._skip():
            return labels
        img   = labels["img"]
        h     = img.shape[0]
        split = random.randint(h // 4, 3 * h // 4)
        k     = random.choice([3, 5, 7])
        top   = cv2.GaussianBlur(img[:split], (k, k), 0)
        labels["img"] = np.vstack([top, img[split:]])
        return labels


# ── Group: noise ──────────────────────────────────────────────────────────────

class GaussianNoise(_BaseT):
    """Additive Gaussian noise (sensor noise in low-light underwater cameras)."""
    def __init__(self, p: float = 0.40, max_std: float = 20.0):
        super().__init__(p)
        self.max_std = max_std

    def __call__(self, labels: dict) -> dict:
        if self._skip():
            return labels
        std   = random.uniform(5.0, self.max_std)
        noise = np.random.normal(0, std, labels["img"].shape).astype(np.float32)
        labels["img"] = np.clip(
            labels["img"].astype(np.float32) + noise, 0, 255
        ).astype(np.uint8)
        return labels


class SaltPepperNoise(_BaseT):
    """Salt-and-pepper pixel corruption (sensor noise / data transmission errors)."""
    def __init__(self, p: float = 0.20, max_amount: float = 0.005):
        super().__init__(p)
        self.max_amount = max_amount

    def __call__(self, labels: dict) -> dict:
        if self._skip():
            return labels
        img    = labels["img"].copy()
        amount = random.uniform(0.0, self.max_amount)
        n      = int(img.size * amount)
        # Salt
        ys = np.random.randint(0, img.shape[0], n)
        xs = np.random.randint(0, img.shape[1], n)
        img[ys, xs] = 255
        # Pepper
        ys = np.random.randint(0, img.shape[0], n)
        xs = np.random.randint(0, img.shape[1], n)
        img[ys, xs] = 0
        labels["img"] = img
        return labels


class JPEGCompression(_BaseT):
    """JPEG compression artifacts (lossy underwater video transmission)."""
    def __init__(self, p: float = 0.30, quality_range: tuple[int, int] = (30, 80)):
        super().__init__(p)
        self.quality_range = quality_range

    def __call__(self, labels: dict) -> dict:
        if self._skip():
            return labels
        q    = random.randint(*self.quality_range)
        _, buf = cv2.imencode(".jpg", labels["img"], [cv2.IMWRITE_JPEG_QUALITY, q])
        labels["img"] = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return labels


# ── Group: occlusion ──────────────────────────────────────────────────────────

class RandomErasing(_BaseT):
    """
    Random rectangular occlusion patches.
    Fill mode is sampled per patch: black / random noise / mean color.
    Simulates fish, hands, equipment, or debris partially covering a coral.
    """
    def __init__(self, p: float = 0.30, max_count: int = 3):
        super().__init__(p)
        self.max_count = max_count

    def __call__(self, labels: dict) -> dict:
        if self._skip():
            return labels
        img  = labels["img"].copy()
        h, w = img.shape[:2]
        mean = img.mean(axis=(0, 1)).astype(np.uint8)

        for _ in range(random.randint(1, self.max_count)):
            rw  = random.randint(w // 20, w // 5)
            rh  = random.randint(h // 20, h // 5)
            rx  = random.randint(0, max(0, w - rw - 1))
            ry  = random.randint(0, max(0, h - rh - 1))
            fill = random.choice(["black", "noise", "mean"])
            if fill == "black":
                img[ry:ry + rh, rx:rx + rw] = 0
            elif fill == "noise":
                img[ry:ry + rh, rx:rx + rw] = np.random.randint(
                    0, 256, (rh, rw, 3), dtype=np.uint8
                )
            else:
                img[ry:ry + rh, rx:rx + rw] = mean

        labels["img"] = img
        return labels


# ── Group: gamma ──────────────────────────────────────────────────────────────

class GammaCorrection(_BaseT):
    """
    Gamma correction for exposure variation between dives / cameras.
    gamma < 1 → brighten (overexposed), gamma > 1 → darken (underexposed).
    """
    def __init__(self, p: float = 0.50, gamma_range: tuple[float, float] = (0.5, 2.0)):
        super().__init__(p)
        self.gamma_range = gamma_range

    def __call__(self, labels: dict) -> dict:
        if self._skip():
            return labels
        g   = random.uniform(*self.gamma_range)
        lut = np.array(
            [((i / 255.0) ** (1.0 / g)) * 255 for i in range(256)], dtype=np.uint8
        )
        labels["img"] = cv2.LUT(labels["img"], lut)
        return labels


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CoralAugConfig:
    """
    Configuration for the coral augmentation pipeline.

    Args:
        groups     : subset of ALL_GROUPS to activate
        intensity  : scales all base probabilities ('light'=0.5×, 'medium'=1×, 'strong'=1.5×)
        p_overrides: per-transform probability overrides, e.g. {"backscatter": 0.8}

    Example:
        cfg = CoralAugConfig(intensity="strong", groups=["underwater", "noise"])
        cfg = CoralAugConfig(p_overrides={"caustics": 0.9, "erasing": 0.0})
    """
    groups:      list[str]       = field(default_factory=lambda: list(ALL_GROUPS))
    intensity:   str             = "medium"          # "light" | "medium" | "strong"
    p_overrides: dict[str, float] = field(default_factory=dict)

    _SCALE = {"light": 0.50, "medium": 1.00, "strong": 1.50}

    def _p(self, name: str, base: float) -> float:
        """Effective probability for a named transform."""
        scale = self._SCALE.get(self.intensity, 1.0)
        return min(1.0, self.p_overrides.get(name, base * scale))

    def build(self) -> list:
        """Instantiate and return the list of active transforms."""
        p  = self._p
        ts: list = []

        if "underwater" in self.groups:
            ts += [
                UnderwaterColorCast(p=p("color_cast",   0.60)),
                LightAttenuation(   p=p("attenuation",  0.50)),
                Backscatter(        p=p("backscatter",   0.40)),
                MotionBlur(         p=p("motion_blur",   0.30)),
                CausticLight(       p=p("caustics",      0.40)),
                Turbidity(          p=p("turbidity",     0.30)),
                ChromaticAberration(p=p("chromatic_ab",  0.25)),
                UnderwaterBlur(     p=p("uw_blur",       0.30)),
            ]
        if "noise" in self.groups:
            ts += [
                GaussianNoise(   p=p("gaussian_noise", 0.40)),
                SaltPepperNoise( p=p("salt_pepper",    0.20)),
                JPEGCompression( p=p("jpeg",           0.30)),
            ]
        if "occlusion" in self.groups:
            ts += [
                RandomErasing(   p=p("erasing",        0.30)),
            ]
        if "gamma" in self.groups:
            ts += [
                GammaCorrection( p=p("gamma",          0.50)),
            ]
        return ts

    def summary(self) -> str:
        """Human-readable summary of active transforms."""
        ts  = self.build()
        lines = [
            f"CoralAugConfig  intensity={self.intensity}  groups={self.groups}",
            f"  {len(ts)} active transforms:",
        ]
        for t in ts:
            lines.append(f"    {t!r}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE WRAPPER
#  Single callable that chains all coral transforms — compatible with
#  Ultralytics' internal Compose class.
# ══════════════════════════════════════════════════════════════════════════════

class CoralPipeline:
    """
    Wraps a list of coral transforms into one callable that fits anywhere
    inside Ultralytics' Compose chain.

    Input / output: labels dict (same contract as Ultralytics transforms).
    Only labels["img"] (numpy HWC uint8 BGR) is modified.
    """
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, labels: dict) -> dict:
        for t in self.transforms:
            labels = t(labels)
        return labels

    def __repr__(self) -> str:
        names = ", ".join(type(t).__name__ for t in self.transforms)
        return f"CoralPipeline([{names}])"


# ══════════════════════════════════════════════════════════════════════════════
#  ULTRALYTICS INTEGRATION
#  Lazy import so the module can be used for --preview without ultralytics.
# ══════════════════════════════════════════════════════════════════════════════

try:
    from ultralytics.data import YOLODataset as _YOLODataset
    from ultralytics.models.yolo.detect import DetectionTrainer as _DetectionTrainer
    from ultralytics.utils import colorstr as _colorstr
    from ultralytics.nn.tasks import attempt_load_weights
    _UL_OK = True
except ImportError:
    _UL_OK = False
    _YOLODataset       = object
    _DetectionTrainer  = object


class CoralDataset(_YOLODataset):
    """
    YOLODataset with CoralPipeline injected into the transform chain.

    The pipeline after this class looks like (training mode):
      [Mosaic / LetterBox]
        → RandomPerspective → Albumentations → RandomHSV → RandomFlip
        → CoralPipeline          ← coral transforms (in-RAM)
        → Format                 ← last: converts to tensor

    For val/test (augment=False) the coral transforms are NOT applied.
    """

    def __init__(self, *args, coral_config: CoralAugConfig | None = None, **kwargs):
        # Must be set BEFORE super().__init__() because build_transforms() is
        # called inside the parent constructor.
        self._coral_config = coral_config or CoralAugConfig()
        super().__init__(*args, **kwargs)

    def build_transforms(self, hyp=None):
        """Override: append CoralPipeline before the terminal Format transform."""
        transforms = super().build_transforms(hyp)

        if self.augment:
            coral_ts = self._coral_config.build()
            if coral_ts:
                pipeline = CoralPipeline(coral_ts)
                # transforms.transforms is a list; [-1] is always Format
                transforms.transforms.insert(-1, pipeline)

        return transforms


class CoralTrainer(_DetectionTrainer):
    """
    DetectionTrainer that swaps the default YOLODataset for CoralDataset.

    Usage:
        from scripts.augmentations import CoralTrainer, CoralAugConfig

        trainer = CoralTrainer(
            overrides={
                "model": "yolov8s.pt",
                "data":  "configs/coral_soft.yaml",
                "epochs": 100,
            },
            coral_config=CoralAugConfig(intensity="strong"),
        )
        trainer.train()
    """

    def __init__(
        self,
        cfg=None,
        overrides: dict | None = None,
        _callbacks=None,
        coral_config: CoralAugConfig | None = None,
    ):
        if not _UL_OK:
            raise ImportError(
                "ultralytics is not installed.\n"
                "Run: pip install ultralytics"
            )
        self._coral_config = coral_config or CoralAugConfig()
        from ultralytics.cfg import DEFAULT_CFG
        super().__init__(
            cfg=cfg or DEFAULT_CFG,
            overrides=overrides,
            _callbacks=_callbacks,
        )

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Return CoralDataset instead of the default YOLODataset."""
        from ultralytics.nn.tasks import unwrap_model
        gs = max(int(unwrap_model(self.model).stride.max()), 32)

        pad = 0.0 if mode == "train" else 0.5
        return CoralDataset(
            img_path   = img_path,
            imgsz      = self.args.imgsz,
            batch_size = batch,
            augment    = mode == "train",
            hyp        = self.args,
            rect       = getattr(self.args, "rect", False) or (mode == "val"),
            cache      = self.args.cache or None,
            single_cls = self.args.single_cls or False,
            stride     = int(gs),
            pad        = pad,
            prefix     = _colorstr(f"{mode}: "),
            task       = self.args.task,
            classes    = self.args.classes,
            data       = self.data,
            fraction   = self.args.fraction if mode == "train" else 1.0,
            coral_config = self._coral_config,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PREVIEW UTILITIES  (no ultralytics needed)
# ══════════════════════════════════════════════════════════════════════════════

def _fake_labels(img: np.ndarray) -> dict:
    """Minimal labels dict for running transforms in preview mode."""
    return {
        "img":    img.copy(),
        "cls":    np.array([], dtype=np.float32),
        "bboxes": np.zeros((0, 4), dtype=np.float32),
    }


def preview_augmentations(
    img_path:     str | Path,
    config:       CoralAugConfig,
    n:            int = 9,
    cell_size:    int = 340,
    save_path:    str | Path | None = None,
) -> None:
    """
    Show a grid of augmented versions of a single image.

    First cell is always the original.
    Requires an active display (cv2.imshow) unless save_path is given.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    pipeline = CoralPipeline(config.build())
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    grid = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)

    # Cell 0: original
    tile = cv2.resize(img, (cell_size, cell_size))
    _put_label(tile, "ORIGINAL", color=(0, 220, 0))
    grid[:cell_size, :cell_size] = tile

    for i in range(1, n):
        labels  = _fake_labels(img)
        labels  = pipeline(labels)
        aug_img = labels["img"]
        r, c    = divmod(i, cols)
        tile    = cv2.resize(aug_img, (cell_size, cell_size))
        _put_label(tile, f"aug {i}")
        y0 = r * cell_size
        x0 = c * cell_size
        grid[y0:y0 + cell_size, x0:x0 + cell_size] = tile

    # Footer info
    info = f"groups={config.groups}  intensity={config.intensity}  ({len(config.build())} transforms)"
    cv2.putText(grid, info, (8, grid.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    if save_path:
        cv2.imwrite(str(save_path), grid)
        print(f"Preview saved → {save_path}")
    else:
        title = f"CoralAug Preview  [{Path(img_path).name}]  (press any key to close)"
        cv2.imshow(title, grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _put_label(img: np.ndarray, text: str, color: tuple = (0, 180, 255)) -> None:
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color,     1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI — chỉ dành cho --preview (preview transforms, không train)
#  Để train, dùng: python scripts/3_train_baseline.py --custom_aug
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Preview coral augmentation transforms on a sample image.\n"
            "Để train với augmentation, dùng:\n"
            "  python scripts/3_train_baseline.py --custom_aug [--aug_groups ...] [--aug_intensity ...]"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Augmentation config ────────────────────────────────────────────────────
    parser.add_argument(
        "--aug_groups", nargs="+", default=list(ALL_GROUPS),
        choices=ALL_GROUPS, metavar="GROUP",
        help=f"Augmentation groups to activate. Choices: {ALL_GROUPS}",
    )
    parser.add_argument(
        "--aug_intensity", type=str, default="medium",
        choices=["light", "medium", "strong"],
        help="Scales all transform probabilities (0.5× / 1× / 1.5×).",
    )

    # ── Preview ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--preview", type=str, required=True, metavar="IMG_PATH",
        help="Đường dẫn ảnh để xem preview augmentation.",
    )
    parser.add_argument(
        "--n_preview", type=int, default=9,
        help="Số ô preview (ô đầu luôn là ảnh gốc).",
    )
    parser.add_argument(
        "--preview_save", type=str, default=None, metavar="OUT.JPG",
        help="Lưu preview ra file thay vì hiện lên màn hình.",
    )

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI entry-point: preview-only. Training handled by 3_train_baseline.py."""
    args = parse_args()
    coral_config = CoralAugConfig(
        groups    = args.aug_groups,
        intensity = args.aug_intensity,
    )
    print(coral_config.summary())
    save = Path(args.preview_save) if args.preview_save else None
    preview_augmentations(
        img_path  = args.preview,
        config    = coral_config,
        n         = args.n_preview,
        save_path = save,
    )


if __name__ == "__main__":
    main()
