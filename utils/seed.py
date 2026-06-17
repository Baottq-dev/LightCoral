# utils/seed.py
# Seed tat dinh cho toan bo pipeline.

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 0, deterministic: bool = True):
    """Cố định seed cho python/numpy/torch. deterministic=True bat cudnn
    deterministic (cham hon mot chut nhung tai lap duoc)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # cho cac op khong co thuat toan deterministic: canh bao thay vi loi
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def make_generator(seed: int = 0) -> torch.Generator:
    """Generator rieng cho physics aug (tach khoi global RNG => khong nhieu
    loan thu tu sample cua dataloader)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g