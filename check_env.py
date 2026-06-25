#!/usr/bin/env python3
"""DeFT environment check — verifies Python, PyTorch, and required packages."""

import os
import sys
import warnings
from pathlib import Path


MIN_PYTHON = (3, 9)
MIN_TORCH = (2, 0)

REQUIRED = ["torch", "numpy", "scipy", "skimage", "monai", "tqdm"]
OPTIONAL = ["pandas", "lpips", "piq"]

PASS = 0
FAIL = 1


def _green(s):
    return f"\033[92m{s}\033[0m"


def _red(s):
    return f"\033[91m{s}\033[0m"


def _yellow(s):
    return f"\033[93m{s}\033[0m"


def check_python():
    current = (sys.version_info.major, sys.version_info.minor)
    if current >= MIN_PYTHON:
        print(f"{_green('[PASS]')} Python {current[0]}.{current[1]} >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")
        return True
    print(f"{_red('[FAIL]')} Python {current[0]}.{current[1]} < {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")
    return False


def check_torch():
    try:
        import torch
        version = tuple(int(x) for x in torch.__version__.split(".")[:2])
        ver_str = torch.__version__
    except ImportError:
        print(f"{_red('[FAIL]')} PyTorch not installed")
        return False

    if version >= MIN_TORCH:
        print(f"{_green('[PASS]')} PyTorch {ver_str} >= {MIN_TORCH[0]}.{MIN_TORCH[1]}")
    else:
        print(f"{_red('[FAIL]')} PyTorch {ver_str} < {MIN_TORCH[0]}.{MIN_TORCH[1]}")
        return False

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"  CUDA: available ({gpu_count} device(s))")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / (1024**3)
            print(f"  GPU[{i}]: {props.name} ({mem_gb:.1f} GB, CUDA {props.major}.{props.minor})")
    elif torch.backends.mps.is_available():
        print("  MPS: available (Apple Silicon)")
    else:
        print(f"  {_yellow('[WARN]')} No GPU detected (CUDA/MPS unavailable), running on CPU.")

    return True


def check_packages(names, required=True):
    label = "required" if required else "optional"
    all_ok = True
    for name in names:
        try:
            __import__(name)
            print(f"{_green('[PASS]')} {name}")
        except ImportError:
            msg = f"{_red('[FAIL]')} {name}"
            if not required:
                msg += " (optional)"
                all_ok = True  # optional failures don't cause overall failure
            else:
                all_ok = False
            print(msg)
    return all_ok


def check_checkpoints():
    deft_root = os.environ.get("DEFT_ROOT", str(Path(__file__).resolve().parent))
    ckpt_dir = Path(deft_root) / "checkpoints"
    if ckpt_dir.is_dir() and any(ckpt_dir.iterdir()):
        print(f"{_green('[PASS]')} checkpoints/ exists with files")
        return True
    if ckpt_dir.is_dir():
        print(f"{_yellow('[WARN]')} checkpoints/ exists but is empty — download checkpoint files first")
    else:
        print(f"{_yellow('[WARN]')} checkpoints/ not found at {ckpt_dir}")
    return True


def main():
    print("=" * 56)
    print(" DeFT Environment Check")
    print("=" * 56)
    print()

    all_ok = True

    print("Python:")
    all_ok &= check_python()
    print()

    print("PyTorch:")
    all_ok &= check_torch()
    print()

    print(f"Required packages ({', '.join(REQUIRED)}):")
    all_ok &= check_packages(REQUIRED, required=True)
    print()

    print(f"Optional packages ({', '.join(OPTIONAL)}):")
    check_packages(OPTIONAL, required=False)
    print()

    print("Checkpoints:")
    check_checkpoints()
    print()

    print("=" * 56)
    if all_ok:
        print(_green(" All checks passed."))
    else:
        print(_red(" Some checks failed. Fix the issues above."))
    print("=" * 56)

    return PASS if all_ok else FAIL


if __name__ == "__main__":
    sys.exit(main())
