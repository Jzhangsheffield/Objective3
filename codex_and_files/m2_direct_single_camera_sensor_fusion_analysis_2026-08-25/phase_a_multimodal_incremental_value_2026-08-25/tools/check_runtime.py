from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the Python/PyTorch runtime before Phase A")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    try:
        import numpy
        import torch
        import torchvision
    except Exception as error:
        print(f"RUNTIME_IMPORT_ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        print(f"python_executable={sys.executable}", file=sys.stderr)
        raise

    report = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "working_directory": str(Path.cwd()),
        "numpy_version": numpy.__version__,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "requested_device": args.device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Device '{args.device}' was requested, but torch.cuda.is_available() is False. "
            "Use a CUDA-enabled PyTorch environment/GPU driver, or explicitly run with -Device cpu."
        )


if __name__ == "__main__":
    main()
