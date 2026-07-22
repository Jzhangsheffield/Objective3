from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refuse to start a stage when its dedicated output directory is non-empty"
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty and has no completed marker: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory is safe to use: {output_dir}")


if __name__ == "__main__":
    main()
