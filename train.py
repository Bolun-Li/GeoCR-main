"""Train GeoCR with the active YOLO11 experiment settings."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def parse_ffc_levels(value: str) -> tuple[int, ...]:
    """Parse a comma-separated subset of P3/P4/P5 feature levels."""
    try:
        levels = tuple(sorted({int(item.strip().upper().removeprefix("P")) for item in value.split(",")}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("FFC levels must look like '3,4,5' or '4,5'") from error
    if not levels or any(level not in {3, 4, 5} for level in levels):
        raise argparse.ArgumentTypeError("FFC levels must be a non-empty subset of 3, 4, 5")
    return levels


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GeoCR on an underwater object-detection dataset")
    parser.add_argument("--data", default="yaml/RUOD.yaml", help="Ultralytics dataset YAML")
    parser.add_argument("--model", default="ultralytics/cfg/geocr/geocr.yaml", help="GeoCR model YAML")
    parser.add_argument("--weights", default="yolo11s.pt", help="Pretrained checkpoint; use '' to train from scratch")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/geocr")
    parser.add_argument("--name", default="geocr-yolo11s")
    parser.add_argument(
        "--ffc-levels", type=parse_ffc_levels, help="Override the FFC levels configured in the model YAML"
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    # Import lazily so argument parsing and repository checks do not require a
    # complete CUDA/PyTorch runtime.
    from ultralytics import YOLO

    model = YOLO(args.model)
    geocr_config = model.model.yaml.get("geocr", {})
    if not bool(geocr_config.get("enabled", False)):
        raise ValueError(f"GeoCR is not enabled in model configuration: {args.model}")
    if args.ffc_levels is not None:
        # Ultralytics rebuilds the training model from this dictionary, so
        # persist command-line overrides before handing it to the trainer.
        geocr_config["levels"] = list(args.ffc_levels)
    if args.weights:
        model.load(args.weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        scale=0.5,
        mosaic=1.0,
        mixup=0.2,
        close_mosaic=15,
        copy_paste=0.5,
        erasing=0.4,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
    )
    model.val(data=args.data)


if __name__ == "__main__":
    main()
