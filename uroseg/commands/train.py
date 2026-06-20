from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import uroseg.utils.utils as _utils
from uroseg.utils.utils import normalize_labels


def extract_dataset_id(nnunet_task: str) -> int:
    match = re.match(r'Dataset(\d+)_', nnunet_task)
    if not match:
        raise ValueError(f"Invalid nnunet_task format: {nnunet_task!r} — expected 'DatasetNNN_Name'")
    return int(match.group(1))


def generate_dataset_json(model: dict, images_tr_dir: Path) -> dict:
    labels = normalize_labels(model["labels"])

    # Collect all unique non-zero label integers; detect if any are regions (lists)
    all_values: set[int] = set()
    has_regions = False
    for v in labels.values():
        if isinstance(v, list):
            all_values.update(int(x) for x in v)
            has_regions = True
        elif int(v) != 0:
            all_values.add(int(v))

    dataset: dict = {
        "channel_names": {"0": "MRI"},
        "labels": labels,
        "numTraining": len(list(images_tr_dir.glob("*.nii.gz"))),
        "file_ending": ".nii.gz",
    }
    if has_regions:
        dataset["regions_class_order"] = sorted(all_values)
    return dataset


def setup_nnunet_env(data_path: Path) -> None:
    nnunet_dir = data_path / "nnUNet"
    os.environ["nnUNet_raw"] = str(nnunet_dir / "raw")
    os.environ["nnUNet_preprocessed"] = str(nnunet_dir / "preprocessed")
    os.environ["nnUNet_results"] = str(nnunet_dir / "results")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an nnU-Net model for a UroSeg organ using AugLab augmentation."
    )
    parser.add_argument("organ",
                        help="Organ name matching resources/models/<organ>.json")
    parser.add_argument("--fold", type=int, default=0,
                        help="nnU-Net fold number (0–4, default: 0)")
    parser.add_argument("--auglab-config", default=None,
                        help="Path to AugLab augmentation config JSON (optional)")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs (default: 1)")
    parser.add_argument("--data-dir", default=None,
                        help="Override UROSEG_DATA / ~/uroseg/ with this path")
    # When invoked via cli.py, argv[1] == 'train' has already been stripped.
    # When called directly in tests with argv = ["uroseg", "train", ...], strip it here.
    argv = sys.argv[1:]
    if argv and argv[0] == "train":
        argv = argv[1:]
    args = parser.parse_args(argv)

    model = _utils.get_model(args.organ)
    data_path = _utils.resolve_data_path(args.data_dir)
    setup_nnunet_env(data_path)

    nnunet_task = model["nnunet_task"]
    dataset_id = extract_dataset_id(nnunet_task)

    raw_dir = data_path / "nnUNet" / "raw" / nnunet_task
    images_tr = raw_dir / "imagesTr"

    if not images_tr.exists():
        print(
            f"Error: training images directory not found: {images_tr}\n"
            f"Place training images in {images_tr}/ (filename pattern: <case>_0000.nii.gz)",
            file=sys.stderr,
        )
        sys.exit(1)

    # Auto-generate dataset.json
    dataset_json = generate_dataset_json(model, images_tr)
    dataset_json_path = raw_dir / "dataset.json"
    dataset_json_path.write_text(json.dumps(dataset_json, indent=2))
    print(f"Generated {dataset_json_path}")

    # Plan and preprocess if not already done
    preprocessed_dir = data_path / "nnUNet" / "preprocessed" / nnunet_task
    if not preprocessed_dir.exists():
        print("Running nnU-Net planning and preprocessing...")
        subprocess.run(
            ["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id), "--verify_dataset_integrity"],
            check=True,
        )

    # Register nnUNetTrainerDAExt into the nnunetv2 package directory.
    # auglab ships the trainer file inside its own package; nnU-Net discovers
    # trainers by scanning its own directory, so the file must be copied there
    # before nnUNetv2_train is invoked.
    from auglab.add_trainer import add_trainer as _add_trainer
    _add_trainer("nnUNetTrainerDAExt")

    # Set AugLab config env var if provided
    if args.auglab_config:
        os.environ["AUGLAB_CONFIG"] = str(args.auglab_config)

    # Run training
    results_dir = data_path / "nnUNet" / "results"
    print(f"Starting training (dataset {dataset_id}, fold {args.fold})...")
    subprocess.run(
        [
            "nnUNetv2_train",
            str(dataset_id),
            "3d_fullres",
            str(args.fold),
            "--trainer", "nnUNetTrainerDAExt",
        ],
        check=True,
    )

    trainer_dir = (
        results_dir
        / nnunet_task
        / "nnUNetTrainerDAExt__nnUNetPlans__3d_fullres"
        / f"fold_{args.fold}"
    )
    print(f"\nTraining complete. Model saved to:\n  {trainer_dir}")
