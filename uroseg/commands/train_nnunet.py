from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import uroseg.utils.utils as _utils
from uroseg.utils.utils import normalize_labels, data_dir_help


def extract_dataset_id(nnunet_task: str) -> int:
    match = re.match(r'Dataset(\d+)_', nnunet_task)
    if not match:
        raise ValueError(f"Invalid nnunet_task format: {nnunet_task!r} — expected 'DatasetNNN_Name'")
    return int(match.group(1))


def generate_dataset_json(model, images_tr_dir: Path) -> dict:
    labels = normalize_labels(model.labels)

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
    os.environ["nnUNet_exports"] = str(nnunet_dir / "exports")


def _count_files(directory: Path, pattern: str) -> int:
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='uroseg train nnunet',
        description="Train a UroSeg model with nnU-Net and AugLab augmentation.",
    )
    parser.add_argument("organ", help="Organ name matching resources/models/<organ>.py")
    parser.add_argument("--fold", "-f", type=int, default=0,
                        help="nnU-Net fold number (0–4, default: 0)")
    parser.add_argument("--auglab-config", default=None,
                        help="Path to AugLab augmentation config JSON (optional)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"],
                        help="Training device (default: cuda)")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs (default: 1)")
    parser.add_argument("--data-dir", default=None, help=data_dir_help())
    args = parser.parse_args()

    mod = _utils.load_model_module(args.organ)
    data_path = _utils.resolve_data_path(args.data_dir)
    setup_nnunet_env(data_path)

    nnunet_task = mod.NNUNET_TASK
    dataset_id = extract_dataset_id(nnunet_task)

    raw_dir = data_path / "nnUNet" / "raw" / nnunet_task
    images_tr = raw_dir / "imagesTr"
    labels_tr = raw_dir / "labelsTr"

    if not images_tr.exists():
        print(
            f"Error: training images directory not found: {images_tr}\n"
            f"Place training images in {images_tr}/ (filename pattern: <case>_0000.nii.gz)",
            file=sys.stderr,
        )
        sys.exit(1)

    dataset_json = generate_dataset_json(mod.MODEL, images_tr)
    dataset_json_path = raw_dir / "dataset.json"
    dataset_json_path.write_text(json.dumps(dataset_json, indent=2))
    print(f"Generated {dataset_json_path}")

    preprocessed_dir = data_path / "nnUNet" / "preprocessed" / nnunet_task

    # Extract fingerprint (skip if already done)
    if not (preprocessed_dir / "dataset_fingerprint.json").exists():
        print("Extracting dataset fingerprint...")
        subprocess.run(
            ["nnUNetv2_extract_fingerprint", "-d", str(dataset_id)],
            check=True,
        )

    # Plan experiment (skip if already done)
    if not (preprocessed_dir / "nnUNetPlans.json").exists():
        print("Planning experiment...")
        subprocess.run(
            ["nnUNetv2_plan_experiment", "-d", str(dataset_id)],
            check=True,
        )

    # Read data_identifier from plans file (nnU-Net stores this inside nnUNetPlans.json)
    # so we derive the preprocessed data directory name rather than hardcoding it
    plans_file = preprocessed_dir / "nnUNetPlans.json"
    data_identifier = "nnUNetPlans_3d_fullres"  # fallback
    if plans_file.exists():
        plans = json.loads(plans_file.read_text())
        identifier = plans.get("configurations", {}).get("3d_fullres", {}).get("data_identifier")
        if identifier:
            data_identifier = identifier
    preprocessed_data_dir = preprocessed_dir / data_identifier

    # Preprocess 3d_fullres only (skip if .pkl count matches label count).
    # .pkl files are always written 1-per-case by nnU-Net regardless of whether
    # data is stored as .npz or .npy, so they are a reliable completion marker.
    n_labels = _count_files(labels_tr, "*.nii.gz")
    n_pkl = _count_files(preprocessed_data_dir, "*.pkl")
    if not preprocessed_data_dir.exists() or n_pkl != n_labels:
        if not preprocessed_data_dir.exists():
            reason = f"{preprocessed_data_dir.name} not found"
        else:
            reason = f"{n_pkl} .pkl vs {n_labels} labels in labelsTr"
        print(f"Preprocessing dataset (3d_fullres) [{reason}]...")
        subprocess.run(
            ["nnUNetv2_preprocess", "-d", str(dataset_id), "-c", "3d_fullres"],
            check=True,
        )
    else:
        print(f"Preprocessing already done ({n_pkl} samples), skipping.")

    from auglab.add_trainer import add_trainer as _add_trainer
    _add_trainer("nnUNetTrainerDAExt")

    if args.auglab_config:
        os.environ["AUGLAB_CONFIG"] = str(args.auglab_config)

    results_dir = data_path / "nnUNet" / "results"
    exports_dir = data_path / "nnUNet" / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    trainer_tag = "nnUNetTrainerDAExtGPU__nnUNetPlans__3d_fullres"

    # Train (--c resumes from checkpoint if interrupted)
    n_npy = _count_files(preprocessed_data_dir, "*.npy")
    n_npz_now = _count_files(preprocessed_data_dir, "*.npz")
    train_cmd = [
        "nnUNetv2_train",
        str(dataset_id), "3d_fullres", str(args.fold),
        "-tr", "nnUNetTrainerDAExtGPU", "--c", "-device", args.device,
    ]
    if n_npy == 2 * n_npz_now and n_npz_now > 0:
        train_cmd.append("--use_compressed")
    print(f"Starting training (dataset {dataset_id}, fold {args.fold})...")
    subprocess.run(train_cmd, check=True)

    # Export model to zip
    zip_name = f"{nnunet_task}__{trainer_tag}__fold_{args.fold}.zip"
    zip_path = exports_dir / zip_name
    (results_dir / nnunet_task / "ensembles").mkdir(parents=True, exist_ok=True)
    print(f"Exporting model to {zip_path}...")
    subprocess.run(
        [
            "nnUNetv2_export_model_to_zip",
            "-d", str(dataset_id),
            "-o", str(zip_path),
            "-c", "3d_fullres",
            "-f", str(args.fold),
            "-tr", "nnUNetTrainerDAExtGPU",
        ],
        check=True,
    )

    # Predict and evaluate test set (if imagesTs/ and labelsTs/ exist)
    images_ts = raw_dir / "imagesTs"
    labels_ts = raw_dir / "labelsTs"
    test_pred_dir = results_dir / nnunet_task / trainer_tag / f"fold_{args.fold}" / "test"

    if images_ts.exists() and _count_files(images_ts, "*.nii.gz") > 0:
        test_pred_dir.mkdir(parents=True, exist_ok=True)
        print("Predicting on test set...")
        subprocess.run(
            [
                "nnUNetv2_predict",
                "-d", str(dataset_id),
                "-i", str(images_ts),
                "-o", str(test_pred_dir),
                "-f", str(args.fold),
                "-c", "3d_fullres",
                "-tr", "nnUNetTrainerDAExtGPU",
            ],
            check=True,
        )

        if labels_ts.exists() and _count_files(labels_ts, "*.nii.gz") > 0:
            trainer_results_dir = results_dir / nnunet_task / trainer_tag
            print("Evaluating test predictions...")
            subprocess.run(
                [
                    "nnUNetv2_evaluate_folder",
                    str(labels_ts),
                    str(test_pred_dir),
                    "-djfile", str(trainer_results_dir / "dataset.json"),
                    "-pfile", str(trainer_results_dir / "plans.json"),
                ],
                check=True,
            )

            summary_json = test_pred_dir / "summary.json"
            if summary_json.exists():
                print("Adding summary.json to export zip...")
                with zipfile.ZipFile(zip_path, 'a') as zf:
                    zf.write(summary_json, arcname=str(summary_json.relative_to(results_dir)))

    # Add dataset file list and splits to zip
    dataset_listing = results_dir / nnunet_task / "dataset_files.json"
    (results_dir / nnunet_task).mkdir(parents=True, exist_ok=True)
    listing: dict[str, list[str]] = {}
    for item in sorted(raw_dir.iterdir()):
        if item.is_dir():
            listing[item.name] = sorted(f.name for f in item.iterdir())
    dataset_listing.write_text(json.dumps(listing, indent=2))
    with zipfile.ZipFile(zip_path, 'a') as zf:
        zf.write(dataset_listing, arcname=str(dataset_listing.relative_to(results_dir)))

    splits_json = preprocessed_dir / "splits_final.json"
    if splits_json.exists():
        with zipfile.ZipFile(zip_path, 'a') as zf:
            zf.write(splits_json, arcname=str(splits_json.relative_to(preprocessed_dir.parent)))

    print(f"\nTraining complete.")
    print(f"  Model: {results_dir / nnunet_task / trainer_tag / f'fold_{args.fold}'}")
    print(f"  Export: {zip_path}")
