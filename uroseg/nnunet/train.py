from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import uroseg.utils.utils as _utils
from uroseg.utils.utils import data_dir_help
from uroseg.nnunet.helpers import setup_env, extract_dataset_id, generate_dataset_json


def _count_files(directory: Path, pattern: str) -> int:
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='uroseg train nnunet',
        description="Train a UroSeg model with nnU-Net and AugLab augmentation.",
    )
    parser.add_argument("organ", help="Organ name matching uroseg/models/<organ>.py")
    parser.add_argument("--fold", "-f", type=int, default=0,
                        help="nnU-Net fold number (0–4, default: 0)")
    parser.add_argument("--auglab-config", default=None,
                        help="Path to AugLab GPU augmentation config JSON (optional)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"],
                        help="Training device (default: cuda)")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs (default: 1)")
    parser.add_argument("--training-dir", "-d", default=None,
                        help="Directory containing training data (default: current dir). "
                             "Raw data expected at <training-dir>/nnUNet/raw/<task>/")
    parser.add_argument("--data-dir", default=None, help=data_dir_help())
    args = parser.parse_args()

    mod = _utils.load_model_module(args.organ)
    training_path = Path(args.training_dir).resolve() if args.training_dir else Path.cwd()
    setup_env(training_path)

    nnunet_task = mod.NNUNET_TASK
    dataset_id = extract_dataset_id(nnunet_task)

    raw_dir = training_path / "nnUNet" / "raw" / nnunet_task
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
    (raw_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    print(f"Generated {raw_dir / 'dataset.json'}")

    preprocessed_dir = training_path / "nnUNet" / "preprocessed" / nnunet_task

    if not (preprocessed_dir / "dataset_fingerprint.json").exists():
        print("Extracting dataset fingerprint...")
        subprocess.run(["nnUNetv2_extract_fingerprint", "-d", str(dataset_id)], check=True)

    if not (preprocessed_dir / "nnUNetPlans.json").exists():
        print("Planning experiment...")
        subprocess.run(["nnUNetv2_plan_experiment", "-d", str(dataset_id)], check=True)

    plans_file = preprocessed_dir / "nnUNetPlans.json"
    data_identifier = "nnUNetPlans_3d_fullres"
    if plans_file.exists():
        plans = json.loads(plans_file.read_text())
        identifier = plans.get("configurations", {}).get("3d_fullres", {}).get("data_identifier")
        if identifier:
            data_identifier = identifier
    preprocessed_data_dir = preprocessed_dir / data_identifier

    n_labels = _count_files(labels_tr, "*.nii.gz")
    n_pkl = _count_files(preprocessed_data_dir, "*.pkl")
    if not preprocessed_data_dir.exists() or n_pkl != n_labels:
        reason = (f"{preprocessed_data_dir.name} not found"
                  if not preprocessed_data_dir.exists()
                  else f"{n_pkl} .pkl vs {n_labels} labels in labelsTr")
        print(f"Preprocessing dataset (3d_fullres) [{reason}]...")
        subprocess.run(
            ["nnUNetv2_preprocess", "-d", str(dataset_id), "-c", "3d_fullres"],
            check=True,
        )
    else:
        print(f"Preprocessing already done ({n_pkl} samples), skipping.")

    from auglab.add_trainer import add_trainer as _add_trainer
    _add_trainer("nnUNetTrainerDAExtGPU")

    if args.auglab_config:
        os.environ["AUGLAB_PARAMS_GPU_JSON"] = str(args.auglab_config)
    elif "AUGLAB_PARAMS_GPU_JSON" not in os.environ:
        from importlib.resources import files as _res_files
        import uroseg.resources.auglab as _auglab_res
        bundled = _res_files(_auglab_res) / "transform_params_gpu.json"
        os.environ["AUGLAB_PARAMS_GPU_JSON"] = str(bundled)

    results_dir = training_path / "nnUNet" / "results"
    exports_dir = training_path / "nnUNet" / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    trainer_tag = "nnUNetTrainerDAExtGPU__nnUNetPlans__3d_fullres"

    n_npy = _count_files(preprocessed_data_dir, "*.npy")
    n_npz = _count_files(preprocessed_data_dir, "*.npz")
    train_cmd = [
        "nnUNetv2_train",
        str(dataset_id), "3d_fullres", str(args.fold),
        "-tr", "nnUNetTrainerDAExtGPU", "--c", "-device", args.device,
    ]
    if n_npy == 2 * n_npz and n_npz > 0:
        train_cmd.append("--use_compressed")
    print(f"Starting training (dataset {dataset_id}, fold {args.fold})...")
    subprocess.run(train_cmd, check=True)

    zip_name = f"{nnunet_task}__{trainer_tag}__fold_{args.fold}.zip"
    zip_path = exports_dir / zip_name
    (results_dir / nnunet_task / "ensembles").mkdir(parents=True, exist_ok=True)
    print(f"Exporting model to {zip_path}...")
    subprocess.run([
        "nnUNetv2_export_model_to_zip",
        "-d", str(dataset_id), "-o", str(zip_path),
        "-c", "3d_fullres", "-f", str(args.fold),
        "-tr", "nnUNetTrainerDAExtGPU",
    ], check=True)

    images_ts = raw_dir / "imagesTs"
    labels_ts = raw_dir / "labelsTs"
    test_pred_dir = results_dir / nnunet_task / trainer_tag / f"fold_{args.fold}" / "test"

    if images_ts.exists() and _count_files(images_ts, "*.nii.gz") > 0:
        test_pred_dir.mkdir(parents=True, exist_ok=True)
        print("Predicting on test set...")
        subprocess.run([
            "nnUNetv2_predict",
            "-d", str(dataset_id), "-i", str(images_ts),
            "-o", str(test_pred_dir), "-f", str(args.fold),
            "-c", "3d_fullres", "-tr", "nnUNetTrainerDAExtGPU",
        ], check=True)

        if labels_ts.exists() and _count_files(labels_ts, "*.nii.gz") > 0:
            trainer_results_dir = results_dir / nnunet_task / trainer_tag
            print("Evaluating test predictions...")
            subprocess.run([
                "nnUNetv2_evaluate_folder",
                str(labels_ts), str(test_pred_dir),
                "-djfile", str(trainer_results_dir / "dataset.json"),
                "-pfile", str(trainer_results_dir / "plans.json"),
            ], check=True)
            summary_json = test_pred_dir / "summary.json"
            if summary_json.exists():
                with zipfile.ZipFile(zip_path, 'a') as zf:
                    zf.write(summary_json, arcname=str(summary_json.relative_to(results_dir)))

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
