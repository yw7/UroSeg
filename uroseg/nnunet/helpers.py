from __future__ import annotations
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from uroseg.utils.image import Image

if TYPE_CHECKING:
    from uroseg.models.base import SegModel


def setup_env(data_dir: Path) -> None:
    nnunet_dir = data_dir / "nnUNet"
    os.environ["nnUNet_raw"] = str(nnunet_dir / "raw")
    os.environ["nnUNet_preprocessed"] = str(nnunet_dir / "preprocessed")
    os.environ["nnUNet_results"] = str(nnunet_dir / "results")
    os.environ["nnUNet_exports"] = str(nnunet_dir / "exports")


def extract_dataset_id(nnunet_task: str) -> int:
    match = re.match(r'Dataset(\d+)_', nnunet_task)
    if not match:
        raise ValueError(f"Invalid nnunet_task format: {nnunet_task!r} — expected 'DatasetNNN_Name'")
    return int(match.group(1))


def generate_dataset_json(model: SegModel, images_tr: Path) -> dict:
    from uroseg.utils.utils import normalize_labels
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
        "numTraining": len(list(images_tr.glob("*.nii.gz"))),
        "file_ending": ".nii.gz",
    }
    if has_regions:
        dataset["regions_class_order"] = sorted(all_values)
    return dataset


def run_predict(
    model_dir: Path,
    inputs: list[Path],
    output_dir: Path,
    fold: int = 0,
    device: str = 'cuda',
) -> None:
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    output_dir.mkdir(parents=True, exist_ok=True)

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    elif device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    device_obj = torch.device(device)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device_obj,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    # Find the trainer dir that contains fold_N (handles 1 or 2 levels of nesting)
    fold_matches = sorted(model_dir.glob(f'**/fold_{fold}'))
    fold_dir = fold_matches[0].parent if fold_matches else model_dir

    predictor.initialize_from_trained_model_folder(
        str(fold_dir),
        use_folds=(fold,),
        checkpoint_name='checkpoint_best.pth',
    )

    list_of_lists = [[str(f)] for f in inputs]
    output_names = [str(output_dir / f.name) for f in inputs]
    predictor.predict_from_files(
        list_of_lists,
        output_names,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )


def run_predict_array(
    model_dir: Path,
    img: Image,
    fold: int = 0,
    device: str = 'cuda',
) -> Image:
    """Run nnunet inference on an in-memory Image. Returns seg Image in same space as input."""
    import numpy as np
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    elif device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    fold_matches = sorted(model_dir.glob(f'**/fold_{fold}'))
    fold_dir = fold_matches[0].parent if fold_matches else model_dir
    predictor.initialize_from_trained_model_folder(
        str(fold_dir),
        use_folds=(fold,),
        checkpoint_name='checkpoint_best.pth',
    )

    spacing = [float(s) for s in img.header.get_zooms()[:3]]

    input_array = img.data[np.newaxis].astype(np.float32)  # (1, x, y, z)
    seg_array = predictor.predict_single_npy_array(
        input_array,
        {'spacing': spacing},
        segmentation_previous_stage=None,
        output_file_truncated=None,
        save_probabilities=False,
    )
    return Image(data=seg_array, affine=img.affine, header=img.header)
