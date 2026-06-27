from __future__ import annotations
import contextlib
import io
import os
import re
import warnings
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


@contextlib.contextmanager
def _suppress_nnunet():
    """Silence nnunet env-var warnings, per-case prints, and Python warnings."""
    env_keys = ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results')
    saved = {}
    for k in env_keys:
        saved[k] = os.environ.get(k)
        if k not in os.environ:
            os.environ[k] = ''
    buf = io.StringIO()
    try:
        with warnings.catch_warnings(), \
             contextlib.redirect_stdout(buf), \
             contextlib.redirect_stderr(buf):
            warnings.simplefilter('ignore')
            yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _init_predictor(model_dir: Path, fold: int = 0, device: str = 'cuda'):
    """Create and initialize an nnUNetPredictor (suppresses nnunet output)."""
    import torch

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    elif device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'

    fold_dir, checkpoint_name = _resolve_fold_dir(model_dir, fold)
    with _suppress_nnunet():
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device(device),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            str(fold_dir),
            use_folds=(fold,),
            checkpoint_name=checkpoint_name,
        )
    return predictor


def _run_inference(predictor, img: Image) -> 'np.ndarray':
    """Run predict_single_npy_array on img (already 1mm canonical). Suppresses output."""
    import numpy as np
    spacing = [float(s) for s in img.header.get_zooms()[:3]]
    input_array = img.data[np.newaxis].astype(np.float32)
    with _suppress_nnunet():
        seg_array = predictor.predict_single_npy_array(input_array, {'spacing': spacing})
    return seg_array


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


def _resolve_fold_dir(model_dir: Path, fold: int) -> tuple[Path, str]:
    """Return (trainer_dir, checkpoint_name) for the given fold."""
    fold_matches = sorted(model_dir.glob(f'**/fold_{fold}'))
    trainer_dir = fold_matches[0].parent if fold_matches else model_dir
    for name in ('checkpoint_best.pth', 'checkpoint_final.pth'):
        if (trainer_dir / f'fold_{fold}' / name).exists():
            return trainer_dir, name
    return trainer_dir, 'checkpoint_best.pth'


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

    fold_dir, checkpoint_name = _resolve_fold_dir(model_dir, fold)

    predictor.initialize_from_trained_model_folder(
        str(fold_dir),
        use_folds=(fold,),
        checkpoint_name=checkpoint_name,
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
    predictor = _init_predictor(model_dir, fold=fold, device=device)
    seg_array = _run_inference(predictor, img)
    return Image(data=seg_array, affine=img.affine, header=img.header)
