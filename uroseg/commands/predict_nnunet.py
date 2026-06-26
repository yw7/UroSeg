from __future__ import annotations
import argparse
from pathlib import Path

from uroseg.utils.utils import add_common_args, resolve_data_path, collect_niftis, data_dir_help


def find_model_dir(nnunet_task: str, data_path: Path) -> Path:
    """Locate the trained model folder inside data_path/nnUNet/results/.

    Searches release ID subdirectories newest-first; falls back to direct
    task folder if weights were placed there manually.
    """
    results_root = data_path / 'nnUNet' / 'results'

    # direct placement (locally trained, no release ID subdir)
    direct = results_root / nnunet_task
    if direct.exists():
        trainer_dirs = list(direct.iterdir())
        if trainer_dirs:
            return direct

    # versioned release subdirs (sorted newest-first by name)
    if results_root.exists():
        release_dirs = sorted(
            (d for d in results_root.iterdir() if d.is_dir() and d.name != nnunet_task),
            reverse=True,
        )
        for release_dir in release_dirs:
            task_dir = release_dir / nnunet_task
            if task_dir.exists():
                return task_dir

    raise FileNotFoundError(
        f"Model weights for '{nnunet_task}' not found under {results_root}.\n"
        f"Run: uroseg install --model <organ>"
    )


def predict(
    model_dir: Path,
    input_files: list[Path],
    output_dir: Path,
    fold: int = 0,
    device: str = 'cuda',
    step_size: float = 0.5,
) -> None:
    """Run nnU-Net prediction on a list of input NIfTI files."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    output_dir.mkdir(parents=True, exist_ok=True)

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    elif device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    device_obj = torch.device(device)

    predictor = nnUNetPredictor(
        tile_step_size=step_size,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device_obj,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    # find checkpoint
    checkpoint = 'checkpoint_best.pth'
    fold_dir = model_dir
    # walk into trainer__plans__config/fold_N structure
    for child in model_dir.iterdir():
        if child.is_dir():
            fold_dir_candidate = child / f'fold_{fold}'
            if fold_dir_candidate.exists():
                fold_dir = child
                break

    predictor.initialize_from_trained_model_folder(
        str(fold_dir),
        use_folds=(fold,),
        checkpoint_name=checkpoint,
    )

    list_of_lists = [[str(f)] for f in input_files]
    output_names = [str(output_dir / f.name) for f in input_files]

    predictor.predict_from_files(
        list_of_lists,
        output_names,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Low-level nnU-Net prediction wrapper.'
    )
    parser.add_argument('--img', required=True, help='Input image file or folder')
    parser.add_argument('--out', required=True, help='Output folder')
    parser.add_argument('--task', required=True, help='nnUNet task name (e.g. Dataset001_Prostate)')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--data-dir', default=None, help=data_dir_help())
    add_common_args(parser)
    args = parser.parse_args()

    data_path = resolve_data_path(args.data_dir)
    model_dir = find_model_dir(args.task, data_path)
    inputs = collect_niftis(args.img)
    predict(model_dir, inputs, Path(args.out), fold=args.fold, device=args.device)
