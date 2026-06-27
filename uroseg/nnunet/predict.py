from __future__ import annotations
import argparse
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uroseg.models.base import SegModel

from uroseg.utils.utils import (
    add_common_args, collect_niftis, build_output_path,
    resolve_data_path, data_dir_help,
)
from uroseg.utils.image import Image


def add_inference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('img', help='Input image file or folder')
    parser.add_argument('out', nargs='?', default='.', help='Output folder (default: current directory)')
    parser.add_argument('--fold', '-f', type=int, default=0, help='nnU-Net fold (default: 0)')
    parser.add_argument('--device', '-d', default='cuda', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--out-suffix', default='_seg', help='Output filename suffix')
    parser.add_argument('--out-prefix', default='', help='Output filename prefix')
    parser.add_argument('--data-dir', default=None, help=data_dir_help())
    add_common_args(parser)


def run_predict_cli(model: SegModel, args, largest_component: bool = False) -> None:
    """Orchestrate the full inference workflow for a given model."""
    from uroseg.nnunet.helpers import run_predict
    from uroseg.models.base import _find_model_dir

    data_path = resolve_data_path(args.data_dir)
    try:
        model_dir = _find_model_dir(model.name, data_path)
    except FileNotFoundError:
        print(f"Model '{model.name}' not installed — downloading...")
        model.install(data_path)
        model_dir = _find_model_dir(model.name, data_path)

    inputs = collect_niftis(args.img)
    if not inputs:
        print(f"No NIfTI files found in {args.img}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_in = Path(tmp) / 'inputs'
        tmp_out = Path(tmp) / 'outputs'
        tmp_in.mkdir()
        tmp_out.mkdir()

        if not args.quiet:
            print(f"Reorienting {len(inputs)} image(s) to RAS...")
        reoriented = []
        for i, p in enumerate(inputs):
            img = Image.load(p)
            img = img.reorient('RAS')
            out_p = tmp_in / f"{i:04d}_{p.name}"
            img.save(out_p)
            reoriented.append(out_p)

        run_predict(
            model_dir=model_dir,
            inputs=reoriented,
            output_dir=tmp_out,
            fold=args.fold,
            device=args.device,
        )

        for inp, pred_file in zip(inputs, sorted(tmp_out.glob('*.nii.gz'))):
            dest = build_output_path(inp, out_dir, args.out_prefix, args.out_suffix)
            if not args.overwrite and dest.exists():
                continue
            img = Image.load(pred_file)
            if largest_component:
                from uroseg.tools.largest_component import keep_largest_component
                img = Image(
                    data=keep_largest_component(img.data),
                    affine=img.affine,
                    header=img.header,
                )
            img.save(dest)

    if not args.quiet:
        print(f"Segmentations saved to {out_dir}")


def main() -> None:
    """Low-level nnU-Net prediction wrapper (uroseg predict_nnunet)."""
    from uroseg.models.base import _find_model_dir
    from uroseg.nnunet.helpers import run_predict

    parser = argparse.ArgumentParser(description='Low-level nnU-Net prediction wrapper.')
    parser.add_argument('--img', required=True, help='Input image file or folder')
    parser.add_argument('--out', required=True, help='Output folder')
    parser.add_argument('--task', required=True, help='nnUNet task name (e.g. Dataset001_Prostate)')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--data-dir', default=None, help=data_dir_help())
    add_common_args(parser)
    args = parser.parse_args()

    data_path = resolve_data_path(args.data_dir)
    # Derive model name from task name: DatasetNNN_Name -> name lower
    task_name = args.task.split('_', 1)[1].lower() if '_' in args.task else args.task.lower()
    try:
        model_dir = _find_model_dir(task_name, data_path)
    except FileNotFoundError:
        # Fallback: old nnUNet/results layout
        results_root = data_path / 'nnUNet' / 'results'
        direct = results_root / args.task
        if direct.exists():
            model_dir = direct
        else:
            raise
    inputs = collect_niftis(args.img)
    run_predict(model_dir, inputs, Path(args.out), fold=args.fold, device=args.device)
