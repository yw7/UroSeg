from __future__ import annotations
import argparse
import sys
import tempfile
from pathlib import Path

from uroseg.utils.image import Image
from uroseg.utils.utils import (
    add_common_args,
    collect_niftis,
    build_output_path,
    resolve_data_path,
    get_all_models,
)
from uroseg.commands.predict_nnunet import find_model_dir, predict


def build_inference_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run segmentation inference for a given organ model.'
    )
    parser.add_argument('organ', help='Organ model name (e.g. prostate, bladder)')
    parser.add_argument('--img', required=True, help='Input image file or folder')
    parser.add_argument('--out', required=True, help='Output folder')
    parser.add_argument('--fold', type=int, default=0, help='nnU-Net fold (default: 0)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--out-suffix', default='_seg', help='Output filename suffix')
    parser.add_argument('--out-prefix', default='', help='Output filename prefix')
    parser.add_argument('--data-dir', default=None, help='Override data path (or set UROSEG_DATA)')
    add_common_args(parser)
    return parser


def resolve_organ(organ: str) -> dict:
    available = get_all_models()
    if organ not in available:
        print(
            f"Unknown organ: '{organ}'\nAvailable: {', '.join(sorted(available))}",
            file=sys.stderr,
        )
        sys.exit(1)
    return available[organ]


def _reorient_to_tmp(input_path: Path, tmp_dir: Path, index: int) -> Path:
    img = Image.load(input_path)
    img = img.reorient('RAS')
    out = tmp_dir / f"{index:04d}_{input_path.name}"
    img.save(out)
    return out


def main() -> None:
    parser = build_inference_parser()
    args = parser.parse_args()

    model = resolve_organ(args.organ)
    data_path = resolve_data_path(args.data_dir)
    model_dir = find_model_dir(model['nnunet_task'], data_path)

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
        reoriented = [_reorient_to_tmp(p, tmp_in, i) for i, p in enumerate(inputs)]

        predict(
            model_dir=model_dir,
            input_files=reoriented,
            output_dir=tmp_out,
            fold=args.fold,
            device=args.device,
        )

        # copy predictions to final output with suffix/prefix naming
        for inp, pred_file in zip(inputs, sorted(tmp_out.glob('*.nii.gz'))):
            dest = build_output_path(inp, out_dir, args.out_prefix, args.out_suffix)
            if not args.overwrite and dest.exists():
                continue
            img = Image.load(pred_file)
            img.save(dest)

    if not args.quiet:
        print(f"Segmentations saved to {out_dir}")
