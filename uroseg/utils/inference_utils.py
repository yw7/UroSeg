from __future__ import annotations
import argparse
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from uroseg.utils.image import Image
from uroseg.utils.utils import add_common_args, collect_niftis, build_output_path, resolve_data_path, data_dir_help
from uroseg.commands.predict_nnunet import find_model_dir, predict


def add_common_inference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--img', '-i', required=True, help='Input image file or folder')
    parser.add_argument('--out', '-o', required=True, help='Output folder')
    parser.add_argument('--fold', '-f', type=int, default=0, help='nnU-Net fold (default: 0)')
    parser.add_argument('--device', '-d', default='cuda', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--out-suffix', default='_seg', help='Output filename suffix')
    parser.add_argument('--out-prefix', default='', help='Output filename prefix')
    parser.add_argument('--data-dir', default=None, help=data_dir_help())
    add_common_args(parser)


def run_nnunet_predict(nnunet_task: str, args) -> None:
    data_path = resolve_data_path(args.data_dir)
    model_dir = find_model_dir(nnunet_task, data_path)

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

        predict(
            model_dir=model_dir,
            input_files=reoriented,
            output_dir=tmp_out,
            fold=args.fold,
            device=args.device,
        )

        for inp, pred_file in zip(inputs, sorted(tmp_out.glob('*.nii.gz'))):
            dest = build_output_path(inp, out_dir, args.out_prefix, args.out_suffix)
            if not args.overwrite and dest.exists():
                continue
            img = Image.load(pred_file)
            img.save(dest)

    if not args.quiet:
        print(f"Segmentations saved to {out_dir}")


def download_weights(url: str, destination: Path) -> None:
    import shutil
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    zip_name = url.split('/')[-1]
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / zip_name
        extract_tmp = Path(tmp) / 'extract'
        extract_tmp.mkdir()
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc=zip_name) as bar:
            def reporthook(count, block_size, total_size):
                if total_size > 0:
                    bar.total = total_size
                bar.update(block_size)
            urllib.request.urlretrieve(url, zip_path, reporthook=reporthook)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_tmp)
        for item in extract_tmp.iterdir():
            target = destination / item.name
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(item), str(target))
