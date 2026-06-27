from __future__ import annotations
import argparse
import functools
import sys
from pathlib import Path

from tqdm.contrib.concurrent import process_map

from uroseg.utils.image import Image, save_nifti_image
from uroseg.utils.utils import add_common_args, collect_niftis, build_output_path


def process_one(triple: tuple[Path, Path, Path], args: argparse.Namespace) -> None:
    img_in, seg_in, img_out = triple
    img = Image.load(img_in)
    seg = Image.load(seg_in)
    cropped = img.crop_to_seg(seg, margin=args.margin)
    img_out.parent.mkdir(parents=True, exist_ok=True)
    save_nifti_image(cropped.data, cropped.affine, cropped.header, str(img_out))


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Crop image to the bounding box of a segmentation.'
    )
    parser.add_argument('--img', '-i', required=True, help='Input image file or folder')
    parser.add_argument('--seg', '-s', required=True, help='Input seg file or folder')
    parser.add_argument('--out', '-o', required=True, help='Output folder')
    parser.add_argument('--out-suffix', default='_crop', help='Output filename suffix')
    parser.add_argument('--out-prefix', default='', help='Output filename prefix')
    parser.add_argument('--margin', '-m', type=int, default=0, metavar='N',
                        help='Voxels of margin to add around bounding box (default: 0)')
    add_common_args(parser)
    args = parser.parse_args()

    imgs = collect_niftis(args.img)
    segs = collect_niftis(args.seg)

    if len(imgs) != len(segs):
        print(f"Mismatch: {len(imgs)} images vs {len(segs)} segs.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    triples = [
        (img, seg, build_output_path(img, out_dir, args.out_prefix, args.out_suffix))
        for img, seg in zip(imgs, segs)
        if args.overwrite
        or not build_output_path(img, out_dir, args.out_prefix, args.out_suffix).exists()
    ]

    process_map(
        functools.partial(process_one, args=args),
        triples,
        max_workers=args.max_workers,
        disable=args.quiet,
        desc='uroseg crop',
    )


def crop(
    input: Path | str,
    seg: Path | str,
    output: Path | str,
    margin: int = 0,
    out_suffix: str = "_crop",
    out_prefix: str = "",
    overwrite: bool = False,
) -> Path:
    import argparse
    input_path, seg_path, output_path = Path(input), Path(seg), Path(output)
    if not output_path.suffix:
        from uroseg.utils.utils import build_output_path
        output_path = build_output_path(input_path, output_path, out_prefix, out_suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    process_one(
        (input_path, seg_path, output_path),
        argparse.Namespace(margin=margin, overwrite=overwrite),
    )
    return output_path


def crop_dir(
    input_dir: Path | str,
    seg_dir: Path | str,
    output_dir: Path | str,
    margin: int = 0,
    out_suffix: str = "_crop",
    out_prefix: str = "",
    overwrite: bool = False,
    n_jobs: int = 1,
) -> None:
    import argparse, functools, sys
    from uroseg.utils.utils import collect_niftis, build_output_path
    imgs = collect_niftis(input_dir)
    segs = collect_niftis(seg_dir)
    if len(imgs) != len(segs):
        print(f"Mismatch: {len(imgs)} images vs {len(segs)} segs.", file=sys.stderr)
        return
    out = Path(output_dir)
    args = argparse.Namespace(margin=margin, overwrite=overwrite)
    triples = [
        (i, s, build_output_path(i, out, out_prefix, out_suffix))
        for i, s in zip(imgs, segs)
        if overwrite or not build_output_path(i, out, out_prefix, out_suffix).exists()
    ]
    process_map(
        functools.partial(process_one, args=args),
        triples,
        max_workers=n_jobs,
        disable=False,
        desc='uroseg crop',
    )
