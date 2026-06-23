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
