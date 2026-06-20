from __future__ import annotations
import argparse
import functools
from pathlib import Path

import numpy as np
from tqdm.contrib.concurrent import process_map

from uroseg.utils.image import Image
from uroseg.utils.utils import add_common_args, collect_niftis, build_output_path


def crop_to_seg(img: Image, seg: Image) -> tuple[Image, Image]:
    bb = seg.bounding_box(label=None)
    if bb is None:
        return img.copy(), seg.copy()
    # Shift the affine origin to the crop start voxel
    start = np.array([s.start for s in bb], dtype=float)
    img_affine = img.affine.copy()
    img_affine[:3, 3] = img.affine[:3, :3] @ start + img.affine[:3, 3]
    seg_affine = seg.affine.copy()
    seg_affine[:3, 3] = seg.affine[:3, :3] @ start + seg.affine[:3, 3]
    cropped_img = Image(img.data[bb], img_affine, img.header)
    cropped_seg = Image(seg.data[bb], seg_affine, seg.header)
    return cropped_img, cropped_seg


def process_one(
    pair: tuple[Path, Path, Path, Path],
    args: argparse.Namespace,
) -> None:
    img_in, seg_in, img_out, seg_out = pair
    img = Image.load(img_in)
    seg = Image.load(seg_in)
    cropped_img, cropped_seg = crop_to_seg(img, seg)
    img_out.parent.mkdir(parents=True, exist_ok=True)
    seg_out.parent.mkdir(parents=True, exist_ok=True)
    cropped_img.save(img_out)
    cropped_seg.save(seg_out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Crop image and segmentation to the bounding box of the segmentation.'
    )
    parser.add_argument('--img', required=True, help='Input image file or folder')
    parser.add_argument('--seg', required=True, help='Input seg file or folder')
    parser.add_argument('--out-img', required=True, help='Output image folder')
    parser.add_argument('--out-seg', required=True, help='Output seg folder')
    parser.add_argument('--img-suffix', default='_crop', help='Suffix for output images')
    parser.add_argument('--img-prefix', default='', help='Prefix for output images')
    parser.add_argument('--seg-suffix', default='_crop', help='Suffix for output segs')
    parser.add_argument('--seg-prefix', default='', help='Prefix for output segs')
    add_common_args(parser)
    args = parser.parse_args()

    imgs = collect_niftis(args.img)
    segs = collect_niftis(args.seg)

    if len(imgs) != len(segs):
        import sys
        print(f"Mismatch: {len(imgs)} images vs {len(segs)} segs.", file=sys.stderr)
        sys.exit(1)

    out_img_dir = Path(args.out_img)
    out_seg_dir = Path(args.out_seg)

    pairs = [
        (
            i, s,
            build_output_path(i, out_img_dir, args.img_prefix, args.img_suffix),
            build_output_path(s, out_seg_dir, args.seg_prefix, args.seg_suffix),
        )
        for i, s in zip(imgs, segs)
        if args.overwrite
        or not build_output_path(i, out_img_dir, args.img_prefix, args.img_suffix).exists()
        or not build_output_path(s, out_seg_dir, args.seg_prefix, args.seg_suffix).exists()
    ]

    process_map(
        functools.partial(process_one, args=args),
        pairs,
        max_workers=args.max_workers,
        disable=args.quiet,
        desc='uroseg crop',
    )
