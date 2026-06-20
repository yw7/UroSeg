from __future__ import annotations
import argparse
import functools
from pathlib import Path

import numpy as np
import nibabel as nib
import nibabel.processing as nibp
from tqdm.contrib.concurrent import process_map

from uroseg.utils.image import Image
from uroseg.utils.utils import add_common_args, collect_niftis, build_output_path


def resample_seg_to_image(seg: Image, ref: Image) -> Image:
    seg_nib = nib.Nifti1Image(seg.data.astype(np.int32), seg.affine)
    ref_nib = nib.Nifti1Image(ref.data, ref.affine, ref.header)
    resampled = nibp.resample_from_to(seg_nib, ref_nib, order=0, cval=0)
    return Image(
        data=np.asanyarray(resampled.dataobj).astype(seg.data.dtype),
        affine=resampled.affine,
        header=ref.header,
    )


def process_one(
    pair: tuple[Path, Path, Path],
    args: argparse.Namespace,
) -> None:
    seg_path, img_path, out_path = pair
    seg = Image.load(seg_path)
    ref = Image.load(img_path)
    result = resample_seg_to_image(seg, ref)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Resample segmentation to match reference image space (nearest-neighbour).'
    )
    parser.add_argument('--seg', required=True, help='Input seg file or folder')
    parser.add_argument('--img', required=True, help='Reference image file or folder')
    parser.add_argument('--out-seg', required=True, help='Output seg folder')
    parser.add_argument('--seg-suffix', default='_transformed', help='Output seg suffix')
    parser.add_argument('--seg-prefix', default='', help='Output seg prefix')
    add_common_args(parser)
    args = parser.parse_args()

    segs = collect_niftis(args.seg)
    imgs = collect_niftis(args.img)

    if len(segs) != len(imgs):
        import sys
        print(f"Mismatch: {len(segs)} segs vs {len(imgs)} images.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_seg)
    pairs = [
        (s, i, build_output_path(s, out_dir, args.seg_prefix, args.seg_suffix))
        for s, i in zip(segs, imgs)
        if args.overwrite
        or not build_output_path(s, out_dir, args.seg_prefix, args.seg_suffix).exists()
    ]

    process_map(
        functools.partial(process_one, args=args),
        pairs,
        max_workers=args.max_workers,
        disable=args.quiet,
        desc='uroseg transform_seg2image',
    )
