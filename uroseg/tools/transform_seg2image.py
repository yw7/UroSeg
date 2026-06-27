from __future__ import annotations
import argparse
import functools
import math
from pathlib import Path

import numpy as np
import nibabel as nib
import nibabel.processing as nibp
import scipy.ndimage as ndi
from tqdm.contrib.concurrent import process_map

from uroseg.utils.image import Image, save_nifti_image, save_nifti_seg
from uroseg.utils.utils import add_common_args, collect_niftis, build_output_path


def _resample_label_mode(seg_nib: nib.Nifti1Image, ref_nib: nib.Nifti1Image) -> np.ndarray:
    """Resample single-voxel landmarks: dilate each label, resample nearest, place at CoM."""
    seg_data = np.asanyarray(seg_nib.dataobj).round().astype(np.uint8)
    seg_zooms = np.array(seg_nib.header.get_zooms()[:3])
    ref_zooms = np.array(ref_nib.header.get_zooms()[:3])

    dilation_size = math.ceil(np.max(ref_zooms / seg_zooms))
    pad_width = int(dilation_size * math.ceil(np.max(seg_zooms / ref_zooms)))

    # Pad seg data and adjust affine
    seg_padded = np.pad(seg_data, pad_width)
    pad_offset = seg_nib.affine[:3, :3] @ np.array([pad_width] * 3)
    padded_affine = seg_nib.affine.copy()
    padded_affine[:3, 3] -= pad_offset
    seg_padded_nib = nib.Nifti1Image(seg_padded, padded_affine)

    # Dilate each label
    dilated = np.zeros_like(seg_padded)
    for lbl in np.unique(seg_padded):
        if lbl == 0:
            continue
        mask = (seg_padded == lbl)
        for _ in range(dilation_size):
            mask = ndi.binary_dilation(mask)
        dilated[mask] = lbl

    dilated_nib = nib.Nifti1Image(dilated, padded_affine)

    # Resample dilated seg to ref space (nearest)
    resampled_nib = nibp.resample_from_to(dilated_nib, ref_nib, order=0, cval=0)
    resampled = np.asanyarray(resampled_nib.dataobj).round().astype(np.uint8)

    # Pad ref for output, place each label at its center of mass
    ref_padded = np.pad(resampled, pad_width)
    output = np.zeros_like(ref_padded)
    for lbl in np.unique(resampled):
        if lbl == 0:
            continue
        coords = ndi.center_of_mass(resampled == lbl)
        idx = tuple(
            int(np.clip(round(c) + pad_width, pad_width, ref_padded.shape[i] - pad_width - 1))
            for i, c in enumerate(coords)
        )
        output[idx] = lbl

    # Remove padding
    s = slice(pad_width, -pad_width if pad_width > 0 else None)
    output = output[s, s, s]
    return output


def resample_seg_to_image(seg: Image, ref: Image, interpolation: str = 'nearest') -> Image:
    seg_nib = nib.Nifti1Image(seg.data.astype(np.int32), seg.affine)
    ref_nib = nib.Nifti1Image(ref.data, ref.affine, ref.header)

    if interpolation == 'nearest':
        resampled = nibp.resample_from_to(seg_nib, ref_nib, order=0, cval=0)
        data = np.asanyarray(resampled.dataobj).astype(seg.data.dtype)
        return Image(data=data, affine=resampled.affine, header=ref.header)

    elif interpolation == 'linear':
        resampled = nibp.resample_from_to(seg_nib, ref_nib, order=1, cval=0)
        data = np.asanyarray(resampled.dataobj).astype(np.float32)
        return Image(data=data, affine=resampled.affine, header=ref.header)

    elif interpolation == 'label':
        data = _resample_label_mode(seg_nib, ref_nib)
        return Image(data=data, affine=ref.affine, header=ref.header)

    else:
        raise ValueError(f"Unknown interpolation mode: {interpolation!r}")


def process_one(
    pair: tuple[Path, Path, Path],
    args: argparse.Namespace,
) -> None:
    seg_path, img_path, out_path = pair
    seg = Image.load(seg_path)
    ref = Image.load(img_path)
    result = resample_seg_to_image(seg, ref, interpolation=args.interpolation)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.interpolation == 'linear':
        save_nifti_image(result.data, result.affine, result.header, str(out_path))
    else:
        save_nifti_seg(result.data, result.affine, result.header, str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Resample segmentation to match reference image space.'
    )
    parser.add_argument('--seg', '-s', required=True, help='Input seg file or folder')
    parser.add_argument('--img', '-i', required=True, help='Reference image file or folder')
    parser.add_argument('--out-seg', required=True, help='Output seg folder')
    parser.add_argument('--seg-suffix', default='_transformed', help='Output seg suffix')
    parser.add_argument('--seg-prefix', default='', help='Output seg prefix')
    parser.add_argument('--interpolation', '-x', default='nearest',
                        choices=['nearest', 'linear', 'label'],
                        help='Interpolation method: nearest (default), linear, or label '
                             '(for single-voxel landmarks — dilate→resample→place at CoM)')
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


def transform_seg2image(
    seg: Path | str,
    img: Path | str,
    output: Path | str,
    interpolation: str = 'nearest',
    seg_suffix: str = "_transformed",
    seg_prefix: str = "",
    overwrite: bool = False,
) -> Path:
    import argparse
    seg_path, img_path, output_path = Path(seg), Path(img), Path(output)
    if not output_path.suffix:
        from uroseg.utils.utils import build_output_path
        output_path = build_output_path(seg_path, output_path, seg_prefix, seg_suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    process_one(
        (seg_path, img_path, output_path),
        argparse.Namespace(interpolation=interpolation, overwrite=overwrite),
    )
    return output_path


def transform_seg2image_dir(
    seg_dir: Path | str,
    img_dir: Path | str,
    output_dir: Path | str,
    interpolation: str = 'nearest',
    seg_suffix: str = "_transformed",
    seg_prefix: str = "",
    overwrite: bool = False,
    n_jobs: int = 1,
) -> None:
    import argparse, functools, sys
    from uroseg.utils.utils import collect_niftis, build_output_path
    segs = collect_niftis(seg_dir)
    imgs = collect_niftis(img_dir)
    if len(segs) != len(imgs):
        print(f"Mismatch: {len(segs)} segs vs {len(imgs)} images.", file=sys.stderr)
        return
    out = Path(output_dir)
    args = argparse.Namespace(interpolation=interpolation, overwrite=overwrite)
    pairs = [
        (s, i, build_output_path(s, out, seg_prefix, seg_suffix))
        for s, i in zip(segs, imgs)
        if overwrite or not build_output_path(s, out, seg_prefix, seg_suffix).exists()
    ]
    process_map(
        functools.partial(process_one, args=args),
        pairs,
        max_workers=n_jobs,
        disable=False,
        desc='uroseg transform_seg2image',
    )
