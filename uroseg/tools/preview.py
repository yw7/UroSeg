from __future__ import annotations
import argparse
import functools
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage, ImageDraw
from tqdm.contrib.concurrent import process_map

from uroseg.utils.image import Image
from uroseg.utils.utils import add_common_args, collect_niftis

LABEL_COLORS = [
    (255, 80, 80),
    (80, 255, 80),
    (80, 80, 255),
    (255, 255, 80),
    (255, 80, 255),
    (80, 255, 255),
]


def _parse_label_text(args_list: list[str]) -> dict[int, str]:
    """Parse --label-text-right / --label-text-left values.

    Accepts either:
    - a single JSON file path (e.g. ``labels.json``)
    - one or more ``label:text`` pairs  (e.g. ``1:L1 2:L2``)

    Returns a ``{label_int: text_str}`` dict.
    """
    if not args_list:
        return {}
    if len(args_list) == 1 and args_list[0].endswith('.json'):
        with open(args_list[0]) as f:
            return {int(k): str(v) for k, v in json.load(f).items()}
    return {int(p.split(':')[0]): p.split(':', 1)[1] for p in args_list}


def make_preview(
    img_data: np.ndarray,
    seg_data: np.ndarray | None = None,
    orient: str = 'sag',
    sliceloc: float = 0.5,
    label_text_right: dict[int, str] | None = None,
    label_text_left: dict[int, str] | None = None,
) -> np.ndarray:
    """Generate a single-slice RGB preview image.

    Parameters
    ----------
    img_data:
        3-D image array.
    seg_data:
        Optional 3-D segmentation array with the same shape as *img_data*.
    orient:
        Slice orientation: ``'sag'`` (sagittal, axis 0), ``'cor'`` (coronal,
        axis 1), or ``'ax'`` (axial, axis 2).
    sliceloc:
        Fractional position along the chosen axis, in [0.0, 1.0].
    label_text_right:
        ``{label_int: text}`` mapping for labels whose text is drawn on the
        right side of the slice.
    label_text_left:
        Same but drawn on the left side.

    Returns
    -------
    np.ndarray
        HxWx3 ``uint8`` RGB array.
    """
    axis_map = {'sag': 0, 'cor': 1, 'ax': 2}
    axis = axis_map[orient]
    idx = int(sliceloc * img_data.shape[axis])
    idx = max(0, min(idx, img_data.shape[axis] - 1))

    # Extract 2-D slice
    slice_img = np.take(img_data, idx, axis=axis).astype(float)
    mn, mx = slice_img.min(), slice_img.max()
    if mx > mn:
        slice_norm = ((slice_img - mn) / (mx - mn) * 255).astype(np.uint8)
    else:
        slice_norm = np.zeros_like(slice_img, dtype=np.uint8)

    # Greyscale → RGB
    rgb = np.stack([slice_norm] * 3, axis=-1)

    # Overlay segmentation colours
    if seg_data is not None:
        seg_slice = np.take(seg_data, idx, axis=axis)
        for label_id in np.unique(seg_slice):
            if label_id == 0:
                continue
            color = LABEL_COLORS[(int(label_id) - 1) % len(LABEL_COLORS)]
            mask = seg_slice == label_id
            for c, val in enumerate(color):
                rgb[..., c][mask] = val

    # Draw text labels if requested
    if (label_text_right or label_text_left) and seg_data is not None:
        seg_slice = np.take(seg_data, idx, axis=axis)
        h, w = seg_slice.shape
        pil = PILImage.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        for side, label_dict in [
            ('right', label_text_right or {}),
            ('left', label_text_left or {}),
        ]:
            for lbl, text in label_dict.items():
                coords = np.argwhere(seg_slice == lbl)
                if len(coords) == 0:
                    continue
                cy, cx = coords.mean(axis=0)
                tx = int(w * 0.85) if side == 'right' else int(w * 0.05)
                ty = int(cy)
                # Black outline for readability
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    draw.text((tx + dx, ty + dy), text, fill=(0, 0, 0))
                draw.text((tx, ty), text, fill=(255, 255, 255))
        rgb = np.array(pil)

    return rgb


def _build_jpg_path(
    inp: Path,
    out_dir: Path,
    prefix: str,
    suffix: str,
    orient: str = 'sag',
    sliceloc: float = 0.5,
) -> Path:
    stem = inp.name
    for ext in ('.nii.gz', '.nii'):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    return out_dir / f'{prefix}{stem}{suffix}_{orient}_{sliceloc}.jpg'


def preview(
    input: Path | str,
    output: Path | str,
    seg: Path | str | None = None,
    orient: str = 'sag',
    sliceloc: float = 0.5,
    label_text_right: dict[int, str] | None = None,
    label_text_left: dict[int, str] | None = None,
    out_suffix: str = "_preview",
    out_prefix: str = "",
    overwrite: bool = False,
) -> Path:
    input_path = Path(input)
    output_path = Path(output)
    if not str(output_path).endswith('.jpg'):
        output_path = _build_jpg_path(input_path, output_path, out_prefix, out_suffix, orient, sliceloc)
    img = Image.load(input_path)
    seg_data = Image.load(seg).data if seg else None
    rgb = make_preview(
        img.data, seg_data,
        orient=orient, sliceloc=sliceloc,
        label_text_right=label_text_right,
        label_text_left=label_text_left,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(rgb).save(str(output_path))
    return output_path


def preview_dir(
    input_dir: Path | str,
    output_dir: Path | str,
    seg_dir: Path | str | None = None,
    orient: str = 'sag',
    sliceloc: float = 0.5,
    label_text_right: dict[int, str] | None = None,
    label_text_left: dict[int, str] | None = None,
    out_suffix: str = "_preview",
    out_prefix: str = "",
    overwrite: bool = False,
    n_jobs: int = 1,
) -> None:
    imgs = collect_niftis(input_dir)
    segs = collect_niftis(seg_dir) if seg_dir else [None] * len(imgs)
    out = Path(output_dir)
    triples = [
        (i, s, _build_jpg_path(i, out, out_prefix, out_suffix, orient, sliceloc))
        for i, s in zip(imgs, segs)
        if overwrite or not _build_jpg_path(i, out, out_prefix, out_suffix, orient, sliceloc).exists()
    ]
    img_paths = [t[0] for t in triples]
    seg_paths = [t[1] for t in triples]
    out_paths = [t[2] for t in triples]
    process_map(
        functools.partial(
            preview,
            orient=orient, sliceloc=sliceloc,
            label_text_right=label_text_right, label_text_left=label_text_left,
            overwrite=overwrite,
        ),
        img_paths, out_paths, seg_paths,
        max_workers=n_jobs,
        disable=False,
        desc='uroseg preview',
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate a JPG preview image (single slice) of NIfTI files.'
    )
    parser.add_argument('--img', '-i', required=True, help='Input image file or folder')
    parser.add_argument('--seg', '-s', default=None, help='Input seg file or folder (optional)')
    parser.add_argument('--out', '-o', required=True, help='Output folder')
    parser.add_argument('--out-suffix', default='_preview', help='Output filename suffix')
    parser.add_argument('--out-prefix', default='', help='Output filename prefix')
    parser.add_argument(
        '--orient', '-t',
        default='sag',
        choices=['sag', 'ax', 'cor'],
        help='Slice orientation: sag (sagittal, default), ax (axial), cor (coronal)',
    )
    parser.add_argument(
        '--sliceloc', '-l',
        type=float,
        default=0.5,
        help='Slice position as fraction 0.0–1.0 along the chosen axis (default: 0.5 = middle)',
    )
    parser.add_argument(
        '--label-text-right', '-ltr',
        nargs='+',
        default=[],
        metavar='LABEL:TEXT',
        dest='label_text_right',
        help='Text labels on the right: JSON file or space-separated label:text pairs '
             '(e.g. --label-text-right 1:L1 2:L2)',
    )
    parser.add_argument(
        '--label-text-left', '-ltl',
        nargs='+',
        default=[],
        metavar='LABEL:TEXT',
        dest='label_text_left',
        help='Text labels on the left: JSON file or space-separated label:text pairs',
    )
    add_common_args(parser)
    args = parser.parse_args()

    imgs = collect_niftis(args.img)
    segs = collect_niftis(args.seg) if args.seg else [None] * len(imgs)
    if args.seg and len(segs) != len(imgs):
        print(
            f'Mismatch: {len(imgs)} images vs {len(segs)} segs.',
            file=sys.stderr,
        )
        sys.exit(1)
    out_dir = Path(args.out)

    ltr = _parse_label_text(args.label_text_right)
    ltl = _parse_label_text(args.label_text_left)

    def _out(i: Path) -> Path:
        return _build_jpg_path(
            i, out_dir, args.out_prefix, args.out_suffix,
            orient=args.orient, sliceloc=args.sliceloc,
        )

    pairs = [
        (i, s, _out(i))
        for i, s in zip(imgs, segs)
        if args.overwrite or not _out(i).exists()
    ]
    img_paths = [p[0] for p in pairs]
    out_paths = [p[2] for p in pairs]
    seg_paths = [p[1] for p in pairs]
    process_map(
        functools.partial(
            preview,
            orient=args.orient, sliceloc=args.sliceloc,
            label_text_right=ltr or None, label_text_left=ltl or None,
            overwrite=args.overwrite,
        ),
        img_paths, out_paths, seg_paths,
        max_workers=args.max_workers,
        disable=args.quiet,
        desc='uroseg preview',
    )
