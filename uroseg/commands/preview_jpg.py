from __future__ import annotations
import argparse
import functools
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
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


def make_preview(img_data: np.ndarray, seg_data: np.ndarray | None = None) -> np.ndarray:
    slices = []
    for axis in range(3):
        idx = img_data.shape[axis] // 2
        sl = np.take(img_data, idx, axis=axis).astype(float)
        sl -= sl.min()
        if sl.max() > 0:
            sl /= sl.max()
        gray = (sl * 255).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)

        if seg_data is not None:
            seg_sl = np.take(seg_data, idx, axis=axis)
            for label_id in np.unique(seg_sl):
                if label_id == 0:
                    continue
                color = LABEL_COLORS[int(label_id - 1) % len(LABEL_COLORS)]
                mask = seg_sl == label_id
                for c, val in enumerate(color):
                    rgb[..., c][mask] = val

        slices.append(rgb)

    max_h = max(s.shape[0] for s in slices)
    padded = []
    for s in slices:
        pad = max_h - s.shape[0]
        s = np.pad(s, ((pad // 2, pad - pad // 2), (0, 0), (0, 0)))
        padded.append(s)

    return np.concatenate(padded, axis=1)


def _build_jpg_path(inp: Path, out_dir: Path, prefix: str, suffix: str) -> Path:
    stem = inp.name
    for ext in ('.nii.gz', '.nii'):
        if stem.endswith(ext):
            stem = stem[:-len(ext)]
            break
    return out_dir / f'{prefix}{stem}{suffix}.jpg'


def process_one(
    pair: tuple[Path, Path | None, Path],
    args: argparse.Namespace,
) -> None:
    img_path, seg_path, out_path = pair
    img = Image.load(img_path)
    seg_data = Image.load(seg_path).data if seg_path else None
    preview = make_preview(img.data, seg_data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(preview).save(str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate JPG preview images (3 orthogonal slices) of NIfTI files.'
    )
    parser.add_argument('--img', required=True, help='Input image file or folder')
    parser.add_argument('--seg', default=None, help='Input seg file or folder (optional)')
    parser.add_argument('--out', required=True, help='Output folder')
    parser.add_argument('--out-suffix', default='_preview', help='Output filename suffix')
    parser.add_argument('--out-prefix', default='', help='Output filename prefix')
    add_common_args(parser)
    args = parser.parse_args()

    imgs = collect_niftis(args.img)
    segs = collect_niftis(args.seg) if args.seg else [None] * len(imgs)
    if args.seg and len(segs) != len(imgs):
        import sys
        print(
            f"Mismatch: {len(imgs)} images vs {len(segs)} segs.",
            file=sys.stderr,
        )
        sys.exit(1)
    out_dir = Path(args.out)

    pairs = [
        (i, s, _build_jpg_path(i, out_dir, args.out_prefix, args.out_suffix))
        for i, s in zip(imgs, segs)
        if args.overwrite
        or not _build_jpg_path(i, out_dir, args.out_prefix, args.out_suffix).exists()
    ]

    process_map(
        functools.partial(process_one, args=args),
        pairs,
        max_workers=args.max_workers,
        disable=args.quiet,
        desc='uroseg preview',
    )
