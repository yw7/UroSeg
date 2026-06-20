from __future__ import annotations
import argparse
import functools
from pathlib import Path

import numpy as np
from scipy import ndimage
from tqdm.contrib.concurrent import process_map

from uroseg.utils.image import Image
from uroseg.utils.utils import add_common_args, build_pairs


def keep_largest_component(data: np.ndarray, labels: list[int] | None = None) -> np.ndarray:
    result = np.zeros_like(data)
    label_ids = labels if labels else [int(v) for v in np.unique(data) if v > 0]
    for label_id in label_ids:
        mask = data == label_id
        labeled_arr, n = ndimage.label(mask)
        if n == 0:
            continue
        sizes = ndimage.sum(mask, labeled_arr, range(1, n + 1))
        largest = int(np.argmax(sizes)) + 1
        result[labeled_arr == largest] = label_id
    return result


def process_one(pair: tuple[Path, Path], args: argparse.Namespace) -> None:
    input_path, output_path = pair
    img = Image.load(input_path)
    labels = args.labels if args.labels else None
    img.data = keep_largest_component(img.data, labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Keep only the largest connected component per label in a segmentation.'
    )
    parser.add_argument('--seg', required=True, help='Input seg file or folder')
    parser.add_argument('--out', required=True, help='Output file or folder')
    parser.add_argument('--labels', nargs='+', type=int, default=None,
                        help='Label IDs to process (default: all non-zero)')
    parser.add_argument('--out-suffix', default='_largest', help='Output filename suffix')
    parser.add_argument('--out-prefix', default='', help='Output filename prefix')
    add_common_args(parser)
    args = parser.parse_args()

    pairs = build_pairs(args.seg, args.out, args.out_suffix, args.out_prefix, args.overwrite)
    process_map(
        functools.partial(process_one, args=args),
        pairs,
        max_workers=args.max_workers,
        disable=args.quiet,
        desc='uroseg largest_component',
    )
