from __future__ import annotations
import argparse
import functools
import json
from pathlib import Path

import numpy as np
from tqdm.contrib.concurrent import process_map

from uroseg.utils.image import Image, save_nifti_seg
from uroseg.utils.utils import add_common_args, build_pairs


def apply_map(data: np.ndarray, mapping: dict) -> np.ndarray:
    result = np.zeros_like(data)
    for src, dst in mapping.items():
        result[data == int(src)] = int(dst)
    return result


def process_one(pair: tuple[Path, Path], mapping: dict, args: argparse.Namespace) -> None:
    input_path, output_path = pair
    img = Image.load(input_path)
    img.data = apply_map(img.data, mapping)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_nifti_seg(img.data, img.affine, img.header, str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description='Remap label IDs in segmentation files.')
    parser.add_argument('--seg', required=True, help='Input seg file or folder')
    parser.add_argument('--out', required=True, help='Output file or folder')
    parser.add_argument('--map', required=True, help='Path to label map JSON {"src": dst}')
    parser.add_argument('--out-suffix', default='_mapped', help='Output filename suffix')
    parser.add_argument('--out-prefix', default='', help='Output filename prefix')
    add_common_args(parser)
    args = parser.parse_args()

    pairs = build_pairs(args.seg, args.out, args.out_suffix, args.out_prefix, args.overwrite)
    with open(args.map) as f:
        mapping = json.load(f)
    process_map(
        functools.partial(process_one, mapping=mapping, args=args),
        pairs,
        max_workers=args.max_workers,
        disable=args.quiet,
        desc='uroseg map',
    )
