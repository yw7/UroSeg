from __future__ import annotations
import argparse
import functools
import json
from pathlib import Path

import numpy as np
from tqdm.contrib.concurrent import process_map

from uroseg.utils.image import Image, save_nifti_seg
from uroseg.utils.utils import add_common_args, build_pairs, collect_niftis


def apply_map(data: np.ndarray, mapping: dict, keep_unmapped: bool = False) -> np.ndarray:
    if keep_unmapped:
        result = data.copy()
        for src, dst in mapping.items():
            result[data == int(src)] = int(dst)
    else:
        result = np.zeros_like(data)
        for src, dst in mapping.items():
            result[data == int(src)] = int(dst)
    return result


def _resolve_companion(companion_path: str | None, input_path: Path) -> Path | None:
    """Given a companion path (file or folder), return the specific file that
    corresponds to *input_path* (matched by filename when companion is a folder)."""
    if companion_path is None:
        return None
    p = Path(companion_path)
    if p.is_dir():
        # Match by stem (filename without extension)
        candidates = collect_niftis(p)
        for c in candidates:
            if c.name == input_path.name:
                return c
        return None
    return p


def process_one(
    pair: tuple[Path, Path],
    mapping: dict,
    keep_unmapped: bool,
    update_seg: str | None,
    update_from_seg: str | None,
    args: argparse.Namespace,
) -> None:
    input_path, output_path = pair
    img = Image.load(input_path)
    out_data = apply_map(img.data, mapping, keep_unmapped=keep_unmapped)

    # --update-seg: fill zeros in output with non-zero values from companion seg
    companion = _resolve_companion(update_seg, input_path)
    if companion is not None:
        seg_img = Image.load(companion)
        mask = out_data == 0
        out_data[mask] = seg_img.data[mask]

    # --update-from-seg: overwrite output where companion seg is non-zero
    companion_from = _resolve_companion(update_from_seg, input_path)
    if companion_from is not None:
        from_img = Image.load(companion_from)
        mask = from_img.data != 0
        out_data[mask] = from_img.data[mask]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_nifti_seg(out_data, img.affine, img.header, str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description='Remap label IDs in segmentation files.')
    parser.add_argument('--seg', '-s', required=True, help='Input seg file or folder')
    parser.add_argument('--out', '-o', required=True, help='Output file or folder')
    parser.add_argument(
        '--map', '-m', nargs='+', default=[],
        help='Label mapping: path to .json file OR space-separated key:value pairs '
             '(e.g. --map 1:2 3:0)',
    )
    parser.add_argument('--out-suffix', default='_mapped', help='Output filename suffix')
    parser.add_argument('--out-prefix', default='', help='Output filename prefix')
    parser.add_argument(
        '--keep-unmapped', action='store_true',
        help='Keep labels not present in the map unchanged (default: zero them out)',
    )
    parser.add_argument(
        '--update-seg', default=None, metavar='PATH',
        help='Seg file or folder: non-zero values in the output fill zeros in this seg '
             '(output takes priority)',
    )
    parser.add_argument(
        '--update-from-seg', default=None, metavar='PATH',
        help='Seg file or folder: non-zero values from this seg overwrite the output '
             'where it is zero',
    )
    add_common_args(parser)
    args = parser.parse_args()

    # Parse --map argument: JSON file or direct key:value pairs
    map_list = args.map
    if not map_list:
        mapping: dict[int, int] = {}
    elif len(map_list) == 1 and map_list[0].endswith('.json'):
        with open(map_list[0], 'r') as f:
            mapping = {int(k): int(v) for k, v in json.load(f).items()}
    else:
        mapping = {
            int(pair.split(':')[0]): int(pair.split(':')[1])
            for pair in map_list
        }

    pairs = build_pairs(args.seg, args.out, args.out_suffix, args.out_prefix, args.overwrite)
    process_map(
        functools.partial(
            process_one,
            mapping=mapping,
            keep_unmapped=args.keep_unmapped,
            update_seg=args.update_seg,
            update_from_seg=args.update_from_seg,
            args=args,
        ),
        pairs,
        max_workers=args.max_workers,
        disable=args.quiet,
        desc='uroseg map',
    )


def map_labels(
    input: Path | str,
    output: Path | str,
    map: dict,
    keep_unmapped: bool = False,
    update_seg: Path | str | None = None,
    update_from_seg: Path | str | None = None,
    out_suffix: str = "_mapped",
    out_prefix: str = "",
    overwrite: bool = False,
) -> Path:
    import argparse
    input_path, output_path = Path(input), Path(output)
    if not output_path.suffix:
        from uroseg.utils.utils import build_output_path
        output_path = build_output_path(input_path, output_path, out_prefix, out_suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    int_map = {int(k): int(v) for k, v in map.items()}
    process_one(
        (input_path, output_path),
        int_map,
        keep_unmapped,
        str(update_seg) if update_seg else None,
        str(update_from_seg) if update_from_seg else None,
        argparse.Namespace(quiet=True, overwrite=overwrite),
    )
    return output_path


def map_labels_dir(
    input_dir: Path | str,
    output_dir: Path | str,
    map: dict,
    keep_unmapped: bool = False,
    update_seg: Path | str | None = None,
    update_from_seg: Path | str | None = None,
    out_suffix: str = "_mapped",
    out_prefix: str = "",
    overwrite: bool = False,
    n_jobs: int = 1,
) -> None:
    import argparse, functools
    from uroseg.utils.utils import build_pairs
    int_map = {int(k): int(v) for k, v in map.items()}
    pairs = build_pairs(input_dir, output_dir, out_suffix, out_prefix, overwrite)
    process_map(
        functools.partial(
            process_one,
            mapping=int_map,
            keep_unmapped=keep_unmapped,
            update_seg=str(update_seg) if update_seg else None,
            update_from_seg=str(update_from_seg) if update_from_seg else None,
            args=argparse.Namespace(quiet=False, overwrite=overwrite),
        ),
        pairs,
        max_workers=n_jobs,
        disable=False,
        desc='uroseg map',
    )
