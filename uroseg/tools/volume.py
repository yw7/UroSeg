from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

from uroseg.utils.image import Image
from uroseg.utils.utils import add_common_args, collect_niftis


def volume(img: Image, labels: dict) -> dict[str, float]:
    """Compute volume in mm³ for each named label from an in-memory Image.

    Skips background (value 0). Supports multi-label entries:
    ``{"prostate": [1, 2, 3]}`` counts voxels where value is any of 1, 2, 3.
    """
    zooms = img.header.get_zooms()[:3]
    voxel_vol = float(zooms[0]) * float(zooms[1]) * float(zooms[2])

    result: dict[str, float] = {}
    for name, value in labels.items():
        values = value if isinstance(value, list) else [value]
        if all(int(v) == 0 for v in values):
            continue
        mask = np.isin(img.data, [int(v) for v in values])
        result[name] = int(np.sum(mask)) * voxel_vol

    return result


def volume_file(seg: Path | str, labels: dict) -> dict[str, float]:
    """Compute volume in mm³ for each named label in a segmentation file."""
    return volume(Image.load(Path(seg)), labels)


def volume_dir(
    input_dir: Path | str,
    output_csv: Path | str,
    labels: dict,
    overwrite: bool = False,
    quiet: bool = False,
) -> None:
    """Compute volumes for all NIfTIs in input_dir and save a CSV.

    CSV rows are files; columns are label names with volumes in mm³.
    """
    output_csv = Path(output_csv)
    if output_csv.exists() and not overwrite:
        return

    inputs = collect_niftis(Path(input_dir))
    rows: list[dict] = []
    for inp in tqdm(inputs, desc='uroseg volume', disable=quiet):
        vols = volume_file(inp, labels)
        rows.append({'filename': inp.name.removesuffix('.gz').removesuffix('.nii'), **vols})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_csv)


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text('filename\n')
        return
    fieldnames = list(rows[0].keys())
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compute label volumes (mm³) for segmentation file(s) and save to CSV.'
    )
    parser.add_argument('--seg', '-s', required=True, help='Input seg file or folder')
    parser.add_argument('--out', '-o', default=None,
                        help='Output CSV path (required for folder; prints to stdout for single file)')
    label_group = parser.add_mutually_exclusive_group(required=True)
    label_group.add_argument('--model', '-m',
                              help='Model name to use its label definitions (e.g. prostate)')
    label_group.add_argument('--labels', '-l',
                              help='Labels as JSON string, e.g. \'{"bladder": 1}\'')
    add_common_args(parser)
    args = parser.parse_args()

    if args.model:
        from uroseg.models import get_model
        labels = get_model(args.model).labels
    else:
        import json
        labels = json.loads(args.labels)

    seg_path = Path(args.seg)
    if seg_path.is_dir():
        if not args.out:
            import sys
            print("--out is required when --seg is a directory", file=sys.stderr)
            sys.exit(1)
        volume_dir(seg_path, args.out, labels, overwrite=args.overwrite, quiet=args.quiet)
    else:
        vols = volume_file(seg_path, labels)
        if args.out:
            _write_csv([{'filename': seg_path.name.removesuffix('.gz').removesuffix('.nii'), **vols}], Path(args.out))
        else:
            for name, vol_mm3 in vols.items():
                print(f"{name}: {vol_mm3:.1f} mm³")
