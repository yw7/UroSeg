from __future__ import annotations
import argparse
import functools
from pathlib import Path

from tqdm.contrib.concurrent import process_map

from uroseg.utils.image import Image
from uroseg.utils.utils import add_common_args, build_pairs


def process_one(pair: tuple[Path, Path], args: argparse.Namespace) -> None:
    input_path, output_path = pair
    img = Image.load(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Copy NIfTI files from one folder to another with optional renaming.'
    )
    parser.add_argument('--img', '-i', required=True, help='Input file or folder')
    parser.add_argument('--out', '-o', required=True, help='Output folder')
    parser.add_argument('--out-suffix', default='', help='Output filename suffix (default: none)')
    parser.add_argument('--out-prefix', default='', help='Output filename prefix (default: none)')
    add_common_args(parser)
    args = parser.parse_args()

    pairs = build_pairs(args.img, args.out, args.out_suffix, args.out_prefix, args.overwrite)
    process_map(
        functools.partial(process_one, args=args),
        pairs,
        max_workers=args.max_workers,
        disable=args.quiet,
        desc='uroseg cpdir',
    )


def cpdir(
    input: Path | str,
    output: Path | str,
    out_suffix: str = "",
    out_prefix: str = "",
    overwrite: bool = False,
) -> Path:
    import argparse
    input_path, output_path = Path(input), Path(output)
    if not output_path.suffix:
        from uroseg.utils.utils import build_output_path
        output_path = build_output_path(input_path, output_path, out_prefix, out_suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    process_one(
        (input_path, output_path),
        argparse.Namespace(overwrite=overwrite),
    )
    return output_path


def cpdir_dir(
    input_dir: Path | str,
    output_dir: Path | str,
    out_suffix: str = "",
    out_prefix: str = "",
    overwrite: bool = False,
    n_jobs: int = 1,
) -> None:
    import argparse, functools
    from uroseg.utils.utils import build_pairs
    pairs = build_pairs(input_dir, output_dir, out_suffix, out_prefix, overwrite)
    process_map(
        functools.partial(process_one, args=argparse.Namespace(overwrite=overwrite)),
        pairs,
        max_workers=n_jobs,
        disable=False,
        desc='uroseg cpdir',
    )
