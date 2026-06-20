from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from importlib.resources import files


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing outputs (default: skip if exists)')
    parser.add_argument('--max-workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress bar and non-error output')


def collect_niftis(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(
        p for p in path.iterdir()
        if p.name.endswith('.nii.gz') or p.name.endswith('.nii')
    )


def build_output_path(inp: Path, out_dir: Path, prefix: str, suffix: str) -> Path:
    out_dir = Path(out_dir)
    stem = inp.name
    for ext in ('.nii.gz', '.nii'):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    return out_dir / f'{prefix}{stem}{suffix}.nii.gz'


def build_pairs(
    inp: str | Path,
    out: str | Path,
    suffix: str,
    prefix: str,
    overwrite: bool,
) -> list[tuple[Path, Path]]:
    inputs = collect_niftis(inp)
    out = Path(out)
    pairs = [(i, build_output_path(i, out, prefix, suffix)) for i in inputs]
    if not overwrite:
        pairs = [(i, o) for i, o in pairs if not o.exists()]
    return pairs


def resolve_data_path(data_dir: str | None = None) -> Path:
    if data_dir:
        return Path(data_dir)
    if 'UROSEG_DATA' in os.environ:
        return Path(os.environ['UROSEG_DATA'])
    return Path.home() / 'uroseg'


def get_model(name: str) -> dict:
    path = files('uroseg.resources.models').joinpath(f'{name}.json')
    return json.loads(path.read_text())


def get_all_models() -> dict[str, dict]:
    models_dir = files('uroseg.resources.models')
    return {
        p.stem: json.loads(p.read_text())
        for p in models_dir.iterdir()
        if p.name.endswith('.json')
    }
