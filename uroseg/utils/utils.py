from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from importlib.resources import files


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--overwrite', '-r', action='store_true',
                        help='Overwrite existing outputs (default: skip if exists)')
    parser.add_argument('--max-workers', '-w', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument('--quiet', '-q', action='store_true',
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
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, KeyError, OSError):
        available = ', '.join(sorted(
            p.name.removesuffix('.json')
            for p in files('uroseg.resources.models').iterdir()
            if p.name.endswith('.json')
        ))
        raise ValueError(
            f"Unknown model '{name}'. Available: {available}\n"
            f"Run 'uroseg list' to see all models."
        )


def normalize_labels(raw: dict) -> dict:
    """Normalize model JSON label dicts to ``{name: int | list[int]}``.

    Accepts four input formats:
    - TotalSpineSeg style (name→value): ``{"background": 0, "disc": [1,2,3]}``
    - Old integer-key style: ``{"0": "background", "1": "prostate"}``
    - Comma-key style: ``{"1,2,3": "prostate", "2": "pz"}``
    - Range-key style: ``{"1-3": "prostate", "2": "pz"}``
    """
    result: dict = {}
    for k, v in raw.items():
        if isinstance(v, (int, list)):
            # TotalSpineSeg format — key is the label name
            result[str(k)] = v
        elif isinstance(v, str):
            # Old/compact format — key encodes label value(s), value is the name
            name = v
            k_str = str(k)
            if ',' in k_str:
                values = [int(x.strip()) for x in k_str.split(',')]
                result[name] = values if len(values) > 1 else values[0]
            elif '-' in k_str and not k_str.startswith('-'):
                start_s, end_s = k_str.split('-', 1)
                values = list(range(int(start_s), int(end_s) + 1))
                result[name] = values if len(values) > 1 else values[0]
            else:
                result[name] = int(k_str)
    return result


def get_all_models() -> dict[str, dict]:
    models_dir = files('uroseg.resources.models')
    return {
        p.name.removesuffix('.json'): json.loads(p.read_text())
        for p in models_dir.iterdir()
        if p.name.endswith('.json')
    }
