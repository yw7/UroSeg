from __future__ import annotations
import argparse
import sys
from pathlib import Path

from uroseg.utils.utils import resolve_data_path, load_model_module, get_all_models, data_dir_help
from uroseg.models.base import _download_zip, _extract_zip


def extract_release_id(url: str) -> str:
    return url.rstrip('/').split('/')[-2]


def get_install_dir(nnunet_task: str, url: str, data_path: Path) -> Path:
    release_id = extract_release_id(url)
    return data_path / 'nnUNet' / 'results' / release_id / nnunet_task


def is_installed(nnunet_task: str, url: str, data_path: Path) -> bool:
    return get_install_dir(nnunet_task, url, data_path).exists()


def download_and_extract(model, nnunet_task: str, data_path: Path) -> None:
    url = model.weights_url
    if not url:
        print(f"  {model.name}: no weights_url — skipping.")
        return
    if is_installed(nnunet_task, url, data_path):
        print(f"  {model.name}: already installed.")
        return
    release_id = extract_release_id(url)
    results_dir = data_path / 'nnUNet' / 'results' / release_id
    print(f"  Downloading {model.name}...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path as _Path
        zip_path = _download_zip(url, _Path(tmp))
        _extract_zip(zip_path, results_dir)
    print(f"  Done. Weights installed at {results_dir / nnunet_task}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Download and install UroSeg model weights.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', nargs='+', metavar='MODEL',
                       help='One or more organ model names (e.g. prostate bladder)')
    group.add_argument('--all', action='store_true', help='Install all available models')
    parser.add_argument('--data-dir', default=None, help=data_dir_help())
    args = parser.parse_args()

    data_path = resolve_data_path(args.data_dir)

    if args.all:
        names = list(get_all_models().keys())
    else:
        names = args.model

    print(f"Installing {len(names)} model(s) to {data_path}...")
    for name in names:
        mod = load_model_module(name)
        download_and_extract(mod.MODEL, mod.NNUNET_TASK, data_path)
