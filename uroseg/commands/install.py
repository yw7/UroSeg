from __future__ import annotations
import argparse
import sys
import zipfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

from uroseg.utils.utils import resolve_data_path, get_model, get_all_models


def extract_release_id(url: str) -> str:
    return url.rstrip('/').split('/')[-2]


def get_install_dir(nnunet_task: str, url: str, data_path: Path) -> Path:
    release_id = extract_release_id(url)
    return data_path / 'nnUNet' / 'results' / release_id / nnunet_task


def is_installed(nnunet_task: str, url: str, data_path: Path) -> bool:
    return get_install_dir(nnunet_task, url, data_path).exists()


def download_and_extract(model: dict, data_path: Path, store_export: bool = False) -> None:
    url = model.get('weights_url')
    if not url:
        print(f"  {model['name']}: no weights_url — skipping.")
        return

    nnunet_task = model['nnunet_task']

    if is_installed(nnunet_task, url, data_path):
        print(f"  {model['name']}: already installed.")
        return

    release_id = extract_release_id(url)
    exports_dir = data_path / 'nnUNet' / 'exports'
    results_dir = data_path / 'nnUNet' / 'results' / release_id
    exports_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    zip_name = url.split('/')[-1]
    zip_path = exports_dir / zip_name

    print(f"  Downloading {model['name']} ({zip_name})...")
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc=zip_name) as bar:
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                bar.total = total_size
            bar.update(block_size)
        urllib.request.urlretrieve(url, zip_path, reporthook=reporthook)

    print(f"  Extracting to {results_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(results_dir)

    if not store_export:
        zip_path.unlink()
        print(f"  Done. Weights installed at {results_dir / nnunet_task}")
    else:
        print(f"  Done. Archive kept at {zip_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Download and install UroSeg model weights.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', nargs='+', metavar='MODEL',
                       help='One or more organ model names (e.g. prostate bladder)')
    group.add_argument('--all', action='store_true', help='Install all available models')
    parser.add_argument('--data-dir', default=None, help='Override data path')
    parser.add_argument('--store-export', action='store_true',
                        help='Keep downloaded zip archive after extraction')
    args = parser.parse_args()

    data_path = resolve_data_path(args.data_dir)

    if args.all:
        models = list(get_all_models().values())
    else:
        models = [get_model(name) for name in args.model]

    print(f"Installing {len(models)} model(s) to {data_path}...")
    for model in models:
        download_and_extract(model, data_path, store_export=args.store_export)
