from __future__ import annotations
import tempfile
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm


def _extract_release_id(url: str) -> str:
    return url.rstrip('/').split('/')[-2]


def _download_zip(url: str, tmp_dir: Path) -> Path:
    zip_name = url.split('/')[-1]
    zip_path = tmp_dir / zip_name
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc=zip_name) as bar:
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                bar.total = total_size
            bar.update(block_size)
        urllib.request.urlretrieve(url, zip_path, reporthook=reporthook)
    return zip_path


def _extract_zip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest)
    zip_path.unlink()


def _find_model_dir(name: str, data_dir: Path) -> Path:
    model_root = data_dir / name
    if not model_root.exists():
        raise FileNotFoundError(
            f"Model '{name}' not found under {data_dir}.\n"
            f"Run: uroseg install --model {name}"
        )
    releases = sorted(
        (d for d in model_root.iterdir() if d.is_dir()),
        reverse=True,
    )
    if not releases:
        raise FileNotFoundError(f"No releases found for model '{name}' under {model_root}.")
    return releases[0]


def _resolve_data_dir() -> Path:
    from uroseg.utils.utils import resolve_data_path
    return resolve_data_path()


class SegModel:
    name: str
    description: str
    weights_url: str
    labels: dict

    def install(self, data_dir: Path) -> None:
        if not self.weights_url:
            print(f"  {self.name}: no weights_url — skipping.")
            return
        release_id = _extract_release_id(self.weights_url)
        dest = data_dir / self.name / release_id
        if dest.exists():
            print(f"  {self.name}: already installed at {dest}")
            return
        print(f"  Downloading {self.name}...")
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = _download_zip(self.weights_url, Path(tmp))
            _extract_zip(zip_path, dest)
        print(f"  Done. Weights installed at {dest}")

    def predict(self, input: Path, output_dir: Path, **kwargs) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement predict()")

    def predict_dir(self, input_dir: Path, output_dir: Path,
                    n_jobs: int = 1, **kwargs) -> None:
        from uroseg.utils.utils import collect_niftis
        inputs = collect_niftis(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if n_jobs == 1:
            for inp in inputs:
                self.predict(inp, output_dir, **kwargs)
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                futures = [ex.submit(self.predict, inp, output_dir, **kwargs) for inp in inputs]
                for f in futures:
                    f.result()


class NNUNetSegModel(SegModel):
    nnunet_task: str

    def predict(self, input: Path, output_dir: Path,
                fold: int = 0, device: str = 'cuda', **kwargs) -> None:
        from uroseg.nnunet.helpers import run_predict
        model_dir = _find_model_dir(self.name, _resolve_data_dir())
        run_predict(model_dir, [Path(input)], Path(output_dir), fold=fold, device=device)
