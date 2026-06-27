from __future__ import annotations
import tempfile
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

from uroseg.utils.image import Image, save_nifti_seg
from uroseg.tools.transform_seg2image import resample_seg_to_image
from uroseg.tools.largest_component import keep_largest_component


def _extract_release_id(url: str) -> str:
    return url.rstrip('/').split('/')[-2]


def _download_zip(url: str, tmp_dir: Path) -> Path:
    import sys
    zip_name = url.split('/')[-1]
    zip_path = tmp_dir / zip_name
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc=zip_name) as bar:
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                bar.total = total_size
            bar.update(block_size)
        try:
            urllib.request.urlretrieve(url, zip_path, reporthook=reporthook)
        except urllib.error.HTTPError as e:
            print(f"\nDownload failed: HTTP {e.code} {e.reason}\n  URL: {url}", file=sys.stderr)
            sys.exit(1)
        except urllib.error.URLError as e:
            print(f"\nDownload failed: {e.reason}\n  URL: {url}", file=sys.stderr)
            sys.exit(1)
        except OSError as e:
            print(f"\nDownload failed: {e}\n  URL: {url}", file=sys.stderr)
            sys.exit(1)
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

    def predict_image(self, img: Image, **kwargs) -> Image:
        """Process a single image in memory. Return seg in the model's working space."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement predict_image()")

    def predict(self, input: Path | str, output_dir: Path | str,
                iso: bool = False, **kwargs) -> Path:
        """Load → predict_image → (optional) transform back to original space → save."""
        input_path = Path(input)
        img_orig = Image.load(input_path)
        seg = self.predict_image(img_orig, **kwargs)
        if not iso:
            seg = resample_seg_to_image(seg, img_orig)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / input_path.name
        save_nifti_seg(seg.data, seg.affine, seg.header, str(out_path))
        return out_path

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

    def predict_image(self, img: Image,
                      fold: int = 0, device: str = 'cuda', **kwargs) -> Image:
        """Reorient → 1 mm iso → nnunet → largest_component. Returns seg in 1 mm canonical space."""
        from uroseg.nnunet.helpers import run_predict_array
        img_canon = img.as_canonical()
        img_1mm = img_canon.resample((1.0, 1.0, 1.0))
        model_dir = _find_model_dir(self.name, _resolve_data_dir())
        seg = run_predict_array(model_dir, img_1mm, fold=fold, device=device)
        seg.data = keep_largest_component(seg.data)
        return seg
