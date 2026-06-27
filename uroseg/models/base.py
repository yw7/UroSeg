from __future__ import annotations
import argparse
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from uroseg.utils.image import Image, save_nifti_seg
from uroseg.tools.transform_seg2image import resample_seg_to_image
from uroseg.tools.largest_component import keep_largest_component


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


def add_inference_args(parser: argparse.ArgumentParser) -> None:
    """Add standard inference args to a parser (used by each model's CLI)."""
    from uroseg.utils.utils import add_common_args, data_dir_help
    parser.add_argument('img', help='Input image file or folder')
    parser.add_argument('out', nargs='?', default='.', help='Output folder (default: current directory)')
    parser.add_argument('--fold', '-f', type=int, default=0, help='nnU-Net fold (default: 0)')
    parser.add_argument('--device', '-d', default='cuda', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--iso', action='store_true', default=False,
                        help='Leave output in 1mm canonical space (default: resample back to input)')
    parser.add_argument('--out-suffix', default='_seg', help='Output filename suffix')
    parser.add_argument('--out-prefix', default='', help='Output filename prefix')
    parser.add_argument('--data-dir', default=None, help=data_dir_help())
    add_common_args(parser)


class SegModel:
    name: str
    description: str
    weights_url: str
    labels: dict
    post_largest_component: bool = False

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

    def init_predictor(self, model_dir: Path, fold: int = 0, device: str = 'cuda'):
        raise NotImplementedError(f"{self.__class__.__name__} does not implement init_predictor()")

    def predict_image(self, predictor, img: Image) -> Image:
        """Pure inference on img (canonical 1mm). Returns seg in model space."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement predict_image()")

    def predict(self, input: Path | str, output_dir: Path | str,
                fold: int = 0, device: str = 'cuda', iso: bool = False) -> Path:
        """Single-file Python API: load → canonical+1mm → infer → post-process → save."""
        input_path = Path(input)
        img_orig = Image.load(input_path)
        img_1mm = img_orig.as_canonical().resample((1.0, 1.0, 1.0))
        model_dir = _find_model_dir(self.name, _resolve_data_dir())
        predictor = self.init_predictor(model_dir, fold=fold, device=device)
        seg = self.predict_image(predictor, img_1mm)
        if self.post_largest_component:
            seg.data = keep_largest_component(seg.data, binarize=True)
        if not iso:
            seg = resample_seg_to_image(seg, img_orig)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / input_path.name
        save_nifti_seg(seg.data, seg.affine, seg.header, str(out_path))
        return out_path

    def _predict_batch(self, predictor, inputs: list, out_dir: Path,
                       iso: bool, dest_fn) -> None:
        """Inner loop shared by predict_dir and predict_cli."""
        for inp in inputs:
            img_orig = Image.load(inp)
            img_1mm = img_orig.as_canonical().resample((1.0, 1.0, 1.0))
            seg = self.predict_image(predictor, img_1mm)
            if self.post_largest_component:
                seg.data = keep_largest_component(seg.data, binarize=True)
            if not iso:
                seg = resample_seg_to_image(seg, img_orig)
            save_nifti_seg(seg.data, seg.affine, seg.header, str(dest_fn(inp)))

    def predict_dir(self, input_dir: Path, output_dir: Path,
                    fold: int = 0, device: str = 'cuda', iso: bool = False) -> None:
        """Batch Python API: predict all NIfTIs in a directory, predictor init once."""
        from uroseg.utils.utils import collect_niftis
        inputs = collect_niftis(Path(input_dir))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir = _find_model_dir(self.name, _resolve_data_dir())
        predictor = self.init_predictor(model_dir, fold=fold, device=device)
        self._predict_batch(predictor, inputs, output_dir, iso,
                            dest_fn=lambda inp: output_dir / inp.name)

    def predict_cli(self, args) -> None:
        """CLI batch prediction: auto-install, tqdm progress, predictor init once."""
        from uroseg.utils.utils import collect_niftis, build_output_path, resolve_data_path
        data_path = resolve_data_path(args.data_dir)
        try:
            model_dir = _find_model_dir(self.name, data_path)
        except FileNotFoundError:
            print(f"Model '{self.name}' not installed — downloading...")
            self.install(data_path)
            model_dir = _find_model_dir(self.name, data_path)
        inputs = collect_niftis(args.img)
        if not inputs:
            print(f"No NIfTI files found in {args.img}", file=sys.stderr)
            sys.exit(1)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        predictor = self.init_predictor(model_dir, fold=args.fold, device=args.device)
        filtered = [inp for inp in inputs
                    if args.overwrite or not build_output_path(
                        inp, out_dir, args.out_prefix, args.out_suffix).exists()]
        self._predict_batch(
            predictor,
            tqdm(filtered, desc=f'uroseg {self.name}', disable=args.quiet),
            out_dir, args.iso,
            dest_fn=lambda inp: build_output_path(inp, out_dir, args.out_prefix, args.out_suffix),
        )
        if not args.quiet:
            print(f"Segmentations saved to {out_dir}")

    @classmethod
    def cli_main(cls) -> None:
        """Entry point for each model's CLI. Call as `MyModel.cli_main()` from main()."""
        parser = argparse.ArgumentParser(prog=f'uroseg {cls.name}', description=cls.description)
        add_inference_args(parser)
        args = parser.parse_args()
        cls().predict_cli(args)


class NNUNetSegModel(SegModel):
    nnunet_task: str

    def init_predictor(self, model_dir: Path, fold: int = 0, device: str = 'cuda'):
        from uroseg.nnunet.helpers import _init_predictor
        return _init_predictor(model_dir, fold=fold, device=device)

    def predict_image(self, predictor, img: Image) -> Image:
        """Pure inference on img (canonical 1mm). Returns seg in 1mm canonical space."""
        from uroseg.nnunet.helpers import _run_inference
        seg_array = _run_inference(predictor, img)
        return Image(data=seg_array, affine=img.affine, header=img.header)
