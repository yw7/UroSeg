# UroSeg Public-Ready Redesign

## Goal

Reorganize UroSeg into a logically structured, contribution-ready package with independent components, proper model class hierarchy, clean public API, and separated training/model-data directories.

---

## 1. Package Structure

```
uroseg/
├── __init__.py           # flat public API (all exports)
├── __main__.py           # unchanged
├── cli.py                # CLI dispatcher (interface unchanged)
├── models/
│   ├── __init__.py       # get_model(name), list_models()
│   ├── base.py           # SegModel, NNUNetSegModel
│   ├── prostate.py       # class Prostate(NNUNetSegModel)
│   └── bladder.py        # class Bladder(NNUNetSegModel)
├── tools/                # renamed from commands/
│   ├── __init__.py
│   ├── map_labels.py
│   ├── resample.py
│   ├── preview.py
│   ├── crop.py
│   ├── largest_component.py
│   ├── reorient.py
│   ├── cpdir.py
│   └── transform_seg2image.py
├── nnunet/
│   ├── __init__.py
│   ├── helpers.py        # nnunet-specific: setup_env, extract_dataset_id, generate_dataset_json, run_predict
│   ├── train.py          # training pipeline (was train_nnunet.py)
│   └── predict.py        # nnunet CLI prediction (was predict_nnunet.py)
├── resources/
│   ├── __init__.py
│   └── auglab/
│       ├── __init__.py
│       └── transform_params_gpu.json
└── utils/
    ├── __init__.py
    ├── utils.py           # resolve_data_path, collect_niftis, normalize_labels, data_dir_help
    └── image.py           # image processing utilities
```

### Files removed / merged

| Old path | Destination |
|---|---|
| `uroseg/models.py` | `uroseg/models/base.py` |
| `uroseg/resources/models/prostate.py` | `uroseg/models/prostate.py` |
| `uroseg/resources/models/bladder.py` | `uroseg/models/bladder.py` |
| `uroseg/utils/inference_utils.py` | `uroseg/nnunet/helpers.py` + `uroseg/nnunet/predict.py` |
| `uroseg/commands/train.py` | `uroseg/nnunet/train.py` |
| `uroseg/commands/train_nnunet.py` | `uroseg/nnunet/train.py` |
| `uroseg/commands/predict_nnunet.py` | `uroseg/nnunet/predict.py` |
| `uroseg/commands/install.py` | `SegModel.install()` in `uroseg/models/base.py`; CLI wired inline in `cli.py` (3 lines, no separate file) |
| `uroseg/commands/<tool>.py` | `uroseg/tools/<tool>.py` (same filename) |

---

## 2. Model Class Architecture

### `uroseg/models/base.py`

```python
class SegModel:
    name: str           # "prostate"
    description: str    # human-readable
    weights_url: str    # GitHub release zip URL
    labels: dict        # {"background": 0, "prostate": [1,2,3], "prostate_tz": 2}

    def install(self, data_dir: Path) -> None:
        """Download zip and extract to data_dir/<name>/<release_id>/."""
        ...

    def predict(self, input: Path, output_dir: Path, **kwargs) -> None:
        raise NotImplementedError

    def predict_dir(self, input_dir: Path, output_dir: Path,
                    n_jobs: int = 1, **kwargs) -> None:
        """Multiprocessing wrapper over predict() using ProcessPoolExecutor."""
        ...
```

Private helpers in `base.py` (not nnunet-specific, reusable by any backend):
- `_extract_release_id(url: str) -> str` — derives release tag from URL second-to-last segment
- `_download_and_extract(url: str, dest: Path) -> None` — downloads zip, extracts in place
- `_find_model_dir(name: str, data_dir: Path) -> Path` — searches `data_dir/<name>/` newest-release-first

### `uroseg/models/base.py` — NNUNetSegModel

```python
class NNUNetSegModel(SegModel):
    nnunet_task: str    # "Dataset101_Prostate"

    def predict(self, input: Path, output_dir: Path,
                fold: int = 0, device: str = "cuda", **kwargs) -> None:
        from uroseg.nnunet.helpers import run_predict
        model_dir = _find_model_dir(self.name, _resolve_data_dir())
        run_predict(model_dir, [input], output_dir, fold=fold, device=device)

    # predict_dir() inherited from SegModel
```

### `uroseg/models/prostate.py`

```python
from uroseg.models.base import NNUNetSegModel

class Prostate(NNUNetSegModel):
    name = "prostate"
    description = "Prostate: whole (1), transition zone (2), peripheral zone (3)"
    weights_url = "https://github.com/yw7/uroseg/releases/download/r20260101/Dataset101_Prostate_r20260101.zip"
    labels = {"background": 0, "prostate": [1, 2, 3], "prostate_tz": 2, "prostate_pz": 3}
    nnunet_task = "Dataset101_Prostate"

def main():
    """CLI entry point: uroseg prostate --img ... --out ..."""
    ...
```

### `uroseg/models/__init__.py`

```python
from uroseg.models.prostate import Prostate
from uroseg.models.bladder import Bladder

_REGISTRY = {cls.name: cls for cls in [Prostate, Bladder]}

def get_model(name: str) -> SegModel:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model: {name!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]()

def list_models() -> list[str]:
    return list(_REGISTRY)
```

### Data layout

Weights are installed to `data_dir/<model_name>/<release_id>/` (e.g. `~/.uroseg/prostate/r20260101/`). The release ID is the second-to-last URL path segment. `_find_model_dir()` searches `data_dir/<name>/` and returns the newest subdirectory (sorted descending by name).

---

## 3. Public API

### `uroseg/__init__.py`

```python
# Models
from uroseg.models import get_model, list_models
from uroseg.models.prostate import Prostate
from uroseg.models.bladder import Bladder

# Tools — single file
from uroseg.tools.map_labels import map_labels
from uroseg.tools.resample import resample
from uroseg.tools.preview import preview
from uroseg.tools.crop import crop
from uroseg.tools.largest_component import largest_component
from uroseg.tools.reorient import reorient
from uroseg.tools.transform_seg2image import transform_seg2image

# Tools — directory (multiprocessing)
from uroseg.tools.map_labels import map_labels_dir
from uroseg.tools.resample import resample_dir
from uroseg.tools.crop import crop_dir
from uroseg.tools.largest_component import largest_component_dir
from uroseg.tools.reorient import reorient_dir
from uroseg.tools.transform_seg2image import transform_seg2image_dir
```

### Per-tool public function pattern

Each tool in `uroseg/tools/<tool>.py` exposes:

```python
def map_labels(input: Path, output: Path, map: dict, **kwargs) -> Path:
    """Process a single file. Returns output path."""
    ...

def map_labels_dir(input_dir: Path, output_dir: Path, map: dict,
                   n_jobs: int = 1, **kwargs) -> None:
    """Process all NIfTI files in input_dir with n_jobs workers."""
    from concurrent.futures import ProcessPoolExecutor
    ...

def main() -> None:
    """CLI entry point — unchanged interface."""
    ...
```

The `_dir` variants use `concurrent.futures.ProcessPoolExecutor`. Tools that have no meaningful parallelism (e.g. `cpdir`, `preview`) still get a `_dir` variant for API consistency.

### Usage examples

```python
import uroseg

# Model prediction
uroseg.Prostate().predict("sub01.nii.gz", "output/")
uroseg.Prostate().predict_dir("images/", "output/", n_jobs=4)

# Install weights
uroseg.Prostate().install(data_dir=Path("~/.uroseg").expanduser())

# Utilities
uroseg.map_labels("seg.nii.gz", "out.nii.gz", map={"1,2": 1})
uroseg.map_labels_dir("segs/", "out/", map={"1,2": 1}, n_jobs=4)
uroseg.resample("img.nii.gz", "out.nii.gz", mm=1.0)
```

### CLI — unchanged

```bash
uroseg prostate --img sub01.nii.gz --out output/
uroseg install --model prostate
uroseg map --img seg.nii.gz --out out.nii.gz --map 1,2:1
uroseg resample --img img.nii.gz --out out.nii.gz --mm 1.0
```

---

## 4. nnUNet Helpers (`uroseg/nnunet/helpers.py`)

Contains only what is nnunet-specific:

```python
def setup_env(data_dir: Path) -> None:
    """Set nnUNet_raw, nnUNet_preprocessed, nnUNet_results, nnUNet_exports."""

def extract_dataset_id(nnunet_task: str) -> int:
    """Parse dataset ID from 'DatasetNNN_Name' string."""

def generate_dataset_json(model: SegModel, images_tr: Path) -> dict:
    """Build nnU-Net dataset.json from model labels and image count."""

def run_predict(model_dir: Path, inputs: list[Path], output_dir: Path,
                fold: int = 0, device: str = "cuda") -> None:
    """Run nnUNetPredictor on a list of input files."""
```

Generic model management (`_download_and_extract`, `_find_model_dir`, `_extract_release_id`) live in `uroseg/models/base.py` so future non-nnunet models (e.g. Monai) inherit them unchanged.

---

## 5. Training (`uroseg/nnunet/train.py`)

Functionally identical to current `train_nnunet.py` with these changes:

- `--training-dir` / `-d` (default: `./`) — where raw training data lives (`nnUNet/raw/<task>/`)
- `--data-dir` (or `UROSEG_DATA`) — where the exported model zip is saved after training
- `_add_trainer("nnUNetTrainerDAExtGPU")` — corrected from current `"nnUNetTrainerDAExt"`

```bash
# Raw data in current dir, export model to ~/.uroseg/
uroseg train nnunet prostate

# Explicit training dir
uroseg train nnunet prostate -d /data/mri/prostate/

# Custom model output dir
uroseg train nnunet prostate -d /data/mri/prostate/ --data-dir /models/uroseg/
```

Directory layout during training:
```
<training-dir>/
  nnUNet/
    raw/Dataset101_Prostate/
    preprocessed/Dataset101_Prostate/
    results/Dataset101_Prostate/
    exports/Dataset101_Prostate__nnUNetTrainerDAExtGPU__nnUNetPlans__3d_fullres__fold_0.zip
```

The exported zip stays in `<training-dir>/nnUNet/exports/` only. The `<data-dir>/<name>/<release>/` layout is exclusively for weights installed via `uroseg install` (downloaded from a release URL). Training never writes to `data-dir`.

---

## 6. What Does NOT Change

- `cli.py` external interface — all `uroseg <cmd>` commands work identically
- All tool logic and flags — only file paths move, no behavior changes
- `pyproject.toml` entry points — `uroseg = "uroseg.cli:main"` stays
- `resources/auglab/` — stays in place, same bundling
- Test structure in `tests/` — tests updated to new import paths but assertions unchanged
