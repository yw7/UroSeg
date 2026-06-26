# UroSeg Model Definition Format

**Date:** 2026-06-25
**Status:** Approved

## Goal

Replace JSON model files with Python modules. Each model file is a self-contained unit: universal metadata (`ModelDef`), backend-specific constants (`NNUNET_TASK`), and a required `inference` function that owns the full prediction pipeline including any pre/post processing.

## Design

### `ModelDef` — universal contract

New file `uroseg/models.py`:

```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ModelDef:
    name: str
    description: str
    weights_url: str
    labels: dict
```

Four fields only — name, description, download URL, labels. Everything else is model-specific.

### Model Python files

Each model in `uroseg/resources/models/<name>.py` is a self-contained CLI entry point:

```python
import argparse
from uroseg.models import ModelDef
from uroseg.utils.inference_utils import add_common_inference_args, run_nnunet_predict

MODEL = ModelDef(
    name="prostate",
    description="Prostate MRI-T2: whole gland, TZ, PZ",
    weights_url="https://github.com/yw7/uroseg/releases/download/r20260101/Dataset101_Prostate_r20260101.zip",
    labels={"background": 0, "prostate": [1, 2, 3], "prostate_tz": 2, "prostate_pz": 3},
)

NNUNET_TASK = "Dataset101_Prostate"


def main():
    parser = argparse.ArgumentParser(prog='uroseg prostate')
    add_common_inference_args(parser)
    args = parser.parse_args()
    run_nnunet_predict(NNUNET_TASK, args)
```

- `MODEL` — `ModelDef` instance (used by install, list)
- `NNUNET_TASK` — nnU-Net dataset string (used by install, train)
- `main()` — full CLI entry point; owns argument parsing and inference pipeline. Can add model-specific args before calling `run_nnunet_predict`, or bypass it entirely for custom pipelines.

Example with a model-specific flag:

```python
def main():
    parser = argparse.ArgumentParser(prog='uroseg kidney')
    add_common_inference_args(parser)
    parser.add_argument('--bilateral', action='store_true', help='Segment both kidneys')
    args = parser.parse_args()
    run_nnunet_predict(NNUNET_TASK, args)
```

The old `prostate.json` and `bladder.json` are deleted.

### Discovery and loading — `utils.py`

Three functions replace the JSON-based loaders:

```python
def load_model_module(name: str):
    """Import and return the uroseg.resources.models.<name> module."""
    from importlib import import_module
    try:
        return import_module(f'uroseg.resources.models.{name}')
    except ModuleNotFoundError:
        available = sorted(m.stem for m in
            files('uroseg.resources.models').iterdir()
            if m.name.endswith('.py') and m.name != '__init__.py')
        raise ValueError(f"Unknown model: {name!r}. Available: {available}")

def get_model(name: str) -> ModelDef:
    return load_model_module(name).MODEL

def get_all_models() -> dict[str, ModelDef]:
    from importlib import import_module
    result = {}
    for p in files('uroseg.resources.models').iterdir():
        if p.name.endswith('.py') and p.name != '__init__.py':
            stem = p.name[:-3]
            result[stem] = import_module(f'uroseg.resources.models.{stem}').MODEL
    return result
```

### Shared inference utilities — `uroseg/utils/inference_utils.py`

New file providing utilities all model `main()` functions can import:

```python
def add_common_inference_args(parser) -> None:
    """Add --img, --out, --fold, --device, --data-dir, --out-suffix, --out-prefix, -r, -w, -q."""

def run_nnunet_predict(nnunet_task: str, args) -> None:
    """Full pipeline: collect inputs, reorient to RAS, run nnU-Net, save outputs."""

def download_weights(url: str, destination: Path) -> None:
    """Download zip from url and extract to destination directory."""
```

`run_nnunet_predict` uses `args.img`, `args.out`, `args.fold`, `args.device`, `args.data_dir`, `args.out_suffix`, `args.out_prefix`, `args.overwrite`, `args.quiet`.

### CLI dispatch — `cli.py`

`_dispatch_model` changes from calling `inference.main()` to:

```python
def _dispatch_model(model: str) -> None:
    from uroseg.utils.utils import load_model_module
    try:
        mod = load_model_module(model)
    except ValueError:
        print(f"Unknown model or subcommand: '{model}'\nRun 'uroseg --help' to see available models and commands.", file=sys.stderr)
        sys.exit(1)
    sys.argv = sys.argv[:1] + sys.argv[2:]
    mod.main()
```

`inference.py` is deleted.

### `install.py`

Uses `download_weights(url, destination)` from `inference_utils.py`. `download_and_extract` becomes:

```python
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
    download_weights(url, results_dir)
    print(f"  Done. Weights installed at {results_dir / nnunet_task}")
```

### `train_nnunet.py`

```python
mod = load_model_module(args.organ)
model = mod.MODEL          # ModelDef — for labels
nnunet_task = mod.NNUNET_TASK
```

Replaces `model['labels']`, `model['nnunet_task']` with `model.labels`, `mod.NNUNET_TASK`.

## Files

| File | Change |
|------|--------|
| `uroseg/models.py` | Create — `ModelDef` dataclass |
| `uroseg/resources/models/prostate.py` | Create — replaces `prostate.json`; has `MODEL`, `NNUNET_TASK`, `main()` |
| `uroseg/resources/models/bladder.py` | Create — replaces `bladder.json`; has `MODEL`, `NNUNET_TASK`, `main()` |
| `uroseg/resources/models/prostate.json` | Delete |
| `uroseg/resources/models/bladder.json` | Delete |
| `uroseg/utils/utils.py` | Add `load_model_module()`; update `get_model()`, `get_all_models()` |
| `uroseg/utils/inference_utils.py` | Create — `add_common_inference_args`, `run_nnunet_predict`, `download_weights` |
| `uroseg/commands/inference.py` | Delete |
| `uroseg/commands/install.py` | Use `download_weights(url, destination)` from `inference_utils` |
| `uroseg/commands/train_nnunet.py` | Use `load_model_module()`, attribute access |
| `uroseg/cli.py` | `_dispatch_model` calls `load_model_module(model).main()` |
| `pyproject.toml` | Remove `*.json` from package-data |
| `tests/` | Update model fixtures; add `ModelDef`, module-loading, and inference_utils tests |

## Tests

- `test_modeldef_fields`: `ModelDef` has `name`, `description`, `weights_url`, `labels`
- `test_load_model_module_prostate`: `load_model_module('prostate')` returns module with `MODEL`, `NNUNET_TASK`, `main`
- `test_get_model_returns_modeldef`: `get_model('prostate')` returns a `ModelDef`
- `test_get_all_models_returns_dict`: `get_all_models()` returns `{'prostate': ModelDef, 'bladder': ModelDef}`
- `test_load_model_module_unknown_raises`: `load_model_module('nonexistent')` raises `ValueError`
- `test_add_common_inference_args`: parser gets all expected flags
- `test_prostate_main_parser`: `uroseg prostate --help` exits 0; parser built with `prog='uroseg prostate'`
- Update existing tests: any `model['key']` access → `model.key` or `mod.NNUNET_TASK`

## Out of scope

- External model files (outside the package) — models live in `uroseg/resources/models/` only
- Multiple NNUNET_TASK per model file — one task per model
- Changing `normalize_labels` — still used by `train_nnunet.py` via `model.labels`
