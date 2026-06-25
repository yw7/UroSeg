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

Each model in `uroseg/resources/models/<name>.py` has three required attributes:

```python
from uroseg.models import ModelDef

MODEL = ModelDef(
    name="prostate",
    description="Prostate MRI-T2: whole gland, TZ, PZ",
    weights_url="https://github.com/yw7/uroseg/releases/download/r20260101/Dataset101_Prostate_r20260101.zip",
    labels={"background": 0, "prostate": [1, 2, 3], "prostate_tz": 2, "prostate_pz": 3},
)

NNUNET_TASK = "Dataset101_Prostate"

def inference(img, predict):
    return predict(img)
```

- `MODEL` — `ModelDef` instance (used by install, list)
- `NNUNET_TASK` — nnU-Net dataset string (used by inference, install, train); required for all current models; future non-nnU-Net backends will use different constants
- `inference(img, predict)` — required; owns the full prediction pipeline. `img` is the input `Image`, `predict` is a callable `Image → Image` that runs nnU-Net. Return value is the final segmentation `Image`. Pre/post processing go here when needed.

Example with custom processing:

```python
def inference(img, predict):
    img = img.resample([0.5, 0.5, 0.5])
    seg = predict(img)
    return keep_largest_component(seg)
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

### Inference pipeline — `inference.py`

```python
mod = load_model_module(organ)
model = mod.MODEL        # ModelDef
nnunet_task = mod.NNUNET_TASK

def predict_fn(img: Image) -> Image:
    # existing nnU-Net predict call
    ...

seg = mod.inference(img, predict_fn)
```

`inference.py` no longer accesses `model['nnunet_task']` or `model['labels']` — it gets `nnunet_task` from `mod.NNUNET_TASK` and delegates the full pipeline to `mod.inference`.

### `install.py`

```python
mod = load_model_module(name)
model = mod.MODEL          # ModelDef — for weights_url, name
nnunet_task = mod.NNUNET_TASK
```

Replaces `model['weights_url']`, `model['nnunet_task']`, `model['name']` with `model.weights_url`, `mod.NNUNET_TASK`, `model.name`.

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
| `uroseg/resources/models/prostate.py` | Create — replaces `prostate.json` |
| `uroseg/resources/models/bladder.py` | Create — replaces `bladder.json` |
| `uroseg/resources/models/prostate.json` | Delete |
| `uroseg/resources/models/bladder.json` | Delete |
| `uroseg/utils/utils.py` | Add `load_model_module()`; update `get_model()`, `get_all_models()` |
| `uroseg/commands/inference.py` | Use `load_model_module()`, `mod.NNUNET_TASK`, `mod.inference()` |
| `uroseg/commands/install.py` | Use `load_model_module()`, attribute access |
| `uroseg/commands/train_nnunet.py` | Use `load_model_module()`, attribute access |
| `pyproject.toml` | Remove `*.json` from package-data |
| `tests/` | Update model fixtures; add `ModelDef` and module-loading tests |

## Tests

- `test_model_def_fields`: `ModelDef` has `name`, `description`, `weights_url`, `labels`
- `test_load_model_module_prostate`: `load_model_module('prostate')` returns module with `MODEL`, `NNUNET_TASK`, `inference`
- `test_get_model_returns_modeldef`: `get_model('prostate')` returns a `ModelDef`
- `test_get_all_models_returns_dict`: `get_all_models()` returns `{'prostate': ModelDef, 'bladder': ModelDef}`
- `test_load_model_module_unknown_raises`: `load_model_module('nonexistent')` raises `ValueError`
- `test_inference_fn_called`: `mod.inference(img, predict_fn)` is called; `predict_fn` receives the (possibly preprocessed) image
- Update existing tests: any `model['key']` access → `model.key` or `mod.NNUNET_TASK`

## Out of scope

- External model files (outside the package) — models live in `uroseg/resources/models/` only
- Multiple NNUNET_TASK per model file — one task per model
- Changing `normalize_labels` — still used by `train_nnunet.py` via `model.labels`
