# UroSeg Model Definition Format — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace JSON model files with Python modules that carry universal metadata (`ModelDef`), backend constants (`NNUNET_TASK`), and a required `inference` function owning the full prediction pipeline.

**Architecture:** A new `uroseg/models.py` defines the `ModelDef` dataclass (4 fields only). Two model Python files in `uroseg/resources/models/` replace the JSON files. `utils.py` gets new Python-module loaders; `inference.py`, `install.py`, and `train_nnunet.py` switch from dict key access to attribute access and call `mod.inference()`.

**Tech Stack:** Python 3.10+, `importlib.import_module`, `importlib.resources.files`, `dataclasses`, pytest

## Global Constraints

- `ModelDef` has exactly four fields: `name: str`, `description: str`, `weights_url: str`, `labels: dict` — no other fields
- Every model Python file must have: `MODEL` (a `ModelDef`), `NNUNET_TASK` (a `str`), `inference(img, predict)` (a callable) — no comments in model files
- `inference(img, predict)` is required on all models — no fallback default
- `load_model_module` raises `ValueError` (not `ModuleNotFoundError`) for unknown names, with list of available models
- `get_model` returns `ModelDef`, `get_all_models` returns `dict[str, ModelDef]`
- JSON files deleted, `*.json` package-data entry removed from `pyproject.toml`
- No changes to `normalize_labels` — still used by `train_nnunet.py`
- No changes to the `Image` class or `predict_nnunet.py`
- All 116 existing tests must remain passing after each task

---

### Task 1: `ModelDef` dataclass and model Python files

**Files:**
- Create: `uroseg/models.py`
- Create: `uroseg/resources/models/prostate.py`
- Create: `uroseg/resources/models/bladder.py`
- Test: `tests/test_models.py` (new)

Note: JSON files and pyproject.toml package-data are NOT changed in this task — they are removed in Task 2 together with the utils.py update, so all 116 existing tests stay green after this task.

**Interfaces:**
- Produces: `ModelDef` dataclass importable as `from uroseg.models import ModelDef`
- Produces: `uroseg.resources.models.prostate` module with `MODEL: ModelDef`, `NNUNET_TASK: str`, `inference(img, predict)`
- Produces: `uroseg.resources.models.bladder` module — same three attributes
- Consumed by: Tasks 2, 3, 4

- [ ] **Step 1: Write the failing tests**

Create `tests/test_models.py`:

```python
from dataclasses import fields
from uroseg.models import ModelDef
import uroseg.resources.models.prostate as prostate_mod
import uroseg.resources.models.bladder as bladder_mod


def test_modeldef_has_four_fields():
    names = {f.name for f in fields(ModelDef)}
    assert names == {'name', 'description', 'weights_url', 'labels'}


def test_modeldef_is_dataclass():
    m = ModelDef(name='x', description='d', weights_url='u', labels={})
    assert m.name == 'x'
    assert m.description == 'd'
    assert m.weights_url == 'u'
    assert m.labels == {}


def test_prostate_model_attributes():
    assert isinstance(prostate_mod.MODEL, ModelDef)
    assert prostate_mod.MODEL.name == 'prostate'
    assert isinstance(prostate_mod.NNUNET_TASK, str)
    assert callable(prostate_mod.inference)


def test_bladder_model_attributes():
    assert isinstance(bladder_mod.MODEL, ModelDef)
    assert bladder_mod.MODEL.name == 'bladder'
    assert isinstance(bladder_mod.NNUNET_TASK, str)
    assert callable(bladder_mod.inference)


def test_prostate_labels_have_background():
    assert 'background' in prostate_mod.MODEL.labels
    assert isinstance(prostate_mod.MODEL.labels['prostate'], list)


def test_bladder_labels_have_background():
    assert 'background' in bladder_mod.MODEL.labels
    assert isinstance(bladder_mod.MODEL.labels['bladder'], int)


def test_inference_passthrough():
    sentinel = object()
    def predict(img):
        return sentinel
    result = prostate_mod.inference(object(), predict)
    assert result is sentinel
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_models.py -v
```

Expected: all fail with `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Create `uroseg/models.py`**

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

- [ ] **Step 4: Create `uroseg/resources/models/prostate.py`**

```python
from uroseg.models import ModelDef

MODEL = ModelDef(
    name="prostate",
    description="Prostate: whole prostate (1), transition zone (2), peripheral zone (3)",
    weights_url="https://github.com/yw7/uroseg/releases/download/r20260101/Dataset101_Prostate_r20260101.zip",
    labels={"background": 0, "prostate": [1, 2, 3], "prostate_tz": 2, "prostate_pz": 3},
)

NNUNET_TASK = "Dataset101_Prostate"


def inference(img, predict):
    return predict(img)
```

- [ ] **Step 5: Create `uroseg/resources/models/bladder.py`**

```python
from uroseg.models import ModelDef

MODEL = ModelDef(
    name="bladder",
    description="Urinary bladder (CT)",
    weights_url="https://github.com/yw7/uroseg/releases/download/r20260101/Dataset010_Bladder_r20260101.zip",
    labels={"background": 0, "bladder": 1},
)

NNUNET_TASK = "Dataset010_Bladder"


def inference(img, predict):
    return predict(img)
```

- [ ] **Step 6: Run tests**

```
pytest tests/test_models.py -v
```

Expected: all 8 tests pass.

- [ ] **Step 7: Run full suite to confirm no regressions**

```
pytest --tb=short -q
```

Expected: all 116 existing tests still pass (JSON files are still present, so utils.py JSON loaders are unaffected). New `test_models.py` tests also pass.

- [ ] **Step 8: Commit**

```bash
git add uroseg/models.py uroseg/resources/models/prostate.py uroseg/resources/models/bladder.py tests/test_models.py
git commit -m "feat: ModelDef dataclass and Python model files"
```

---

### Task 2: `utils.py` — Python module loaders + delete JSON files

**Files:**
- Modify: `uroseg/utils/utils.py`
- Modify: `tests/test_utils.py`
- Delete: `uroseg/resources/models/prostate.json`
- Delete: `uroseg/resources/models/bladder.json`
- Modify: `pyproject.toml` lines 28-29 (remove `*.json` package-data entry)

**Interfaces:**
- Consumes: `ModelDef` from `uroseg.models` (Task 1)
- Consumes: `uroseg.resources.models.prostate` and `.bladder` modules (Task 1)
- Produces: `load_model_module(name: str)` → module object with `MODEL`, `NNUNET_TASK`, `inference`
- Produces: `get_model(name: str) -> ModelDef`
- Produces: `get_all_models() -> dict[str, ModelDef]`
- Consumed by: Tasks 3, 4

- [ ] **Step 1: Write failing tests**

In `tests/test_utils.py`, replace the `test_get_model_*` and `test_get_all_models` tests and add new ones. Find the block at lines 85–112 and replace it with:

```python
def test_load_model_module_prostate():
    from uroseg.utils.utils import load_model_module
    mod = load_model_module('prostate')
    assert hasattr(mod, 'MODEL')
    assert hasattr(mod, 'NNUNET_TASK')
    assert callable(mod.inference)


def test_load_model_module_unknown_raises_value_error():
    from uroseg.utils.utils import load_model_module
    with pytest.raises(ValueError, match='Unknown model'):
        load_model_module('nonexistent_model_xyz')


def test_get_model_returns_modeldef():
    from uroseg.models import ModelDef
    model = get_model('prostate')
    assert isinstance(model, ModelDef)
    assert model.name == 'prostate'
    assert 'labels' in vars(model) or hasattr(model, 'labels')
    assert 'background' in model.labels
    assert isinstance(model.labels['prostate'], list)


def test_get_model_bladder_returns_modeldef():
    from uroseg.models import ModelDef
    model = get_model('bladder')
    assert isinstance(model, ModelDef)
    assert model.name == 'bladder'
    assert 'bladder' in model.labels


def test_get_model_unknown_raises():
    with pytest.raises(Exception):
        get_model('nonexistent_organ')


def test_get_all_models_returns_modeldef_dict():
    from uroseg.models import ModelDef
    models = get_all_models()
    assert 'prostate' in models
    assert 'bladder' in models
    assert all(isinstance(m, ModelDef) for m in models.values())
```

- [ ] **Step 2: Run failing tests**

```
pytest tests/test_utils.py::test_load_model_module_prostate tests/test_utils.py::test_get_model_returns_modeldef tests/test_utils.py::test_get_all_models_returns_modeldef_dict -v
```

Expected: fail — `load_model_module` not found, and `get_model` still returns a dict.

- [ ] **Step 3: Update `uroseg/utils/utils.py`**

Replace the entire file:

```python
from __future__ import annotations
import argparse
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


def load_model_module(name: str):
    from importlib import import_module
    try:
        return import_module(f'uroseg.resources.models.{name}')
    except ModuleNotFoundError:
        available = sorted(
            p.name[:-3]
            for p in files('uroseg.resources.models').iterdir()
            if p.name.endswith('.py') and p.name != '__init__.py'
        )
        raise ValueError(
            f"Unknown model: {name!r}. Available: {available}\n"
            f"Run 'uroseg list' to see all models."
        )


def get_model(name: str):
    from uroseg.models import ModelDef
    return load_model_module(name).MODEL


def get_all_models() -> dict:
    from importlib import import_module
    result = {}
    for p in files('uroseg.resources.models').iterdir():
        if p.name.endswith('.py') and p.name != '__init__.py':
            stem = p.name[:-3]
            result[stem] = import_module(f'uroseg.resources.models.{stem}').MODEL
    return result


def normalize_labels(raw: dict) -> dict:
    """Normalize label dicts to ``{name: int | list[int]}``.

    Accepts four input formats:
    - TotalSpineSeg style (name→value): ``{"background": 0, "disc": [1,2,3]}``
    - Old integer-key style: ``{"0": "background", "1": "prostate"}``
    - Comma-key style: ``{"1,2,3": "prostate", "2": "pz"}``
    - Range-key style: ``{"1-3": "prostate", "2": "pz"}``
    """
    result: dict = {}
    for k, v in raw.items():
        if isinstance(v, (int, list)):
            result[str(k)] = v
        elif isinstance(v, str):
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
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_utils.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Delete JSON files and update pyproject.toml**

```bash
git rm uroseg/resources/models/prostate.json uroseg/resources/models/bladder.json
```

In `pyproject.toml`, remove the entire `[tool.setuptools.package-data]` section (lines 28-29):

```toml
# Remove these two lines entirely:
[tool.setuptools.package-data]
"uroseg.resources.models" = ["*.json"]
```

The file should go straight from `[tool.setuptools.packages.find]` to `[tool.setuptools_scm]`.

- [ ] **Step 6: Run full suite**

```
pytest --tb=short -q
```

Expected: all tests pass (including all 116 existing tests). The new `load_model_module` / `get_model` / `get_all_models` now scan `.py` files; JSON files are gone; tests that previously loaded JSON now load Python modules instead.

- [ ] **Step 7: Commit**

```bash
git add uroseg/utils/utils.py tests/test_utils.py pyproject.toml
git commit -m "feat: replace JSON loaders with load_model_module; get_model returns ModelDef; delete JSON files"
```

---

### Task 3: `inference_utils.py` + model `main()` + update `cli.py` + delete `inference.py`

**Files:**
- Create: `uroseg/utils/inference_utils.py`
- Modify: `uroseg/resources/models/prostate.py` (add `main()`)
- Modify: `uroseg/resources/models/bladder.py` (add `main()`)
- Modify: `uroseg/cli.py` (`_dispatch_model` calls `mod.main()`)
- Delete: `uroseg/commands/inference.py`
- Modify: `tests/test_inference.py` (rewrite to test `inference_utils` and model parsers)

**Interfaces:**
- Consumes: `load_model_module(name)` from `uroseg.utils.utils` (Task 2)
- Consumes: `find_model_dir`, `predict` from `uroseg.commands.predict_nnunet`
- Produces: `add_common_inference_args(parser)` — exported from `uroseg.utils.inference_utils`
- Produces: `run_nnunet_predict(nnunet_task: str, args) -> None` — exported from `uroseg.utils.inference_utils`
- Produces: `download_weights(url: str, destination: Path) -> None` — exported from `uroseg.utils.inference_utils`
- Produces: `prostate.main()` and `bladder.main()` — callable by `cli.py`
- Consumed by: Task 4 (`download_weights`), model `main()` calls in cli dispatch

- [ ] **Step 1: Write failing tests**

Replace `tests/test_inference.py` entirely:

```python
import argparse
import pytest
from unittest.mock import MagicMock, patch
from uroseg.utils.inference_utils import add_common_inference_args
import uroseg.resources.models.prostate as prostate_mod
import uroseg.resources.models.bladder as bladder_mod


def test_add_common_inference_args_required():
    parser = argparse.ArgumentParser()
    add_common_inference_args(parser)
    args = parser.parse_args(['-i', 'img.nii.gz', '-o', 'out/'])
    assert args.img == 'img.nii.gz'
    assert args.out == 'out/'


def test_add_common_inference_args_defaults():
    parser = argparse.ArgumentParser()
    add_common_inference_args(parser)
    args = parser.parse_args(['-i', 'img.nii.gz', '-o', 'out/'])
    assert args.fold == 0
    assert args.device == 'cuda'
    assert args.out_suffix == '_seg'
    assert args.out_prefix == ''
    assert args.overwrite is False
    assert args.quiet is False


def test_prostate_module_has_main():
    assert callable(prostate_mod.main)


def test_bladder_module_has_main():
    assert callable(bladder_mod.main)


def test_prostate_main_parser_prog():
    with patch('sys.argv', ['uroseg', '-h']):
        parser = argparse.ArgumentParser(prog='uroseg prostate')
        add_common_inference_args(parser)
        assert parser.prog == 'uroseg prostate'


def test_load_model_module_called_for_valid_organ():
    from uroseg.utils.utils import load_model_module
    mod = load_model_module('prostate')
    assert mod.MODEL.name == 'prostate'
    assert mod.NNUNET_TASK == 'Dataset101_Prostate'


def test_load_model_module_invalid_organ_raises():
    from uroseg.utils.utils import load_model_module
    with pytest.raises(ValueError):
        load_model_module('nonexistent_organ_xyz')
```

- [ ] **Step 2: Run failing tests**

```
pytest tests/test_inference.py -v
```

Expected: `test_add_common_inference_args_*` fail (`inference_utils` not yet created), model `has_main` tests fail.

- [ ] **Step 3: Create `uroseg/utils/inference_utils.py`**

```python
from __future__ import annotations
import argparse
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from uroseg.utils.image import Image
from uroseg.utils.utils import add_common_args, collect_niftis, build_output_path, resolve_data_path
from uroseg.commands.predict_nnunet import find_model_dir, predict


def add_common_inference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--img', '-i', required=True, help='Input image file or folder')
    parser.add_argument('--out', '-o', required=True, help='Output folder')
    parser.add_argument('--fold', '-f', type=int, default=0, help='nnU-Net fold (default: 0)')
    parser.add_argument('--device', '-d', default='cuda', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--out-suffix', default='_seg', help='Output filename suffix')
    parser.add_argument('--out-prefix', default='', help='Output filename prefix')
    parser.add_argument('--data-dir', default=None, help='Override data path (or set UROSEG_DATA)')
    add_common_args(parser)


def run_nnunet_predict(nnunet_task: str, args) -> None:
    data_path = resolve_data_path(args.data_dir)
    model_dir = find_model_dir(nnunet_task, data_path)

    inputs = collect_niftis(args.img)
    if not inputs:
        print(f"No NIfTI files found in {args.img}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_in = Path(tmp) / 'inputs'
        tmp_out = Path(tmp) / 'outputs'
        tmp_in.mkdir()
        tmp_out.mkdir()

        if not args.quiet:
            print(f"Reorienting {len(inputs)} image(s) to RAS...")
        reoriented = []
        for i, p in enumerate(inputs):
            img = Image.load(p)
            img = img.reorient('RAS')
            out_p = tmp_in / f"{i:04d}_{p.name}"
            img.save(out_p)
            reoriented.append(out_p)

        predict(
            model_dir=model_dir,
            input_files=reoriented,
            output_dir=tmp_out,
            fold=args.fold,
            device=args.device,
        )

        for inp, pred_file in zip(inputs, sorted(tmp_out.glob('*.nii.gz'))):
            dest = build_output_path(inp, out_dir, args.out_prefix, args.out_suffix)
            if not args.overwrite and dest.exists():
                continue
            img = Image.load(pred_file)
            img.save(dest)

    if not args.quiet:
        print(f"Segmentations saved to {out_dir}")


def download_weights(url: str, destination: Path) -> None:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    zip_name = url.split('/')[-1]
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / zip_name
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc=zip_name) as bar:
            def reporthook(count, block_size, total_size):
                if total_size > 0:
                    bar.total = total_size
                bar.update(block_size)
            urllib.request.urlretrieve(url, zip_path, reporthook=reporthook)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(destination)
```

- [ ] **Step 4: Add `main()` to `prostate.py`**

The full updated `uroseg/resources/models/prostate.py`:

```python
import argparse
from uroseg.models import ModelDef
from uroseg.utils.inference_utils import add_common_inference_args, run_nnunet_predict

MODEL = ModelDef(
    name="prostate",
    description="Prostate: whole prostate (1), transition zone (2), peripheral zone (3)",
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

- [ ] **Step 5: Add `main()` to `bladder.py`**

The full updated `uroseg/resources/models/bladder.py`:

```python
import argparse
from uroseg.models import ModelDef
from uroseg.utils.inference_utils import add_common_inference_args, run_nnunet_predict

MODEL = ModelDef(
    name="bladder",
    description="Urinary bladder (CT)",
    weights_url="https://github.com/yw7/uroseg/releases/download/r20260101/Dataset010_Bladder_r20260101.zip",
    labels={"background": 0, "bladder": 1},
)

NNUNET_TASK = "Dataset010_Bladder"


def main():
    parser = argparse.ArgumentParser(prog='uroseg bladder')
    add_common_inference_args(parser)
    args = parser.parse_args()
    run_nnunet_predict(NNUNET_TASK, args)
```

- [ ] **Step 6: Update `cli.py` — `_dispatch_model` calls `mod.main()`**

Replace `_dispatch_model`:

```python
def _dispatch_model(model: str) -> None:
    from uroseg.utils.utils import load_model_module
    try:
        mod = load_model_module(model)
    except ValueError:
        print(
            f"Unknown model or subcommand: '{model}'\n"
            f"Run 'uroseg --help' to see available models and commands.",
            file=sys.stderr,
        )
        sys.exit(1)
    sys.argv = sys.argv[:1] + sys.argv[2:]
    mod.main()
```

- [ ] **Step 7: Delete `uroseg/commands/inference.py`**

```bash
git rm uroseg/commands/inference.py
```

- [ ] **Step 8: Run tests**

```
pytest tests/test_inference.py -v
```

Expected: all 7 tests pass.

- [ ] **Step 9: Run full suite**

```
pytest --tb=short -q
```

Expected: all tests pass. `test_cli.py` tests that previously went through `inference.py` now route through `mod.main()` — they should pass since the dispatch logic is equivalent.

- [ ] **Step 10: Commit**

```bash
git add uroseg/utils/inference_utils.py uroseg/resources/models/prostate.py uroseg/resources/models/bladder.py uroseg/cli.py tests/test_inference.py
git rm uroseg/commands/inference.py
git commit -m "feat: inference_utils.py; model main() as CLI entry; cli.py dispatches to mod.main()"
```

---

### Task 4: `install.py` — use `download_weights`

**Files:**
- Modify: `uroseg/commands/install.py`
- Modify: `tests/test_install.py`

**Interfaces:**
- Consumes: `download_weights(url: str, destination: Path)` from `uroseg.utils.inference_utils` (Task 3)
- Consumes: `load_model_module(name)`, `get_all_models()` from `uroseg.utils.utils` (Task 2)
- `download_and_extract(model: ModelDef, nnunet_task: str, data_path: Path)` — updated signature (no `store_export`)

- [ ] **Step 1: Write failing test**

In `tests/test_install.py`, update `test_download_and_extract_no_url` to use the new 3-arg signature (drop `store_export`):

```python
import pytest
from pathlib import Path
from uroseg.models import ModelDef
from uroseg.commands.install import (
    extract_release_id,
    get_install_dir,
    is_installed,
    download_and_extract,
)


def test_extract_release_id():
    url = 'https://github.com/yw7/uroseg/releases/download/r20260101/Dataset001_Prostate_r20260101.zip'
    assert extract_release_id(url) == 'r20260101'


def test_get_install_dir(tmp_path):
    url = 'https://github.com/yw7/uroseg/releases/download/r20260101/Dataset001_Prostate_r20260101.zip'
    d = get_install_dir('Dataset001_Prostate', url, tmp_path)
    assert d == tmp_path / 'nnUNet' / 'results' / 'r20260101' / 'Dataset001_Prostate'


def test_is_installed_false(tmp_path):
    url = 'https://example.com/releases/download/r20260101/X.zip'
    assert not is_installed('Dataset001_Prostate', url, tmp_path)


def test_is_installed_true(tmp_path):
    url = 'https://example.com/releases/download/r20260101/X.zip'
    install_dir = get_install_dir('Dataset001_Prostate', url, tmp_path)
    install_dir.mkdir(parents=True)
    assert is_installed('Dataset001_Prostate', url, tmp_path)


def test_download_and_extract_no_url(tmp_path, capsys):
    model = ModelDef(name='test', description='', weights_url='', labels={})
    download_and_extract(model, 'Dataset999_Test', tmp_path)
    captured = capsys.readouterr()
    assert 'skip' in captured.out.lower() or 'no weights' in captured.out.lower()
```

- [ ] **Step 2: Run failing test**

```
pytest tests/test_install.py::test_download_and_extract_no_url -v
```

Expected: fail — `download_and_extract` still has 4 parameters.

- [ ] **Step 3: Update `uroseg/commands/install.py`**

Full replacement:

```python
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from uroseg.utils.utils import resolve_data_path, load_model_module, get_all_models
from uroseg.utils.inference_utils import download_weights


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
    download_weights(url, results_dir)
    print(f"  Done. Weights installed at {results_dir / nnunet_task}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Download and install UroSeg model weights.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', nargs='+', metavar='MODEL',
                       help='One or more organ model names (e.g. prostate bladder)')
    group.add_argument('--all', action='store_true', help='Install all available models')
    parser.add_argument('--data-dir', default=None, help='Override data path')
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
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_install.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Run full suite**

```
pytest --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add uroseg/commands/install.py tests/test_install.py
git commit -m "feat: install.py uses download_weights(url, destination) from inference_utils"
```

---

### Task 5: `train_nnunet.py` — attribute access

**Files:**
- Modify: `uroseg/commands/train_nnunet.py`

**Interfaces:**
- Consumes: `load_model_module(name)` from `uroseg.utils.utils` (Task 2)
- `generate_dataset_json(model: ModelDef, images_tr_dir)` — `model.labels` instead of `model["labels"]`

- [ ] **Step 1: Check existing tests still pass before touching the file**

```
pytest tests/test_commands.py -v -k train
```

Expected: existing train tests pass.

- [ ] **Step 2: Update `uroseg/commands/train_nnunet.py`**

Two changes only — replace dict access with attribute access. Find and change:

```python
# Old
model = _utils.get_model(args.organ)
...
nnunet_task = model["nnunet_task"]
...
dataset_json = generate_dataset_json(model, images_tr)
```

```python
# New
mod = _utils.load_model_module(args.organ)
...
nnunet_task = mod.NNUNET_TASK
...
dataset_json = generate_dataset_json(mod.MODEL, images_tr)
```

And in `generate_dataset_json`, change `model["labels"]` → `model.labels`:

```python
def generate_dataset_json(model, images_tr_dir: Path) -> dict:
    labels = normalize_labels(model.labels)
    ...
```

Also update the help text on the `organ` argument:

```python
parser.add_argument("organ", help="Organ name matching resources/models/<organ>.py")
```

Full updated `uroseg/commands/train_nnunet.py`:

```python
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import uroseg.utils.utils as _utils
from uroseg.utils.utils import normalize_labels


def extract_dataset_id(nnunet_task: str) -> int:
    match = re.match(r'Dataset(\d+)_', nnunet_task)
    if not match:
        raise ValueError(f"Invalid nnunet_task format: {nnunet_task!r} — expected 'DatasetNNN_Name'")
    return int(match.group(1))


def generate_dataset_json(model, images_tr_dir: Path) -> dict:
    labels = normalize_labels(model.labels)

    all_values: set[int] = set()
    has_regions = False
    for v in labels.values():
        if isinstance(v, list):
            all_values.update(int(x) for x in v)
            has_regions = True
        elif int(v) != 0:
            all_values.add(int(v))

    dataset: dict = {
        "channel_names": {"0": "MRI"},
        "labels": labels,
        "numTraining": len(list(images_tr_dir.glob("*.nii.gz"))),
        "file_ending": ".nii.gz",
    }
    if has_regions:
        dataset["regions_class_order"] = sorted(all_values)
    return dataset


def setup_nnunet_env(data_path: Path) -> None:
    nnunet_dir = data_path / "nnUNet"
    os.environ["nnUNet_raw"] = str(nnunet_dir / "raw")
    os.environ["nnUNet_preprocessed"] = str(nnunet_dir / "preprocessed")
    os.environ["nnUNet_results"] = str(nnunet_dir / "results")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='uroseg train nnunet',
        description="Train a UroSeg model with nnU-Net and AugLab augmentation.",
    )
    parser.add_argument("organ",
                        help="Organ name matching resources/models/<organ>.py")
    parser.add_argument("--fold", "-f", type=int, default=0,
                        help="nnU-Net fold number (0–4, default: 0)")
    parser.add_argument("--auglab-config", default=None,
                        help="Path to AugLab augmentation config JSON (optional)")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs (default: 1)")
    parser.add_argument("--data-dir", default=None,
                        help="Override UROSEG_DATA / ~/uroseg/ with this path")
    args = parser.parse_args()

    mod = _utils.load_model_module(args.organ)
    data_path = _utils.resolve_data_path(args.data_dir)
    setup_nnunet_env(data_path)

    nnunet_task = mod.NNUNET_TASK
    dataset_id = extract_dataset_id(nnunet_task)

    raw_dir = data_path / "nnUNet" / "raw" / nnunet_task
    images_tr = raw_dir / "imagesTr"

    if not images_tr.exists():
        print(
            f"Error: training images directory not found: {images_tr}\n"
            f"Place training images in {images_tr}/ (filename pattern: <case>_0000.nii.gz)",
            file=sys.stderr,
        )
        sys.exit(1)

    dataset_json = generate_dataset_json(mod.MODEL, images_tr)
    dataset_json_path = raw_dir / "dataset.json"
    dataset_json_path.write_text(json.dumps(dataset_json, indent=2))
    print(f"Generated {dataset_json_path}")

    preprocessed_dir = data_path / "nnUNet" / "preprocessed" / nnunet_task
    if not preprocessed_dir.exists():
        print("Running nnU-Net planning and preprocessing...")
        subprocess.run(
            ["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id), "--verify_dataset_integrity"],
            check=True,
        )

    from auglab.add_trainer import add_trainer as _add_trainer
    _add_trainer("nnUNetTrainerDAExt")

    if args.auglab_config:
        os.environ["AUGLAB_CONFIG"] = str(args.auglab_config)

    results_dir = data_path / "nnUNet" / "results"
    print(f"Starting training (dataset {dataset_id}, fold {args.fold})...")
    subprocess.run(
        [
            "nnUNetv2_train",
            str(dataset_id),
            "3d_fullres",
            str(args.fold),
            "--trainer", "nnUNetTrainerDAExt",
        ],
        check=True,
    )

    trainer_dir = (
        results_dir
        / nnunet_task
        / "nnUNetTrainerDAExt__nnUNetPlans__3d_fullres"
        / f"fold_{args.fold}"
    )
    print(f"\nTraining complete. Model saved to:\n  {trainer_dir}")
```

- [ ] **Step 3: Run full suite**

```
pytest --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add uroseg/commands/train_nnunet.py
git commit -m "feat: train_nnunet.py uses load_model_module and ModelDef attribute access"
```

---

### Task 6: Update README for Python model files

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: completed model file format from Tasks 1-5

- [ ] **Step 1: Update "Training — Step 1" section**

Replace the heading "### 1. Create the model JSON" and its JSON examples with:

````markdown
### 1. Create the model Python file

Add `uroseg/resources/models/<organ>.py`:

```python
import argparse
from uroseg.models import ModelDef
from uroseg.utils.inference_utils import add_common_inference_args, run_nnunet_predict

MODEL = ModelDef(
    name="kidney",
    description="Kidney (CT)",
    weights_url="https://github.com/yw7/uroseg/releases/download/r.../Dataset020_Kidney_r....zip",
    labels={"background": 0, "kidney": 1},
)

NNUNET_TASK = "Dataset020_Kidney"


def main():
    parser = argparse.ArgumentParser(prog='uroseg kidney')
    add_common_inference_args(parser)
    args = parser.parse_args()
    run_nnunet_predict(NNUNET_TASK, args)
```

Region-based model (sigmoid per region):

```python
import argparse
from uroseg.models import ModelDef
from uroseg.utils.inference_utils import add_common_inference_args, run_nnunet_predict

MODEL = ModelDef(
    name="prostate",
    description="Prostate MRI-T2: whole prostate (1), peripheral zone (2), central zone (3)",
    weights_url="https://github.com/yw7/uroseg/releases/download/r.../Dataset001_Prostate_r....zip",
    labels={"background": 0, "prostate": [1, 2, 3], "prostate_tz": 2, "prostate_pz": 3},
)

NNUNET_TASK = "Dataset001_Prostate"


def main():
    parser = argparse.ArgumentParser(prog='uroseg prostate')
    add_common_inference_args(parser)
    args = parser.parse_args()
    run_nnunet_predict(NNUNET_TASK, args)
```

`labels` values can be an `int` (single label) or a `list[int]` (region = union of sub-labels). When any label is a list, `uroseg train nnunet` automatically sets `regions_class_order` for nnU-Net region-based training. To add model-specific CLI flags, add `parser.add_argument(...)` calls before `parse_args()` and use them in your own inference logic instead of calling `run_nnunet_predict`.
````

- [ ] **Step 2: Update "Contributing — Adding a New Organ Model" section**

Replace step 1 from "Create `uroseg/resources/models/<organ>.json`" to "Create `uroseg/resources/models/<organ>.py`" per the Python format above.

- [ ] **Step 3: Verify no stale JSON references**

```bash
grep -n "\.json\|Create.*json" README.md
```

Expected: no model JSON creation references (install commands referencing `--model` names are fine).

- [ ] **Step 4: Run full suite**

```
pytest --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: update README to Python model file format with main()"
```
