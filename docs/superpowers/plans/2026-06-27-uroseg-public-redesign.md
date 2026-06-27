# UroSeg Public-Ready Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize UroSeg into a logically structured, contribution-ready package with a proper model class hierarchy, clean public API, consolidated nnunet helpers, and separated training/model-data directories.

**Architecture:** Replace the `ModelDef` dataclass with a `SegModel`/`NNUNetSegModel` class hierarchy; move tools from `commands/` to `tools/`; consolidate nnunet plumbing into `nnunet/`; expose a flat public API in `uroseg/__init__.py`. All existing CLI behavior stays identical.

**Tech Stack:** Python 3.10+, nibabel, numpy, tqdm, concurrent.futures, nnunetv2, auglab

## Global Constraints

- All existing `uroseg <cmd>` CLI commands must work identically after every task
- Class name: `SegModel` (not `Model`, not `ModelDef`)
- Training raw-data arg: `--training-dir` / `-d`, default `./`
- Data layout for installed models: `data_dir/<model_name>/<release_id>/`
- `_find_model_dir` and `_download_zip`/`_extract_zip` live in `models/base.py`, NOT in `nnunet/`
- `nnunet/helpers.py` contains only: `setup_env`, `extract_dataset_id`, `generate_dataset_json`, `run_predict`
- `_add_trainer` arg must be `"nnUNetTrainerDAExtGPU"` (not `"nnUNetTrainerDAExt"`)
- Public tool functions: `<tool>(input, output, ...)` returns `Path`; `<tool>_dir(input_dir, output_dir, ..., n_jobs=1)` returns `None`
- `resources/auglab/` stays unchanged; pyproject.toml package-data entry stays
- No behavior changes to any tool — only file locations and import paths change

---

## File Map

| New path | Source | Action |
|---|---|---|
| `uroseg/models/__init__.py` | new | create |
| `uroseg/models/base.py` | new | create |
| `uroseg/models/prostate.py` | `uroseg/resources/models/prostate.py` | rewrite |
| `uroseg/models/bladder.py` | `uroseg/resources/models/bladder.py` | rewrite |
| `uroseg/nnunet/__init__.py` | new | create |
| `uroseg/nnunet/helpers.py` | `train_nnunet.py` + `predict_nnunet.py` | extract |
| `uroseg/nnunet/train.py` | `commands/train_nnunet.py` + `commands/train.py` | rewrite |
| `uroseg/nnunet/predict.py` | `commands/predict_nnunet.py` + `utils/inference_utils.py` | merge |
| `uroseg/tools/__init__.py` | new | create |
| `uroseg/tools/map_labels.py` | `commands/map_labels.py` | move + extend |
| `uroseg/tools/resample.py` | `commands/resample.py` | move + extend |
| `uroseg/tools/crop.py` | `commands/crop_image2seg.py` | move + rename + extend |
| `uroseg/tools/largest_component.py` | `commands/largest_component.py` | move + extend |
| `uroseg/tools/reorient.py` | `commands/reorient_canonical.py` | move + rename + extend |
| `uroseg/tools/transform_seg2image.py` | `commands/transform_seg2image.py` | move + extend |
| `uroseg/tools/preview.py` | `commands/preview_jpg.py` | move + rename + extend |
| `uroseg/tools/cpdir.py` | `commands/cpdir.py` | move + extend |
| `uroseg/__init__.py` | existing | rewrite |
| `uroseg/cli.py` | existing | update imports |
| `uroseg/utils/utils.py` | existing | update shims |

**Deleted:** `uroseg/models.py`, `uroseg/resources/models/`, `uroseg/commands/`, `uroseg/utils/inference_utils.py`

---

### Task 1: SegModel base classes

**Files:**
- Create: `uroseg/models/__init__.py`
- Create: `uroseg/models/base.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Produces:
  - `SegModel` — base class with `name`, `description`, `weights_url`, `labels` class attrs; `install(data_dir: Path)`, `predict(input: Path, output_dir: Path, **kwargs)`, `predict_dir(input_dir, output_dir, n_jobs=1, **kwargs)`
  - `NNUNetSegModel(SegModel)` — adds `nnunet_task: str`; implements `predict()` with lazy import of `uroseg.nnunet.helpers.run_predict`
  - `_extract_release_id(url: str) -> str`
  - `_download_zip(url: str, tmp_dir: Path) -> Path`
  - `_extract_zip(zip_path: Path, dest: Path) -> None`
  - `_find_model_dir(name: str, data_dir: Path) -> Path`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_models.py  (full replacement)
from __future__ import annotations
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_extract_release_id():
    from uroseg.models.base import _extract_release_id
    url = 'https://github.com/x/releases/download/r20260101/X.zip'
    assert _extract_release_id(url) == 'r20260101'


def test_find_model_dir_newest_first(tmp_path):
    from uroseg.models.base import _find_model_dir
    (tmp_path / 'prostate' / 'r20260101').mkdir(parents=True)
    (tmp_path / 'prostate' / 'r20260201').mkdir(parents=True)
    result = _find_model_dir('prostate', tmp_path)
    assert result.name == 'r20260201'


def test_find_model_dir_not_found(tmp_path):
    from uroseg.models.base import _find_model_dir
    with pytest.raises(FileNotFoundError):
        _find_model_dir('nonexistent', tmp_path)


def test_segmodel_install_skips_when_no_url(tmp_path, capsys):
    from uroseg.models.base import SegModel
    class EmptyModel(SegModel):
        name = 'test'; description = 'd'; weights_url = ''; labels = {}
    EmptyModel().install(tmp_path)
    assert 'skip' in capsys.readouterr().out.lower()


def test_segmodel_install_skips_when_already_installed(tmp_path, capsys):
    from uroseg.models.base import SegModel
    class M(SegModel):
        name = 'x'; description = ''; weights_url = 'https://h/releases/download/r1/x.zip'; labels = {}
    (tmp_path / 'x' / 'r1').mkdir(parents=True)
    M().install(tmp_path)
    assert 'already installed' in capsys.readouterr().out.lower()


def test_nnunet_segmodel_is_segmodel():
    from uroseg.models.base import SegModel, NNUNetSegModel
    assert issubclass(NNUNetSegModel, SegModel)


def test_segmodel_predict_raises():
    from uroseg.models.base import SegModel
    class M(SegModel):
        name = 'x'; description = ''; weights_url = ''; labels = {}
    with pytest.raises(NotImplementedError):
        M().predict(Path('x.nii.gz'), Path('/tmp'))


def test_segmodel_predict_dir_calls_predict(tmp_path):
    from uroseg.models.base import SegModel
    import nibabel as nib, numpy as np
    f = tmp_path / 'a.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((5,5,5), dtype=np.int16), np.eye(4)), f)

    class M(SegModel):
        name = 'x'; description = ''; weights_url = ''; labels = {}
        calls = []
        def predict(self, input, output_dir, **kwargs):
            M.calls.append(input)

    M().predict_dir(tmp_path, tmp_path / 'out', n_jobs=1)
    assert len(M.calls) == 1
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_models.py -v
```
Expected: FAIL (ImportError or AttributeError — modules don't exist yet)

- [ ] **Step 3: Create `uroseg/models/__init__.py`**

```python
# uroseg/models/__init__.py
```
(empty for now — populated in Task 2)

- [ ] **Step 4: Create `uroseg/models/base.py`**

```python
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
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_models.py -v
```
Expected: all 8 tests PASS

- [ ] **Step 6: Commit**

```bash
git add uroseg/models/__init__.py uroseg/models/base.py tests/test_models.py
git commit -m "feat: add SegModel/NNUNetSegModel base classes with install and predict_dir"
```

---

### Task 2: Concrete model classes + registry + delete old ModelDef

**Files:**
- Create: `uroseg/models/prostate.py`
- Create: `uroseg/models/bladder.py`
- Modify: `uroseg/models/__init__.py`
- Modify: `uroseg/utils/utils.py` (update shims for load_model_module, get_all_models)
- Delete: `uroseg/models.py`
- Delete: `uroseg/resources/models/prostate.py`, `uroseg/resources/models/bladder.py`, `uroseg/resources/models/__init__.py`
- Modify: `tests/test_models.py`
- Modify: `tests/test_utils.py`
- Modify: `tests/test_inference.py`
- Modify: `tests/test_install.py`

**Interfaces:**
- Consumes: `NNUNetSegModel` from Task 1
- Produces:
  - `Prostate(NNUNetSegModel)` — class with `name="prostate"`, `nnunet_task="Dataset101_Prostate"`, `MODEL=Prostate()` module-level compat alias, `NNUNET_TASK` alias, `main()`
  - `Bladder(NNUNetSegModel)` — same pattern, `name="bladder"`, `nnunet_task="Dataset010_Bladder"`
  - `get_model(name: str) -> SegModel` — from `uroseg.models`
  - `list_models() -> list[str]` — from `uroseg.models`

- [ ] **Step 1: Create `uroseg/models/prostate.py`**

```python
from __future__ import annotations
from uroseg.models.base import NNUNetSegModel


class Prostate(NNUNetSegModel):
    name = "prostate"
    description = "Prostate: whole (1), transition zone (2), peripheral zone (3)"
    weights_url = "https://github.com/yw7/uroseg/releases/download/r20260101/Dataset101_Prostate_r20260101.zip"
    labels = {"background": 0, "prostate": [1, 2, 3], "prostate_tz": 2, "prostate_pz": 3}
    nnunet_task = "Dataset101_Prostate"


# Backward-compat aliases (used by existing consumers of resources/models/prostate)
MODEL = Prostate()
NNUNET_TASK = Prostate.nnunet_task


def main() -> None:
    import argparse
    from uroseg.nnunet.predict import add_inference_args, run_predict_cli
    parser = argparse.ArgumentParser(prog='uroseg prostate')
    add_inference_args(parser)
    args = parser.parse_args()
    run_predict_cli(Prostate(), args)
```

- [ ] **Step 2: Create `uroseg/models/bladder.py`**

```python
from __future__ import annotations
from uroseg.models.base import NNUNetSegModel


class Bladder(NNUNetSegModel):
    name = "bladder"
    description = "Urinary bladder (CT)"
    weights_url = "https://github.com/yw7/uroseg/releases/download/r20260101/Dataset010_Bladder_r20260101.zip"
    labels = {"background": 0, "bladder": 1}
    nnunet_task = "Dataset010_Bladder"


MODEL = Bladder()
NNUNET_TASK = Bladder.nnunet_task


def main() -> None:
    import argparse
    from uroseg.nnunet.predict import add_inference_args, run_predict_cli
    parser = argparse.ArgumentParser(prog='uroseg bladder')
    add_inference_args(parser)
    args = parser.parse_args()
    run_predict_cli(Bladder(), args)
```

- [ ] **Step 3: Update `uroseg/models/__init__.py`**

```python
from __future__ import annotations
from uroseg.models.base import SegModel, NNUNetSegModel
from uroseg.models.prostate import Prostate
from uroseg.models.bladder import Bladder

_REGISTRY: dict[str, type[SegModel]] = {
    cls.name: cls for cls in [Prostate, Bladder]
}


def get_model(name: str) -> SegModel:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model: {name!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]()


def list_models() -> list[str]:
    return list(_REGISTRY)
```

- [ ] **Step 4: Update `uroseg/utils/utils.py` — replace load_model_module, get_model, get_all_models with shims**

Replace the three functions (lines 66–92 in current file) with:

```python
def load_model_module(name: str):
    """Shim: load model module from uroseg.models.<name>."""
    from uroseg.models import list_models
    import importlib
    if name not in list_models():
        available = list_models()
        raise ValueError(
            f"Unknown model: {name!r}. Available: {available}\n"
            f"Run 'uroseg list' to see all models."
        )
    return importlib.import_module(f'uroseg.models.{name}')


def get_model(name: str):
    from uroseg.models import get_model as _get_model
    return _get_model(name)


def get_all_models() -> dict:
    from uroseg.models import list_models, get_model as _get_model
    return {name: _get_model(name) for name in list_models()}
```

- [ ] **Step 5: Delete old model files**

```bash
rm uroseg/models.py
rm uroseg/resources/models/prostate.py
rm uroseg/resources/models/bladder.py
rm uroseg/resources/models/__init__.py
rmdir uroseg/resources/models
```

- [ ] **Step 6: Update `tests/test_models.py` — add class-based tests**

Append to the existing test file (keep all existing passing tests):

```python
def test_prostate_class_attrs():
    from uroseg.models.prostate import Prostate
    p = Prostate()
    assert p.name == 'prostate'
    assert p.nnunet_task == 'Dataset101_Prostate'
    assert p.labels['background'] == 0
    assert isinstance(p.labels['prostate'], list)


def test_bladder_class_attrs():
    from uroseg.models.bladder import Bladder
    b = Bladder()
    assert b.name == 'bladder'
    assert b.labels['bladder'] == 1


def test_get_model_returns_prostate():
    from uroseg.models import get_model
    from uroseg.models.prostate import Prostate
    m = get_model('prostate')
    assert isinstance(m, Prostate)
    assert m.name == 'prostate'


def test_list_models():
    from uroseg.models import list_models
    models = list_models()
    assert 'prostate' in models
    assert 'bladder' in models


def test_get_model_unknown_raises():
    from uroseg.models import get_model
    with pytest.raises(ValueError, match='Unknown model'):
        get_model('nonexistent_xyz')


def test_prostate_main_is_callable():
    from uroseg.models.prostate import main
    assert callable(main)


def test_compat_model_attr():
    from uroseg.models.prostate import MODEL, NNUNET_TASK
    assert MODEL.name == 'prostate'
    assert NNUNET_TASK == 'Dataset101_Prostate'
```

- [ ] **Step 7: Update `tests/test_utils.py`**

Replace the three model-assertion tests at the bottom (lines 99–127) with:

```python
def test_load_model_module_prostate():
    from uroseg.utils.utils import load_model_module
    mod = load_model_module('prostate')
    assert hasattr(mod, 'MODEL')
    assert hasattr(mod, 'NNUNET_TASK')
    assert callable(mod.main)


def test_load_model_module_unknown_raises_value_error():
    from uroseg.utils.utils import load_model_module
    with pytest.raises(ValueError, match='Unknown model'):
        load_model_module('nonexistent_model_xyz')


def test_get_model_returns_prostate():
    from uroseg.utils.utils import get_model
    from uroseg.models.prostate import Prostate
    model = get_model('prostate')
    assert isinstance(model, Prostate)
    assert model.name == 'prostate'
    assert 'background' in model.labels
    assert isinstance(model.labels['prostate'], list)


def test_get_model_bladder():
    from uroseg.utils.utils import get_model
    from uroseg.models.bladder import Bladder
    model = get_model('bladder')
    assert isinstance(model, Bladder)
    assert model.name == 'bladder'


def test_get_model_unknown_raises():
    from uroseg.utils.utils import get_model
    with pytest.raises(Exception):
        get_model('nonexistent_organ')


def test_get_all_models_returns_dict():
    from uroseg.utils.utils import get_all_models
    models = get_all_models()
    assert 'prostate' in models
    assert 'bladder' in models
    assert all(hasattr(m, 'name') for m in models.values())
```

Also remove the `from uroseg.models import ModelDef` import at the top of test_utils.py.

- [ ] **Step 8: Update `tests/test_inference.py`**

Replace the two model-import lines at top:
```python
# OLD:
import uroseg.resources.models.prostate as prostate_mod
import uroseg.resources.models.bladder as bladder_mod
# NEW:
import uroseg.models.prostate as prostate_mod
import uroseg.models.bladder as bladder_mod
```

Replace `test_load_model_module_*` tests (lines 44–54):
```python
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

- [ ] **Step 9: Rewrite `tests/test_install.py`**

```python
# tests/test_install.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_extract_release_id():
    from uroseg.models.base import _extract_release_id
    url = 'https://github.com/yw7/uroseg/releases/download/r20260101/Dataset001_Prostate_r20260101.zip'
    assert _extract_release_id(url) == 'r20260101'


def test_find_model_dir_returns_newest(tmp_path):
    from uroseg.models.base import _find_model_dir
    (tmp_path / 'prostate' / 'r20260101').mkdir(parents=True)
    (tmp_path / 'prostate' / 'r20260601').mkdir(parents=True)
    result = _find_model_dir('prostate', tmp_path)
    assert result.name == 'r20260601'


def test_find_model_dir_not_found(tmp_path):
    from uroseg.models.base import _find_model_dir
    with pytest.raises(FileNotFoundError):
        _find_model_dir('prostate', tmp_path)


def test_install_skips_no_url(tmp_path, capsys):
    from uroseg.models.base import SegModel
    class M(SegModel):
        name = 'test'; description = ''; weights_url = ''; labels = {}
    M().install(tmp_path)
    out = capsys.readouterr().out
    assert 'skip' in out.lower() or 'no weights' in out.lower()


def test_install_skips_if_already_installed(tmp_path, capsys):
    from uroseg.models.prostate import Prostate
    release_id = 'r20260101'
    (tmp_path / 'prostate' / release_id).mkdir(parents=True)
    Prostate().install(tmp_path)
    assert 'already installed' in capsys.readouterr().out.lower()


def test_install_downloads_to_temp_and_extracts(tmp_path):
    from uroseg.models.prostate import Prostate
    with patch('uroseg.models.base._download_zip') as mock_dl, \
         patch('uroseg.models.base._extract_zip') as mock_ex:
        mock_dl.return_value = tmp_path / 'x.zip'
        Prostate().install(tmp_path)
        assert mock_dl.called
        assert mock_ex.called
        # dest must be data_dir/prostate/<release_id>/
        dest = mock_ex.call_args[0][1]
        assert dest.parent.name == 'prostate'
```

- [ ] **Step 10: Run tests**

```bash
pytest tests/test_models.py tests/test_utils.py tests/test_inference.py tests/test_install.py -v
```
Expected: all PASS

- [ ] **Step 11: Run full suite to check no regressions**

```bash
pytest -x --ignore=tests/test_train.py
```
Expected: PASS (train tests reference old train_nnunet imports, fixed in Task 4)

- [ ] **Step 12: Commit**

```bash
git add uroseg/models/ tests/test_models.py tests/test_utils.py tests/test_inference.py tests/test_install.py uroseg/utils/utils.py
git commit -m "feat: concrete Prostate/Bladder model classes + registry; remove ModelDef dataclass"
```

---

### Task 3: `uroseg/nnunet/helpers.py`

**Files:**
- Create: `uroseg/nnunet/__init__.py`
- Create: `uroseg/nnunet/helpers.py`
- Modify: `tests/test_train.py` (update imports only)

**Interfaces:**
- Consumes: `SegModel` from Task 1 (type hint only, lazy import)
- Produces:
  - `setup_env(data_dir: Path) -> None`
  - `extract_dataset_id(nnunet_task: str) -> int`
  - `generate_dataset_json(model, images_tr: Path) -> dict`
  - `run_predict(model_dir: Path, inputs: list[Path], output_dir: Path, fold: int = 0, device: str = "cuda") -> None`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_train.py` — replace existing imports at top:
```python
# Replace:
from uroseg.commands.train_nnunet import extract_dataset_id, generate_dataset_json, setup_nnunet_env
# With (these will be added as imports within each test function — see below)
```

Add these tests to `tests/test_train.py`:
```python
def test_helpers_setup_env_sets_vars(tmp_path):
    from uroseg.nnunet.helpers import setup_env
    import os
    setup_env(tmp_path)
    assert os.environ["nnUNet_raw"] == str(tmp_path / "nnUNet" / "raw")
    assert os.environ["nnUNet_preprocessed"] == str(tmp_path / "nnUNet" / "preprocessed")
    assert os.environ["nnUNet_results"] == str(tmp_path / "nnUNet" / "results")
    assert os.environ["nnUNet_exports"] == str(tmp_path / "nnUNet" / "exports")


def test_helpers_extract_dataset_id():
    from uroseg.nnunet.helpers import extract_dataset_id
    assert extract_dataset_id("Dataset101_Prostate") == 101
    assert extract_dataset_id("Dataset010_Bladder") == 10


def test_helpers_extract_dataset_id_bad_format():
    from uroseg.nnunet.helpers import extract_dataset_id
    with pytest.raises(ValueError):
        extract_dataset_id("BadName")
```

- [ ] **Step 2: Run to confirm fail**

```bash
pytest tests/test_train.py::test_helpers_setup_env_sets_vars -v
```
Expected: FAIL (ImportError)

- [ ] **Step 3: Create `uroseg/nnunet/__init__.py`**

```python
# uroseg/nnunet/__init__.py
```

- [ ] **Step 4: Create `uroseg/nnunet/helpers.py`**

```python
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uroseg.models.base import SegModel


def setup_env(data_dir: Path) -> None:
    nnunet_dir = data_dir / "nnUNet"
    os.environ["nnUNet_raw"] = str(nnunet_dir / "raw")
    os.environ["nnUNet_preprocessed"] = str(nnunet_dir / "preprocessed")
    os.environ["nnUNet_results"] = str(nnunet_dir / "results")
    os.environ["nnUNet_exports"] = str(nnunet_dir / "exports")


def extract_dataset_id(nnunet_task: str) -> int:
    match = re.match(r'Dataset(\d+)_', nnunet_task)
    if not match:
        raise ValueError(f"Invalid nnunet_task format: {nnunet_task!r} — expected 'DatasetNNN_Name'")
    return int(match.group(1))


def generate_dataset_json(model: SegModel, images_tr: Path) -> dict:
    from uroseg.utils.utils import normalize_labels
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
        "numTraining": len(list(images_tr.glob("*.nii.gz"))),
        "file_ending": ".nii.gz",
    }
    if has_regions:
        dataset["regions_class_order"] = sorted(all_values)
    return dataset


def run_predict(
    model_dir: Path,
    inputs: list[Path],
    output_dir: Path,
    fold: int = 0,
    device: str = 'cuda',
) -> None:
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    output_dir.mkdir(parents=True, exist_ok=True)

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    elif device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    device_obj = torch.device(device)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device_obj,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    # Walk into trainer__plans__config/fold_N structure
    fold_dir = model_dir
    for child in model_dir.iterdir():
        if child.is_dir():
            candidate = child / f'fold_{fold}'
            if candidate.exists():
                fold_dir = child
                break

    predictor.initialize_from_trained_model_folder(
        str(fold_dir),
        use_folds=(fold,),
        checkpoint_name='checkpoint_best.pth',
    )

    list_of_lists = [[str(f)] for f in inputs]
    output_names = [str(output_dir / f.name) for f in inputs]
    predictor.predict_from_files(
        list_of_lists,
        output_names,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_train.py::test_helpers_setup_env_sets_vars tests/test_train.py::test_helpers_extract_dataset_id tests/test_train.py::test_helpers_extract_dataset_id_bad_format -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add uroseg/nnunet/__init__.py uroseg/nnunet/helpers.py tests/test_train.py
git commit -m "feat: uroseg/nnunet/helpers.py with setup_env, extract_dataset_id, generate_dataset_json, run_predict"
```

---

### Task 4: `uroseg/nnunet/train.py`

**Files:**
- Create: `uroseg/nnunet/train.py`
- Delete: `uroseg/commands/train_nnunet.py`
- Delete: `uroseg/commands/train.py`
- Modify: `tests/test_train.py`

**Interfaces:**
- Consumes: `setup_env`, `extract_dataset_id`, `generate_dataset_json` from `uroseg.nnunet.helpers`
- Produces: `main()` — CLI entry point called by `uroseg train nnunet`

- [ ] **Step 1: Write failing test**

Add to `tests/test_train.py`:
```python
def test_train_nnunet_main_accepts_training_dir(tmp_path):
    """--training-dir/-d sets the raw data location; separate from --data-dir."""
    from uroseg.nnunet.train import main
    from uroseg.models.base import NNUNetSegModel
    from unittest.mock import patch, MagicMock
    import sys

    class FakeModel(NNUNetSegModel):
        name = 'bladder'; description = ''; weights_url = ''; labels = {'background': 0, 'bladder': 1}
        nnunet_task = 'Dataset010_Bladder'

    training_dir = tmp_path / 'train'
    raw_dir = training_dir / 'nnUNet' / 'raw' / 'Dataset010_Bladder'
    (raw_dir / 'imagesTr').mkdir(parents=True)
    import nibabel as nib, numpy as np
    nib.save(nib.Nifti1Image(np.zeros((5,5,5), dtype=np.int16), np.eye(4)),
             raw_dir / 'imagesTr' / 'case_000_0000.nii.gz')

    with patch('uroseg.nnunet.train.subprocess.run'), \
         patch('auglab.add_trainer.add_trainer'), \
         patch('uroseg.utils.utils.load_model_module') as mock_load:
        mock_load.return_value = MagicMock(MODEL=FakeModel(), NNUNET_TASK='Dataset010_Bladder')
        with patch.object(sys, 'argv', ['uroseg', 'bladder', '-d', str(training_dir)]):
            main()


def test_train_nnunet_default_training_dir_is_cwd(tmp_path, monkeypatch):
    """Default --training-dir is current working directory."""
    import os
    from uroseg.nnunet.train import main
    from uroseg.models.base import NNUNetSegModel
    from unittest.mock import patch, MagicMock
    import sys

    class FakeModel(NNUNetSegModel):
        name = 'bladder'; description = ''; weights_url = ''; labels = {'background': 0, 'bladder': 1}
        nnunet_task = 'Dataset010_Bladder'

    monkeypatch.chdir(tmp_path)
    raw_dir = tmp_path / 'nnUNet' / 'raw' / 'Dataset010_Bladder'
    (raw_dir / 'imagesTr').mkdir(parents=True)
    import nibabel as nib, numpy as np
    nib.save(nib.Nifti1Image(np.zeros((5,5,5), dtype=np.int16), np.eye(4)),
             raw_dir / 'imagesTr' / 'case_000_0000.nii.gz')

    with patch('uroseg.nnunet.train.subprocess.run'), \
         patch('auglab.add_trainer.add_trainer'), \
         patch('uroseg.utils.utils.load_model_module') as mock_load:
        mock_load.return_value = MagicMock(MODEL=FakeModel(), NNUNET_TASK='Dataset010_Bladder')
        with patch.object(sys, 'argv', ['uroseg', 'bladder']):
            main()
```

- [ ] **Step 2: Run to confirm fail**

```bash
pytest tests/test_train.py::test_train_nnunet_main_accepts_training_dir -v
```
Expected: FAIL (ImportError — `uroseg.nnunet.train` doesn't exist)

- [ ] **Step 3: Create `uroseg/nnunet/train.py`**

Port from `uroseg/commands/train_nnunet.py` with these changes:
1. Imports come from `uroseg.nnunet.helpers` instead of local
2. `--data-dir` is for the nnUNet env setup (preprocessed, results, exports); `--training-dir`/`-d` (default `./`) is for raw training data
3. `_add_trainer("nnUNetTrainerDAExtGPU")` (was `"nnUNetTrainerDAExt"`)

```python
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import uroseg.utils.utils as _utils
from uroseg.utils.utils import data_dir_help
from uroseg.nnunet.helpers import setup_env, extract_dataset_id, generate_dataset_json


def _count_files(directory: Path, pattern: str) -> int:
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='uroseg train nnunet',
        description="Train a UroSeg model with nnU-Net and AugLab augmentation.",
    )
    parser.add_argument("organ", help="Organ name matching uroseg/models/<organ>.py")
    parser.add_argument("--fold", "-f", type=int, default=0,
                        help="nnU-Net fold number (0–4, default: 0)")
    parser.add_argument("--auglab-config", default=None,
                        help="Path to AugLab GPU augmentation config JSON (optional)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"],
                        help="Training device (default: cuda)")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs (default: 1)")
    parser.add_argument("--training-dir", "-d", default=None,
                        help="Directory containing training data (default: current dir). "
                             "Raw data expected at <training-dir>/nnUNet/raw/<task>/")
    parser.add_argument("--data-dir", default=None, help=data_dir_help())
    args = parser.parse_args()

    mod = _utils.load_model_module(args.organ)
    training_path = Path(args.training_dir).resolve() if args.training_dir else Path.cwd()
    setup_env(training_path)

    nnunet_task = mod.NNUNET_TASK
    dataset_id = extract_dataset_id(nnunet_task)

    raw_dir = training_path / "nnUNet" / "raw" / nnunet_task
    images_tr = raw_dir / "imagesTr"
    labels_tr = raw_dir / "labelsTr"

    if not images_tr.exists():
        print(
            f"Error: training images directory not found: {images_tr}\n"
            f"Place training images in {images_tr}/ (filename pattern: <case>_0000.nii.gz)",
            file=sys.stderr,
        )
        sys.exit(1)

    dataset_json = generate_dataset_json(mod.MODEL, images_tr)
    (raw_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    print(f"Generated {raw_dir / 'dataset.json'}")

    preprocessed_dir = training_path / "nnUNet" / "preprocessed" / nnunet_task

    if not (preprocessed_dir / "dataset_fingerprint.json").exists():
        print("Extracting dataset fingerprint...")
        subprocess.run(["nnUNetv2_extract_fingerprint", "-d", str(dataset_id)], check=True)

    if not (preprocessed_dir / "nnUNetPlans.json").exists():
        print("Planning experiment...")
        subprocess.run(["nnUNetv2_plan_experiment", "-d", str(dataset_id)], check=True)

    plans_file = preprocessed_dir / "nnUNetPlans.json"
    data_identifier = "nnUNetPlans_3d_fullres"
    if plans_file.exists():
        plans = json.loads(plans_file.read_text())
        identifier = plans.get("configurations", {}).get("3d_fullres", {}).get("data_identifier")
        if identifier:
            data_identifier = identifier
    preprocessed_data_dir = preprocessed_dir / data_identifier

    n_labels = _count_files(labels_tr, "*.nii.gz")
    n_pkl = _count_files(preprocessed_data_dir, "*.pkl")
    if not preprocessed_data_dir.exists() or n_pkl != n_labels:
        reason = (f"{preprocessed_data_dir.name} not found"
                  if not preprocessed_data_dir.exists()
                  else f"{n_pkl} .pkl vs {n_labels} labels in labelsTr")
        print(f"Preprocessing dataset (3d_fullres) [{reason}]...")
        subprocess.run(
            ["nnUNetv2_preprocess", "-d", str(dataset_id), "-c", "3d_fullres"],
            check=True,
        )
    else:
        print(f"Preprocessing already done ({n_pkl} samples), skipping.")

    from auglab.add_trainer import add_trainer as _add_trainer
    _add_trainer("nnUNetTrainerDAExtGPU")

    if args.auglab_config:
        os.environ["AUGLAB_PARAMS_GPU_JSON"] = str(args.auglab_config)
    elif "AUGLAB_PARAMS_GPU_JSON" not in os.environ:
        from importlib.resources import files as _res_files
        import uroseg.resources.auglab as _auglab_res
        bundled = _res_files(_auglab_res) / "transform_params_gpu.json"
        os.environ["AUGLAB_PARAMS_GPU_JSON"] = str(bundled)

    results_dir = training_path / "nnUNet" / "results"
    exports_dir = training_path / "nnUNet" / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    trainer_tag = "nnUNetTrainerDAExtGPU__nnUNetPlans__3d_fullres"

    n_npy = _count_files(preprocessed_data_dir, "*.npy")
    n_npz = _count_files(preprocessed_data_dir, "*.npz")
    train_cmd = [
        "nnUNetv2_train",
        str(dataset_id), "3d_fullres", str(args.fold),
        "-tr", "nnUNetTrainerDAExtGPU", "--c", "-device", args.device,
    ]
    if n_npy == 2 * n_npz and n_npz > 0:
        train_cmd.append("--use_compressed")
    print(f"Starting training (dataset {dataset_id}, fold {args.fold})...")
    subprocess.run(train_cmd, check=True)

    zip_name = f"{nnunet_task}__{trainer_tag}__fold_{args.fold}.zip"
    zip_path = exports_dir / zip_name
    (results_dir / nnunet_task / "ensembles").mkdir(parents=True, exist_ok=True)
    print(f"Exporting model to {zip_path}...")
    subprocess.run([
        "nnUNetv2_export_model_to_zip",
        "-d", str(dataset_id), "-o", str(zip_path),
        "-c", "3d_fullres", "-f", str(args.fold),
        "-tr", "nnUNetTrainerDAExtGPU",
    ], check=True)

    images_ts = raw_dir / "imagesTs"
    labels_ts = raw_dir / "labelsTs"
    test_pred_dir = results_dir / nnunet_task / trainer_tag / f"fold_{args.fold}" / "test"

    if images_ts.exists() and _count_files(images_ts, "*.nii.gz") > 0:
        test_pred_dir.mkdir(parents=True, exist_ok=True)
        print("Predicting on test set...")
        subprocess.run([
            "nnUNetv2_predict",
            "-d", str(dataset_id), "-i", str(images_ts),
            "-o", str(test_pred_dir), "-f", str(args.fold),
            "-c", "3d_fullres", "-tr", "nnUNetTrainerDAExtGPU",
        ], check=True)

        if labels_ts.exists() and _count_files(labels_ts, "*.nii.gz") > 0:
            trainer_results_dir = results_dir / nnunet_task / trainer_tag
            print("Evaluating test predictions...")
            subprocess.run([
                "nnUNetv2_evaluate_folder",
                str(labels_ts), str(test_pred_dir),
                "-djfile", str(trainer_results_dir / "dataset.json"),
                "-pfile", str(trainer_results_dir / "plans.json"),
            ], check=True)
            summary_json = test_pred_dir / "summary.json"
            if summary_json.exists():
                with zipfile.ZipFile(zip_path, 'a') as zf:
                    zf.write(summary_json, arcname=str(summary_json.relative_to(results_dir)))

    dataset_listing = results_dir / nnunet_task / "dataset_files.json"
    (results_dir / nnunet_task).mkdir(parents=True, exist_ok=True)
    listing: dict[str, list[str]] = {}
    for item in sorted(raw_dir.iterdir()):
        if item.is_dir():
            listing[item.name] = sorted(f.name for f in item.iterdir())
    dataset_listing.write_text(json.dumps(listing, indent=2))
    with zipfile.ZipFile(zip_path, 'a') as zf:
        zf.write(dataset_listing, arcname=str(dataset_listing.relative_to(results_dir)))

    splits_json = preprocessed_dir / "splits_final.json"
    if splits_json.exists():
        with zipfile.ZipFile(zip_path, 'a') as zf:
            zf.write(splits_json, arcname=str(splits_json.relative_to(preprocessed_dir.parent)))

    print(f"\nTraining complete.")
    print(f"  Model: {results_dir / nnunet_task / trainer_tag / f'fold_{args.fold}'}")
    print(f"  Export: {zip_path}")
```

- [ ] **Step 4: Delete old train files**

```bash
rm uroseg/commands/train_nnunet.py uroseg/commands/train.py
```

- [ ] **Step 5: Update existing train tests to import from new path**

In `tests/test_train.py`, update all imports from `uroseg.commands.train_nnunet` to `uroseg.nnunet.train` and from `uroseg.commands.train_nnunet.setup_nnunet_env` to `uroseg.nnunet.helpers.setup_env`. Specifically:

```python
# Replace in test_setup_nnunet_env_sets_vars:
from uroseg.commands.train_nnunet import setup_nnunet_env
setup_nnunet_env(tmp_path)
# With:
from uroseg.nnunet.helpers import setup_env
setup_env(tmp_path)

# Replace in test_extract_dataset_id_*:
from uroseg.commands.train_nnunet import extract_dataset_id
# With:
from uroseg.nnunet.helpers import extract_dataset_id

# Replace in test_generate_dataset_json_*:
from uroseg.commands.train_nnunet import generate_dataset_json
# With:
from uroseg.nnunet.helpers import generate_dataset_json

# Replace in test_train_generates_dataset_json, test_train_calls_nnunet_train, etc.:
from uroseg.commands.train_nnunet import main
# With:
from uroseg.nnunet.train import main

# In test_train_cli_help, update argv:
[sys.executable, '-m', 'uroseg.cli', 'train', 'nnunet', '--help']
# (unchanged — CLI interface stays the same)
```

Also update `test_train_calls_nnunet_train` mock target:
```python
patch("uroseg.commands.train_nnunet.subprocess.run")
# becomes:
patch("uroseg.nnunet.train.subprocess.run")
```

And update model loading mock paths similarly (change `uroseg.commands.train_nnunet` → `uroseg.nnunet.train`).

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_train.py -v
```
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add uroseg/nnunet/train.py tests/test_train.py
git commit -m "feat: uroseg/nnunet/train.py with --training-dir/-d; fix _add_trainer to DAExtGPU"
```

---

### Task 5: `uroseg/nnunet/predict.py`

**Files:**
- Create: `uroseg/nnunet/predict.py`
- Delete: `uroseg/commands/predict_nnunet.py`
- Delete: `uroseg/utils/inference_utils.py`
- Modify: `tests/test_inference.py`

**Interfaces:**
- Consumes: `run_predict` from `uroseg.nnunet.helpers`; `SegModel` from `uroseg.models.base`
- Produces:
  - `add_inference_args(parser: ArgumentParser) -> None` — adds `--img/-i`, `--out/-o`, `--fold/-f`, `--device/-d`, `--out-suffix`, `--out-prefix`, `--data-dir` (was `add_common_inference_args`)
  - `run_predict_cli(model: SegModel, args) -> None` — full inference workflow (was `run_nnunet_predict`)
  - `main() -> None` — low-level CLI for `uroseg predict_nnunet`

- [ ] **Step 1: Write failing tests**

Replace `tests/test_inference.py` contents:
```python
from __future__ import annotations
import argparse
import pytest
from unittest.mock import MagicMock, patch
import uroseg.models.prostate as prostate_mod
import uroseg.models.bladder as bladder_mod


def test_add_inference_args_required():
    from uroseg.nnunet.predict import add_inference_args
    parser = argparse.ArgumentParser()
    add_inference_args(parser)
    args = parser.parse_args(['-i', 'img.nii.gz', '-o', 'out/'])
    assert args.img == 'img.nii.gz'
    assert args.out == 'out/'


def test_add_inference_args_defaults():
    from uroseg.nnunet.predict import add_inference_args
    parser = argparse.ArgumentParser()
    add_inference_args(parser)
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
        from uroseg.nnunet.predict import add_inference_args
        add_inference_args(parser)
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

- [ ] **Step 2: Run to confirm fail**

```bash
pytest tests/test_inference.py::test_add_inference_args_required -v
```
Expected: FAIL (ImportError)

- [ ] **Step 3: Create `uroseg/nnunet/predict.py`**

```python
from __future__ import annotations
import argparse
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uroseg.models.base import SegModel

from uroseg.utils.utils import (
    add_common_args, collect_niftis, build_output_path,
    resolve_data_path, data_dir_help,
)
from uroseg.utils.image import Image


def add_inference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--img', '-i', required=True, help='Input image file or folder')
    parser.add_argument('--out', '-o', required=True, help='Output folder')
    parser.add_argument('--fold', '-f', type=int, default=0, help='nnU-Net fold (default: 0)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--out-suffix', default='_seg', help='Output filename suffix')
    parser.add_argument('--out-prefix', default='', help='Output filename prefix')
    parser.add_argument('--data-dir', default=None, help=data_dir_help())
    add_common_args(parser)


def run_predict_cli(model: SegModel, args) -> None:
    from uroseg.nnunet.helpers import run_predict
    from uroseg.models.base import _find_model_dir

    data_path = resolve_data_path(args.data_dir)
    model_dir = _find_model_dir(model.name, data_path)

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

        run_predict(
            model_dir=model_dir,
            inputs=reoriented,
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


def main() -> None:
    """Low-level nnU-Net prediction wrapper (uroseg predict_nnunet)."""
    from uroseg.models.base import _find_model_dir
    from uroseg.nnunet.helpers import run_predict

    parser = argparse.ArgumentParser(description='Low-level nnU-Net prediction wrapper.')
    parser.add_argument('--img', required=True, help='Input image file or folder')
    parser.add_argument('--out', required=True, help='Output folder')
    parser.add_argument('--task', required=True, help='nnUNet task name (e.g. Dataset001_Prostate)')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--data-dir', default=None, help=data_dir_help())
    add_common_args(parser)
    args = parser.parse_args()

    # For low-level use, derive model name from task name (DatasetNNN_Name -> name lower)
    data_path = resolve_data_path(args.data_dir)
    task_name = args.task.split('_', 1)[1].lower() if '_' in args.task else args.task.lower()
    try:
        model_dir = _find_model_dir(task_name, data_path)
    except FileNotFoundError:
        # Fallback: old nnUNet/results layout
        from uroseg.nnunet.helpers import extract_dataset_id
        results_root = data_path / 'nnUNet' / 'results'
        direct = results_root / args.task
        if direct.exists():
            model_dir = direct
        else:
            raise
    inputs = collect_niftis(args.img)
    run_predict(model_dir, inputs, Path(args.out), fold=args.fold, device=args.device)
```

- [ ] **Step 4: Delete old files**

```bash
rm uroseg/commands/predict_nnunet.py uroseg/utils/inference_utils.py
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_inference.py -v
```
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add uroseg/nnunet/predict.py tests/test_inference.py
git commit -m "feat: uroseg/nnunet/predict.py; merge predict_nnunet + inference_utils"
```

---

### Task 6: Move tools to `uroseg/tools/` + add public API

**Files:**
- Create: `uroseg/tools/__init__.py`
- Create: `uroseg/tools/map_labels.py` (from `commands/map_labels.py` + public API)
- Create: `uroseg/tools/resample.py` (from `commands/resample.py` + public API)
- Create: `uroseg/tools/crop.py` (from `commands/crop_image2seg.py` + public API)
- Create: `uroseg/tools/largest_component.py` (from `commands/largest_component.py` + public API)
- Create: `uroseg/tools/reorient.py` (from `commands/reorient_canonical.py` + public API)
- Create: `uroseg/tools/transform_seg2image.py` (from `commands/transform_seg2image.py` + public API)
- Create: `uroseg/tools/preview.py` (from `commands/preview_jpg.py` + public API)
- Create: `uroseg/tools/cpdir.py` (from `commands/cpdir.py` + public API)
- Delete: all `uroseg/commands/<tool>.py` files listed above
- Modify: `tests/test_commands.py` (update import paths)

**Interfaces:**
- Consumes: nothing new (same utils as before)
- Produces public functions for each tool (signatures below)

Public function signatures (for all tools, the internal `process_one` function is unchanged):

```python
# map_labels.py
def map_labels(input: Path | str, output: Path | str, map: dict,
               keep_unmapped: bool = False,
               update_seg: Path | str | None = None,
               update_from_seg: Path | str | None = None,
               out_suffix: str = "_mapped", out_prefix: str = "",
               overwrite: bool = False) -> Path: ...

def map_labels_dir(input_dir: Path | str, output_dir: Path | str, map: dict,
                   keep_unmapped: bool = False,
                   update_seg: Path | str | None = None,
                   update_from_seg: Path | str | None = None,
                   out_suffix: str = "_mapped", out_prefix: str = "",
                   overwrite: bool = False, n_jobs: int = 1) -> None: ...

# resample.py
def resample(input: Path | str, output: Path | str,
             mm: float | list[float] = 1.0,
             out_suffix: str = "_resampled", out_prefix: str = "",
             overwrite: bool = False) -> Path: ...

def resample_dir(input_dir: Path | str, output_dir: Path | str,
                 mm: float | list[float] = 1.0,
                 out_suffix: str = "_resampled", out_prefix: str = "",
                 overwrite: bool = False, n_jobs: int = 1) -> None: ...

# crop.py
def crop(input: Path | str, seg: Path | str, output: Path | str,
         margin: int = 0,
         out_suffix: str = "_crop", out_prefix: str = "",
         overwrite: bool = False) -> Path: ...

def crop_dir(input_dir: Path | str, seg_dir: Path | str, output_dir: Path | str,
             margin: int = 0,
             out_suffix: str = "_crop", out_prefix: str = "",
             overwrite: bool = False, n_jobs: int = 1) -> None: ...

# largest_component.py
def largest_component(input: Path | str, output: Path | str,
                      labels: list[int] | None = None,
                      dilate: int = 0, binarize: bool = False,
                      out_suffix: str = "_largest", out_prefix: str = "",
                      overwrite: bool = False) -> Path: ...

def largest_component_dir(input_dir: Path | str, output_dir: Path | str,
                          labels: list[int] | None = None,
                          dilate: int = 0, binarize: bool = False,
                          out_suffix: str = "_largest", out_prefix: str = "",
                          overwrite: bool = False, n_jobs: int = 1) -> None: ...

# reorient.py
def reorient(input: Path | str, output: Path | str,
             out_suffix: str = "_reoriented", out_prefix: str = "",
             overwrite: bool = False) -> Path: ...

def reorient_dir(input_dir: Path | str, output_dir: Path | str,
                 out_suffix: str = "_reoriented", out_prefix: str = "",
                 overwrite: bool = False, n_jobs: int = 1) -> None: ...

# transform_seg2image.py
def transform_seg2image(seg: Path | str, img: Path | str, output: Path | str,
                        interpolation: str = 'nearest',
                        seg_suffix: str = "_transformed", seg_prefix: str = "",
                        overwrite: bool = False) -> Path: ...

def transform_seg2image_dir(seg_dir: Path | str, img_dir: Path | str, output_dir: Path | str,
                            interpolation: str = 'nearest',
                            seg_suffix: str = "_transformed", seg_prefix: str = "",
                            overwrite: bool = False, n_jobs: int = 1) -> None: ...

# preview.py
def preview(input: Path | str, output: Path | str,
            seg: Path | str | None = None,
            orient: str = 'sag', sliceloc: float = 0.5,
            label_text_right: dict[int, str] | None = None,
            label_text_left: dict[int, str] | None = None,
            out_suffix: str = "_preview", out_prefix: str = "",
            overwrite: bool = False) -> Path: ...

def preview_dir(input_dir: Path | str, output_dir: Path | str,
                seg_dir: Path | str | None = None,
                orient: str = 'sag', sliceloc: float = 0.5,
                label_text_right: dict[int, str] | None = None,
                label_text_left: dict[int, str] | None = None,
                out_suffix: str = "_preview", out_prefix: str = "",
                overwrite: bool = False, n_jobs: int = 1) -> None: ...

# cpdir.py
def cpdir(input: Path | str, output: Path | str,
          out_suffix: str = "", out_prefix: str = "",
          overwrite: bool = False) -> Path: ...

def cpdir_dir(input_dir: Path | str, output_dir: Path | str,
              out_suffix: str = "", out_prefix: str = "",
              overwrite: bool = False, n_jobs: int = 1) -> None: ...
```

- [ ] **Step 1: Write failing tests**

Add to `tests/test_commands.py`:
```python
# ── public API functions ──────────────────────────────────────────────────────

def test_map_labels_public_api(seg_file, tmp_path):
    from uroseg.tools.map_labels import map_labels
    out = tmp_path / 'out.nii.gz'
    result = map_labels(seg_file, out, map={1: 10, 2: 20})
    assert result == out
    assert out.exists()
    img = Image.load(out)
    assert img.data[3, 3, 3] == 10


def test_resample_public_api(img_file, tmp_path):
    from uroseg.tools.resample import resample
    out = tmp_path / 'out.nii.gz'
    result = resample(img_file, out, mm=2.0)
    assert result == out
    assert out.exists()


def test_reorient_public_api(img_file, tmp_path):
    from uroseg.tools.reorient import reorient
    out = tmp_path / 'out.nii.gz'
    result = reorient(img_file, out)
    assert result == out
    assert out.exists()


def test_largest_component_public_api(seg_file, tmp_path):
    from uroseg.tools.largest_component import largest_component
    out = tmp_path / 'out.nii.gz'
    result = largest_component(seg_file, out)
    assert result == out
    assert out.exists()


def test_map_labels_dir_public_api(tmp_path):
    from uroseg.tools.map_labels import map_labels_dir
    import nibabel as nib
    in_dir = tmp_path / 'in'
    in_dir.mkdir()
    out_dir = tmp_path / 'out'
    data = np.zeros((5,5,5), dtype=np.int16)
    data[1,1,1] = 1
    for i in range(2):
        nib.save(nib.Nifti1Image(data, np.eye(4)), in_dir / f'seg{i}.nii.gz')
    map_labels_dir(in_dir, out_dir, map={1: 99}, out_suffix='_mapped', n_jobs=1)
    assert len(list(out_dir.glob('*.nii.gz'))) == 2
```

- [ ] **Step 2: Run to confirm fail**

```bash
pytest tests/test_commands.py::test_map_labels_public_api -v
```
Expected: FAIL (ImportError from `uroseg.tools`)

- [ ] **Step 3: Create `uroseg/tools/__init__.py`**

```python
# uroseg/tools/__init__.py
```

- [ ] **Step 4: Create `uroseg/tools/map_labels.py`**

Copy `uroseg/commands/map_labels.py` exactly, then add at the end:

```python
def map_labels(
    input: Path | str,
    output: Path | str,
    map: dict,
    keep_unmapped: bool = False,
    update_seg: Path | str | None = None,
    update_from_seg: Path | str | None = None,
    out_suffix: str = "_mapped",
    out_prefix: str = "",
    overwrite: bool = False,
) -> Path:
    import argparse
    input_path, output_path = Path(input), Path(output)
    if not output_path.suffix:
        from uroseg.utils.utils import build_output_path
        output_path = build_output_path(input_path, output_path, out_prefix, out_suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    int_map = {int(k): int(v) for k, v in map.items()}
    process_one(
        (input_path, output_path),
        int_map,
        keep_unmapped,
        str(update_seg) if update_seg else None,
        str(update_from_seg) if update_from_seg else None,
        argparse.Namespace(quiet=True, overwrite=overwrite),
    )
    return output_path


def map_labels_dir(
    input_dir: Path | str,
    output_dir: Path | str,
    map: dict,
    keep_unmapped: bool = False,
    update_seg: Path | str | None = None,
    update_from_seg: Path | str | None = None,
    out_suffix: str = "_mapped",
    out_prefix: str = "",
    overwrite: bool = False,
    n_jobs: int = 1,
) -> None:
    import argparse, functools
    from uroseg.utils.utils import build_pairs
    int_map = {int(k): int(v) for k, v in map.items()}
    pairs = build_pairs(input_dir, output_dir, out_suffix, out_prefix, overwrite)
    process_map(
        functools.partial(
            process_one,
            mapping=int_map,
            keep_unmapped=keep_unmapped,
            update_seg=str(update_seg) if update_seg else None,
            update_from_seg=str(update_from_seg) if update_from_seg else None,
            args=argparse.Namespace(quiet=False, overwrite=overwrite),
        ),
        pairs,
        max_workers=n_jobs,
        disable=False,
        desc='uroseg map',
    )
```

- [ ] **Step 5: Create `uroseg/tools/resample.py`**

Copy `uroseg/commands/resample.py` exactly, then add at the end:

```python
def resample(
    input: Path | str,
    output: Path | str,
    mm: float | list[float] = 1.0,
    out_suffix: str = "_resampled",
    out_prefix: str = "",
    overwrite: bool = False,
) -> Path:
    import argparse
    input_path, output_path = Path(input), Path(output)
    if not output_path.suffix:
        from uroseg.utils.utils import build_output_path
        output_path = build_output_path(input_path, output_path, out_prefix, out_suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mm_list = [mm] if isinstance(mm, float) else list(mm)
    process_one(
        (input_path, output_path),
        argparse.Namespace(mm=mm_list, overwrite=overwrite),
    )
    return output_path


def resample_dir(
    input_dir: Path | str,
    output_dir: Path | str,
    mm: float | list[float] = 1.0,
    out_suffix: str = "_resampled",
    out_prefix: str = "",
    overwrite: bool = False,
    n_jobs: int = 1,
) -> None:
    import argparse, functools
    from uroseg.utils.utils import build_pairs
    mm_list = [mm] if isinstance(mm, float) else list(mm)
    pairs = build_pairs(input_dir, output_dir, out_suffix, out_prefix, overwrite)
    process_map(
        functools.partial(process_one, args=argparse.Namespace(mm=mm_list, overwrite=overwrite)),
        pairs,
        max_workers=n_jobs,
        disable=False,
        desc='uroseg resample',
    )
```

- [ ] **Step 6: Create `uroseg/tools/crop.py`**

Copy `uroseg/commands/crop_image2seg.py` exactly, then add at the end:

```python
def crop(
    input: Path | str,
    seg: Path | str,
    output: Path | str,
    margin: int = 0,
    out_suffix: str = "_crop",
    out_prefix: str = "",
    overwrite: bool = False,
) -> Path:
    import argparse
    input_path, seg_path, output_path = Path(input), Path(seg), Path(output)
    if not output_path.suffix:
        from uroseg.utils.utils import build_output_path
        output_path = build_output_path(input_path, output_path, out_prefix, out_suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    process_one(
        (input_path, seg_path, output_path),
        argparse.Namespace(margin=margin, overwrite=overwrite),
    )
    return output_path


def crop_dir(
    input_dir: Path | str,
    seg_dir: Path | str,
    output_dir: Path | str,
    margin: int = 0,
    out_suffix: str = "_crop",
    out_prefix: str = "",
    overwrite: bool = False,
    n_jobs: int = 1,
) -> None:
    import argparse, functools, sys
    from uroseg.utils.utils import collect_niftis, build_output_path
    imgs = collect_niftis(input_dir)
    segs = collect_niftis(seg_dir)
    if len(imgs) != len(segs):
        print(f"Mismatch: {len(imgs)} images vs {len(segs)} segs.", file=sys.stderr)
        return
    out = Path(output_dir)
    args = argparse.Namespace(margin=margin, overwrite=overwrite)
    triples = [
        (i, s, build_output_path(i, out, out_prefix, out_suffix))
        for i, s in zip(imgs, segs)
        if overwrite or not build_output_path(i, out, out_prefix, out_suffix).exists()
    ]
    process_map(
        functools.partial(process_one, args=args),
        triples,
        max_workers=n_jobs,
        disable=False,
        desc='uroseg crop',
    )
```

- [ ] **Step 7: Create `uroseg/tools/largest_component.py`**

Copy `uroseg/commands/largest_component.py` exactly, then add at the end:

```python
def largest_component(
    input: Path | str,
    output: Path | str,
    labels: list[int] | None = None,
    dilate: int = 0,
    binarize: bool = False,
    out_suffix: str = "_largest",
    out_prefix: str = "",
    overwrite: bool = False,
) -> Path:
    import argparse
    input_path, output_path = Path(input), Path(output)
    if not output_path.suffix:
        from uroseg.utils.utils import build_output_path
        output_path = build_output_path(input_path, output_path, out_prefix, out_suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    process_one(
        (input_path, output_path),
        argparse.Namespace(labels=labels, dilate=dilate, binarize=binarize, overwrite=overwrite),
    )
    return output_path


def largest_component_dir(
    input_dir: Path | str,
    output_dir: Path | str,
    labels: list[int] | None = None,
    dilate: int = 0,
    binarize: bool = False,
    out_suffix: str = "_largest",
    out_prefix: str = "",
    overwrite: bool = False,
    n_jobs: int = 1,
) -> None:
    import argparse, functools
    from uroseg.utils.utils import build_pairs
    pairs = build_pairs(input_dir, output_dir, out_suffix, out_prefix, overwrite)
    process_map(
        functools.partial(
            process_one,
            args=argparse.Namespace(labels=labels, dilate=dilate, binarize=binarize, overwrite=overwrite),
        ),
        pairs,
        max_workers=n_jobs,
        disable=False,
        desc='uroseg largest_component',
    )
```

- [ ] **Step 8: Create `uroseg/tools/reorient.py`**

Copy `uroseg/commands/reorient_canonical.py` exactly, then add at the end:

```python
def reorient(
    input: Path | str,
    output: Path | str,
    out_suffix: str = "_reoriented",
    out_prefix: str = "",
    overwrite: bool = False,
) -> Path:
    import argparse
    input_path, output_path = Path(input), Path(output)
    if not output_path.suffix:
        from uroseg.utils.utils import build_output_path
        output_path = build_output_path(input_path, output_path, out_prefix, out_suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    process_one(
        (input_path, output_path),
        argparse.Namespace(overwrite=overwrite),
    )
    return output_path


def reorient_dir(
    input_dir: Path | str,
    output_dir: Path | str,
    out_suffix: str = "_reoriented",
    out_prefix: str = "",
    overwrite: bool = False,
    n_jobs: int = 1,
) -> None:
    import argparse, functools
    from uroseg.utils.utils import build_pairs
    pairs = build_pairs(input_dir, output_dir, out_suffix, out_prefix, overwrite)
    process_map(
        functools.partial(process_one, args=argparse.Namespace(overwrite=overwrite)),
        pairs,
        max_workers=n_jobs,
        disable=False,
        desc='uroseg reorient',
    )
```

- [ ] **Step 9: Create `uroseg/tools/transform_seg2image.py`**

Copy `uroseg/commands/transform_seg2image.py` exactly, then add at the end:

```python
def transform_seg2image(
    seg: Path | str,
    img: Path | str,
    output: Path | str,
    interpolation: str = 'nearest',
    seg_suffix: str = "_transformed",
    seg_prefix: str = "",
    overwrite: bool = False,
) -> Path:
    import argparse
    seg_path, img_path, output_path = Path(seg), Path(img), Path(output)
    if not output_path.suffix:
        from uroseg.utils.utils import build_output_path
        output_path = build_output_path(seg_path, output_path, seg_prefix, seg_suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    process_one(
        (seg_path, img_path, output_path),
        argparse.Namespace(interpolation=interpolation, overwrite=overwrite),
    )
    return output_path


def transform_seg2image_dir(
    seg_dir: Path | str,
    img_dir: Path | str,
    output_dir: Path | str,
    interpolation: str = 'nearest',
    seg_suffix: str = "_transformed",
    seg_prefix: str = "",
    overwrite: bool = False,
    n_jobs: int = 1,
) -> None:
    import argparse, functools, sys
    from uroseg.utils.utils import collect_niftis, build_output_path
    segs = collect_niftis(seg_dir)
    imgs = collect_niftis(img_dir)
    if len(segs) != len(imgs):
        print(f"Mismatch: {len(segs)} segs vs {len(imgs)} images.", file=sys.stderr)
        return
    out = Path(output_dir)
    args = argparse.Namespace(interpolation=interpolation, overwrite=overwrite)
    pairs = [
        (s, i, build_output_path(s, out, seg_prefix, seg_suffix))
        for s, i in zip(segs, imgs)
        if overwrite or not build_output_path(s, out, seg_prefix, seg_suffix).exists()
    ]
    process_map(
        functools.partial(process_one, args=args),
        pairs,
        max_workers=n_jobs,
        disable=False,
        desc='uroseg transform_seg2image',
    )
```

- [ ] **Step 10: Create `uroseg/tools/preview.py`**

Copy `uroseg/commands/preview_jpg.py` exactly, then add at the end:

```python
def preview(
    input: Path | str,
    output: Path | str,
    seg: Path | str | None = None,
    orient: str = 'sag',
    sliceloc: float = 0.5,
    label_text_right: dict[int, str] | None = None,
    label_text_left: dict[int, str] | None = None,
    out_suffix: str = "_preview",
    out_prefix: str = "",
    overwrite: bool = False,
) -> Path:
    import argparse
    input_path = Path(input)
    output_path = Path(output)
    if not str(output_path).endswith('.jpg'):
        output_path = _build_jpg_path(input_path, output_path, out_prefix, out_suffix, orient, sliceloc)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seg_path = Path(seg) if seg else None
    ltr = [[f"{k}:{v}" for k, v in label_text_right.items()] if label_text_right else []]
    ltl = [[f"{k}:{v}" for k, v in label_text_left.items()] if label_text_left else []]
    args = argparse.Namespace(
        orient=orient, sliceloc=sliceloc,
        label_text_right=ltr[0] if ltr else [],
        label_text_left=ltl[0] if ltl else [],
        overwrite=overwrite,
    )
    process_one((input_path, seg_path, output_path), args)
    return output_path


def preview_dir(
    input_dir: Path | str,
    output_dir: Path | str,
    seg_dir: Path | str | None = None,
    orient: str = 'sag',
    sliceloc: float = 0.5,
    label_text_right: dict[int, str] | None = None,
    label_text_left: dict[int, str] | None = None,
    out_suffix: str = "_preview",
    out_prefix: str = "",
    overwrite: bool = False,
    n_jobs: int = 1,
) -> None:
    import argparse, functools
    from uroseg.utils.utils import collect_niftis
    imgs = collect_niftis(input_dir)
    segs = collect_niftis(seg_dir) if seg_dir else [None] * len(imgs)
    out = Path(output_dir)
    ltr = [f"{k}:{v}" for k, v in label_text_right.items()] if label_text_right else []
    ltl = [f"{k}:{v}" for k, v in label_text_left.items()] if label_text_left else []
    args = argparse.Namespace(
        orient=orient, sliceloc=sliceloc,
        label_text_right=ltr, label_text_left=ltl, overwrite=overwrite,
    )
    pairs = [
        (i, s, _build_jpg_path(i, out, out_prefix, out_suffix, orient, sliceloc))
        for i, s in zip(imgs, segs)
        if overwrite or not _build_jpg_path(i, out, out_prefix, out_suffix, orient, sliceloc).exists()
    ]
    process_map(
        functools.partial(process_one, args=args),
        pairs,
        max_workers=n_jobs,
        disable=False,
        desc='uroseg preview',
    )
```

- [ ] **Step 11: Create `uroseg/tools/cpdir.py`**

Copy `uroseg/commands/cpdir.py` exactly, then add at the end:

```python
def cpdir(
    input: Path | str,
    output: Path | str,
    out_suffix: str = "",
    out_prefix: str = "",
    overwrite: bool = False,
) -> Path:
    import argparse
    input_path, output_path = Path(input), Path(output)
    if not output_path.suffix:
        from uroseg.utils.utils import build_output_path
        output_path = build_output_path(input_path, output_path, out_prefix, out_suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    process_one(
        (input_path, output_path),
        argparse.Namespace(overwrite=overwrite),
    )
    return output_path


def cpdir_dir(
    input_dir: Path | str,
    output_dir: Path | str,
    out_suffix: str = "",
    out_prefix: str = "",
    overwrite: bool = False,
    n_jobs: int = 1,
) -> None:
    import argparse, functools
    from uroseg.utils.utils import build_pairs
    pairs = build_pairs(input_dir, output_dir, out_suffix, out_prefix, overwrite)
    process_map(
        functools.partial(process_one, args=argparse.Namespace(overwrite=overwrite)),
        pairs,
        max_workers=n_jobs,
        disable=False,
        desc='uroseg cpdir',
    )
```

- [ ] **Step 12: Update `tests/test_commands.py` import paths**

Replace all `from uroseg.commands.` with `from uroseg.tools.` throughout the file. Specific replacements:
- `from uroseg.commands.map_labels import` → `from uroseg.tools.map_labels import`
- `from uroseg.commands.resample import` → `from uroseg.tools.resample import`
- `from uroseg.commands.reorient_canonical import` → `from uroseg.tools.reorient import`
- `from uroseg.commands.largest_component import` → `from uroseg.tools.largest_component import`
- `from uroseg.commands.preview_jpg import` → `from uroseg.tools.preview import`
- `from uroseg.commands.transform_seg2image import` → `from uroseg.tools.transform_seg2image import`

(CLI subprocess tests call `uroseg.cli` which will be updated in Task 8, so don't change those yet.)

- [ ] **Step 13: Run tool tests**

```bash
pytest tests/test_commands.py -v -k "not cli"
```
Expected: all non-CLI tests PASS

- [ ] **Step 14: Commit**

```bash
git add uroseg/tools/ tests/test_commands.py
git commit -m "feat: uroseg/tools/ with public map_labels/resample/crop/largest_component/reorient/transform_seg2image/preview/cpdir API"
```

---

### Task 7: `uroseg/__init__.py` flat public API

**Files:**
- Modify: `uroseg/__init__.py`

**Interfaces:**
- Consumes: all model classes and tool functions from Tasks 1–6
- Produces: flat public namespace for `import uroseg`

- [ ] **Step 1: Write failing test**

Create `tests/test_public_api.py`:
```python
def test_public_api_models():
    import uroseg
    assert hasattr(uroseg, 'Prostate')
    assert hasattr(uroseg, 'Bladder')
    assert hasattr(uroseg, 'get_model')
    assert hasattr(uroseg, 'list_models')
    assert uroseg.get_model('prostate').name == 'prostate'


def test_public_api_tools():
    import uroseg
    for name in ['map_labels', 'map_labels_dir',
                 'resample', 'resample_dir',
                 'reorient', 'reorient_dir',
                 'largest_component', 'largest_component_dir',
                 'crop', 'crop_dir',
                 'transform_seg2image', 'transform_seg2image_dir',
                 'preview', 'preview_dir',
                 'cpdir', 'cpdir_dir']:
        assert hasattr(uroseg, name), f"uroseg.{name} missing"
        assert callable(getattr(uroseg, name))
```

- [ ] **Step 2: Run to confirm fail**

```bash
pytest tests/test_public_api.py -v
```
Expected: FAIL

- [ ] **Step 3: Rewrite `uroseg/__init__.py`**

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
from uroseg.tools.cpdir import cpdir

# Tools — directory (multiprocessing)
from uroseg.tools.map_labels import map_labels_dir
from uroseg.tools.resample import resample_dir
from uroseg.tools.preview import preview_dir
from uroseg.tools.crop import crop_dir
from uroseg.tools.largest_component import largest_component_dir
from uroseg.tools.reorient import reorient_dir
from uroseg.tools.transform_seg2image import transform_seg2image_dir
from uroseg.tools.cpdir import cpdir_dir

__all__ = [
    'get_model', 'list_models', 'Prostate', 'Bladder',
    'map_labels', 'map_labels_dir',
    'resample', 'resample_dir',
    'preview', 'preview_dir',
    'crop', 'crop_dir',
    'largest_component', 'largest_component_dir',
    'reorient', 'reorient_dir',
    'transform_seg2image', 'transform_seg2image_dir',
    'cpdir', 'cpdir_dir',
]
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_public_api.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add uroseg/__init__.py tests/test_public_api.py
git commit -m "feat: flat public API in uroseg/__init__.py"
```

---

### Task 8: Update `cli.py` + list_models + cleanup + delete `commands/`

**Files:**
- Modify: `uroseg/cli.py`
- Modify: `uroseg/commands/list_models.py` → move to `uroseg/tools/list_models.py`
- Delete: remaining `uroseg/commands/` files (`list_models.py`, `__init__.py`)
- Modify: `uroseg/utils/utils.py` (remove dead code: `normalize_labels` stays; `files` import may be unused)
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `get_model`, `list_models` from `uroseg.models`; all tools from `uroseg.tools`

- [ ] **Step 1: Write failing test**

Add to `tests/test_cli.py`:
```python
def test_cli_uses_new_tool_paths():
    """Verify CLI still dispatches map/resample/etc correctly after tools/ move."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'map', '--help'],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert '--seg' in result.stdout or '--seg' in result.stderr
```

- [ ] **Step 2: Run existing CLI tests to see which fail**

```bash
pytest tests/test_cli.py -v
```
Note which tests fail — most should still pass since CLI subprocess tests run via the existing cli.py.

- [ ] **Step 3: Create `uroseg/tools/list_models.py`**

Copy `uroseg/commands/list_models.py` but update the imports:

```python
from __future__ import annotations
from uroseg.models import list_models as _list_models, get_model

_COMMANDS = {
    'map':                 'Remap label IDs',
    'resample':            'Resample image to target voxel size',
    'reorient':            'Reorient image to RAS canonical',
    'largest_component':   'Keep largest connected component per label',
    'crop':                'Crop image to segmentation bounding box',
    'preview':             'Generate JPG slice preview',
    'transform_seg2image': 'Resample segmentation to reference image space',
    'cpdir':               'Copy NIfTI files with optional renaming',
    'install':             'Download model weights',
    'train nnunet':        'Train with nnU-Net + AugLab',
}


def show_help() -> None:
    names = _list_models()
    lines = ['uroseg — urological anatomy segmentation', '']
    lines.append('Models:')
    if names:
        name_w = max(len(n) for n in names) + 2
        for name in sorted(names):
            model = get_model(name)
            lines.append(f'  {name:<{name_w}}{model.description}')
    else:
        lines.append('  (no models installed)')
    lines.append('')
    lines.append('Commands:')
    cmd_w = max(len(c) for c in _COMMANDS) + 2
    for cmd, desc in _COMMANDS.items():
        lines.append(f'  {cmd:<{cmd_w}}{desc}')
    lines.append('')
    lines.append("Run 'uroseg <model|command> --help' for per-command usage.")
    print('\n'.join(lines))


def main() -> None:
    show_help()
```

- [ ] **Step 4: Rewrite `uroseg/cli.py`**

```python
from __future__ import annotations
import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        from uroseg.tools.list_models import show_help
        show_help()
        return

    cmd = sys.argv[1]

    if cmd == 'list':
        from uroseg.tools.list_models import show_help
        show_help()
        return
    elif cmd == 'install':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        _run_install()
    elif cmd == 'train':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.nnunet.train import main as _train_dispatch
        _train_dispatch_wrapper()
    elif cmd == 'map':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.tools.map_labels import main as run
        run()
    elif cmd == 'resample':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.tools.resample import main as run
        run()
    elif cmd == 'preview':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.tools.preview import main as run
        run()
    elif cmd == 'crop':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.tools.crop import main as run
        run()
    elif cmd == 'largest_component':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.tools.largest_component import main as run
        run()
    elif cmd == 'reorient':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.tools.reorient import main as run
        run()
    elif cmd == 'cpdir':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.tools.cpdir import main as run
        run()
    elif cmd == 'transform_seg2image':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.tools.transform_seg2image import main as run
        run()
    elif cmd == 'predict_nnunet':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.nnunet.predict import main as run
        run()
    elif cmd.startswith('-'):
        print(f"Unknown option: {cmd}. Run 'uroseg --help' for help.", file=sys.stderr)
        sys.exit(1)
    else:
        _dispatch_model(cmd)


def _run_install() -> None:
    import argparse
    from uroseg.models import get_model, list_models
    from uroseg.utils.utils import resolve_data_path, data_dir_help
    parser = argparse.ArgumentParser(description='Download and install UroSeg model weights.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', nargs='+', metavar='MODEL',
                       help='One or more organ model names (e.g. prostate bladder)')
    group.add_argument('--all', action='store_true', help='Install all available models')
    parser.add_argument('--data-dir', default=None, help=data_dir_help())
    args = parser.parse_args()
    data_path = resolve_data_path(args.data_dir)
    names = list_models() if args.all else args.model
    print(f"Installing {len(names)} model(s) to {data_path}...")
    for name in names:
        get_model(name).install(data_path)


def _train_dispatch_wrapper() -> None:
    _ENGINES = {'nnunet': 'Train with nnU-Net + AugLab'}
    _HELP = 'Usage: uroseg train <engine> ORGAN [options]\nEngines:\n  nnunet    Train with nnU-Net + AugLab'
    argv = sys.argv[1:]
    if argv and argv[0] in ('-h', '--help'):
        print(_HELP)
        sys.exit(0)
    if not argv:
        print(_HELP, file=sys.stderr)
        sys.exit(1)
    engine = argv[0]
    if engine not in _ENGINES:
        print(f"Unknown engine: '{engine}'\n{_HELP}", file=sys.stderr)
        sys.exit(1)
    sys.argv = sys.argv[:1] + sys.argv[2:]
    if engine == 'nnunet':
        from uroseg.nnunet.train import main as run
        run()


def _dispatch_model(model: str) -> None:
    import importlib
    from uroseg.models import list_models
    if model not in list_models():
        print(
            f"Unknown model or subcommand: '{model}'\n"
            f"Run 'uroseg --help' to see available models and commands.",
            file=sys.stderr,
        )
        sys.exit(1)
    sys.argv = sys.argv[:1] + sys.argv[2:]
    mod = importlib.import_module(f'uroseg.models.{model}')
    mod.main()


if __name__ == '__main__':
    main()
```

- [ ] **Step 5: Delete remaining `commands/` files**

```bash
rm uroseg/commands/list_models.py uroseg/commands/__init__.py
rmdir uroseg/commands
```

- [ ] **Step 6: Update `tests/test_cli.py`**

Replace `from uroseg.commands.list_models import main` with `from uroseg.tools.list_models import main` in `test_list_prints_model_names`.

All subprocess-based tests should pass unchanged since the CLI interface is the same.

- [ ] **Step 7: Run full test suite**

```bash
pytest -v
```
Expected: all tests PASS

- [ ] **Step 8: Commit**

```bash
git add uroseg/cli.py uroseg/tools/list_models.py tests/test_cli.py
git commit -m "feat: update cli.py to use tools/ and nnunet/; inline install; delete commands/"
```

---

## Self-Review

After writing, verifying spec coverage:

1. **Package structure** → All 8 tasks create the required directory layout ✓
2. **SegModel / NNUNetSegModel** → Task 1 ✓
3. **Prostate / Bladder classes** → Task 2 ✓
4. **`get_model` / `list_models` registry** → Task 2 ✓
5. **Data layout `data_dir/<name>/<release_id>/`** → Task 2 (`_find_model_dir`, `install`) ✓
6. **`_download_zip` / `_extract_zip` / `_find_model_dir` in `models/base.py`** → Task 1 ✓
7. **`nnunet/helpers.py`** → Task 3 ✓
8. **`nnunet/train.py` with `--training-dir`/`-d`** → Task 4 ✓
9. **`_add_trainer("nnUNetTrainerDAExtGPU")`** → Task 4 ✓
10. **`nnunet/predict.py`** → Task 5 ✓
11. **Tools in `uroseg/tools/` with public API** → Task 6 ✓
12. **Flat `uroseg/__init__.py`** → Task 7 ✓
13. **`cli.py` unchanged interface** → Task 8 ✓
14. **`resources/auglab/` unchanged** → not touched ✓
15. **Tests updated throughout** → each task updates its own tests ✓
