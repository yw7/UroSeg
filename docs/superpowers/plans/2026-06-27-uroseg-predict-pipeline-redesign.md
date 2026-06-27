# Predict Pipeline Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the tmp-dir file-based predict pipeline with a clean in-memory pipeline: one predictor init per batch, canonical→1mm preprocessing in the caller, suppressed nnunet output, tqdm progress bar, and a generic model interface.

**Architecture:** `_suppress_nnunet` / `_init_predictor` / `_run_inference` in helpers.py provide the suppressed nnunet primitives. `SegModel` gains `init_predictor` + `predict_image(predictor, img)` (pure inference) + `post_largest_component` attribute; `predict()` applies canonical→1mm, init, infer, post-process, iso. `run_predict_cli` in predict.py becomes a generic, model-agnostic orchestrator with tqdm and `--iso`.

**Tech Stack:** Python 3.10+, nibabel, numpy, nnunetv2, tqdm, unittest.mock

## Global Constraints

- `predict_image(predictor, img)` — pure inference only; no preprocessing, no postprocessing, no filesystem access
- `canonical → 1mm iso` preprocessing applied by BOTH `SegModel.predict()` and `run_predict_cli` before calling `predict_image`
- `_suppress_nnunet()` is the single suppression point; used inside `_init_predictor` (around `initialize_from_trained_model_folder`) and inside `_run_inference` (around `predict_single_npy_array`)
- `allow_tqdm=False` on the nnUNetPredictor constructor (suppresses per-tile tqdm)
- `init_predictor` called exactly once per CLI invocation (outside the file loop)
- `run_predict_cli` has no nnunet imports and no `tempfile`
- `post_largest_component = True` on `Prostate`; `False` (default) on `SegModel`
- `run_predict_cli` loses the `largest_component` parameter; reads `model.post_largest_component`
- `--iso` added to `add_inference_args` as store_true flag

---

### Task 1: `_suppress_nnunet`, `_init_predictor`, `_run_inference` in helpers.py

**Files:**
- Modify: `uroseg/nnunet/helpers.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `_resolve_fold_dir` already in helpers.py; `Image` already imported at module level
- Produces:
  - `_suppress_nnunet() → contextmanager` — used in Tasks 2 and 3 via `_init_predictor` / `_run_inference`
  - `_init_predictor(model_dir: Path, fold: int = 0, device: str = 'cuda') → nnUNetPredictor`
  - `_run_inference(predictor, img: Image) → np.ndarray`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_models.py`:

```python
def test_suppress_nnunet_silences_print(capsys):
    """_suppress_nnunet redirects stdout so prints are hidden."""
    from uroseg.nnunet.helpers import _suppress_nnunet
    with _suppress_nnunet():
        print("should be hidden")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_suppress_nnunet_sets_env_vars():
    """_suppress_nnunet sets dummy env vars for nnunet keys."""
    import os
    from uroseg.nnunet.helpers import _suppress_nnunet
    keys = ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results')
    for k in keys:
        os.environ.pop(k, None)
    with _suppress_nnunet():
        for k in keys:
            assert os.environ.get(k) == ''


def test_suppress_nnunet_restores_env(monkeypatch):
    """_suppress_nnunet restores original env var values on exit."""
    import os
    from uroseg.nnunet.helpers import _suppress_nnunet
    monkeypatch.setenv('nnUNet_raw', 'original_value')
    with _suppress_nnunet():
        pass
    assert os.environ['nnUNet_raw'] == 'original_value'


def test_init_predictor_initializes_from_folder(tmp_path):
    """_init_predictor calls initialize_from_trained_model_folder on the predictor."""
    import sys
    from unittest.mock import MagicMock, patch
    from uroseg.nnunet.helpers import _init_predictor

    fold_dir = tmp_path / 'trainer' / 'fold_0'
    fold_dir.mkdir(parents=True)
    (fold_dir / 'checkpoint_final.pth').touch()

    mock_predictor = MagicMock()
    mock_nnunet = MagicMock()
    mock_nnunet.nnUNetPredictor.return_value = mock_predictor

    with patch.dict(sys.modules, {
        'nnunetv2': MagicMock(),
        'nnunetv2.inference': MagicMock(),
        'nnunetv2.inference.predict_from_raw_data': mock_nnunet,
    }):
        result = _init_predictor(tmp_path, fold=0, device='cpu')

    assert result is mock_predictor
    mock_predictor.initialize_from_trained_model_folder.assert_called_once()


def test_run_inference_calls_predict_single(tmp_path):
    """_run_inference passes (1,x,y,z) array and spacing; returns seg array."""
    import numpy as np
    import nibabel as nib
    from unittest.mock import MagicMock
    from uroseg.utils.image import Image
    from uroseg.nnunet.helpers import _run_inference

    data = np.ones((8, 8, 8), dtype=np.float32)
    img = Image(data=data, affine=np.eye(4), header=nib.Nifti1Image(data, np.eye(4)).header)
    expected = np.zeros((8, 8, 8), dtype=np.uint8)

    mock_predictor = MagicMock()
    mock_predictor.predict_single_npy_array.return_value = expected

    result = _run_inference(mock_predictor, img)

    assert result is expected
    call_args = mock_predictor.predict_single_npy_array.call_args
    assert call_args[0][0].shape == (1, 8, 8, 8)   # (1, x, y, z)
    assert call_args[0][1]['spacing'] == [1.0, 1.0, 1.0]
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_models.py::test_suppress_nnunet_silences_print \
       tests/test_models.py::test_suppress_nnunet_sets_env_vars \
       tests/test_models.py::test_suppress_nnunet_restores_env \
       tests/test_models.py::test_init_predictor_initializes_from_folder \
       tests/test_models.py::test_run_inference_calls_predict_single -v
```
Expected: FAIL — `ImportError: cannot import name '_suppress_nnunet'`

- [ ] **Step 3: Add `_suppress_nnunet`, `_init_predictor`, `_run_inference` to `uroseg/nnunet/helpers.py`**

Add `import contextlib, io, warnings` to the existing top-level imports block.

Insert after the existing `setup_env` function and before `extract_dataset_id`:

```python
@contextlib.contextmanager
def _suppress_nnunet():
    """Silence nnunet env-var warnings, per-case prints, and Python warnings."""
    env_keys = ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results')
    saved = {}
    for k in env_keys:
        saved[k] = os.environ.get(k)
        if k not in os.environ:
            os.environ[k] = ''
    buf = io.StringIO()
    try:
        with warnings.catch_warnings(), \
             contextlib.redirect_stdout(buf), \
             contextlib.redirect_stderr(buf):
            warnings.simplefilter('ignore')
            yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _init_predictor(model_dir: Path, fold: int = 0, device: str = 'cuda'):
    """Create and initialize an nnUNetPredictor (suppresses nnunet output)."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    elif device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    fold_dir, checkpoint_name = _resolve_fold_dir(model_dir, fold)
    with _suppress_nnunet():
        predictor.initialize_from_trained_model_folder(
            str(fold_dir),
            use_folds=(fold,),
            checkpoint_name=checkpoint_name,
        )
    return predictor


def _run_inference(predictor, img: Image) -> 'np.ndarray':
    """Run predict_single_npy_array on img (already 1mm canonical). Suppresses output."""
    import numpy as np
    spacing = [float(s) for s in img.header.get_zooms()[:3]]
    input_array = img.data[np.newaxis].astype(np.float32)
    with _suppress_nnunet():
        seg_array = predictor.predict_single_npy_array(
            input_array,
            {'spacing': spacing},
            segmentation_previous_stage=None,
            output_file_truncated=None,
            save_probabilities=False,
        )
    return seg_array
```

Also add `import os` to the top-level imports (it is not yet there). Also refactor `run_predict_array` to use the new helpers — replace its body with:

```python
def run_predict_array(
    model_dir: Path,
    img: Image,
    fold: int = 0,
    device: str = 'cuda',
) -> Image:
    """Run nnunet inference on an in-memory Image. Returns seg Image in same space as input."""
    predictor = _init_predictor(model_dir, fold=fold, device=device)
    seg_array = _run_inference(predictor, img)
    return Image(data=seg_array, affine=img.affine, header=img.header)
```

- [ ] **Step 4: Run the 5 new tests**

```
pytest tests/test_models.py::test_suppress_nnunet_silences_print \
       tests/test_models.py::test_suppress_nnunet_sets_env_vars \
       tests/test_models.py::test_suppress_nnunet_restores_env \
       tests/test_models.py::test_init_predictor_initializes_from_folder \
       tests/test_models.py::test_run_inference_calls_predict_single -v
```
Expected: all PASS

- [ ] **Step 5: Run full suite**

```
pytest tests/ -q --ignore=tests/test_install.py
```
Expected: same count as before (153 passed)

- [ ] **Step 6: Commit**

```bash
git add uroseg/nnunet/helpers.py tests/test_models.py
git commit -m "feat: _suppress_nnunet, _init_predictor, _run_inference in helpers.py"
```

---

### Task 2: Refactor `SegModel` and `NNUNetSegModel` in `base.py`

**Files:**
- Modify: `uroseg/models/base.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Consumes: `_init_predictor`, `_run_inference` from Task 1
- Produces:
  - `SegModel.post_largest_component: bool = False`
  - `SegModel.init_predictor(self, model_dir: Path, fold: int = 0, device: str = 'cuda') → Any`
  - `SegModel.predict_image(self, predictor: Any, img: Image) → Image` — signature change: takes predictor + preprocessed img
  - `SegModel.predict(self, input, output_dir, fold=0, device='cuda', iso=False) → Path` — applies canonical→1mm, init_predictor, predict_image, post_largest_component, iso
  - `NNUNetSegModel.init_predictor` — wraps `_init_predictor`
  - `NNUNetSegModel.predict_image(predictor, img_1mm)` — pure inference via `_run_inference`

- [ ] **Step 1: Update the failing tests in `tests/test_models.py`**

**Delete** these four existing tests (they use the old `predict_image(img)` signature):
- `test_segmodel_predict_image_raises_not_implemented`
- `test_segmodel_predict_raises_via_predict_image`
- `test_segmodel_predict_io_wrapper_iso_false`
- `test_segmodel_predict_io_wrapper_iso_true`
- `test_nnunet_predict_image_runs_pipeline`

**Add** these replacements:

```python
def test_segmodel_init_predictor_raises_not_implemented():
    """init_predictor() raises NotImplementedError on the base class."""
    from uroseg.models.base import SegModel
    class M(SegModel):
        name = 'x'; description = ''; weights_url = ''; labels = {}
    with pytest.raises(NotImplementedError):
        M().init_predictor(Path('/tmp'), fold=0, device='cpu')


def test_segmodel_predict_image_raises_not_implemented():
    """predict_image() raises NotImplementedError on the base class."""
    import numpy as np
    import nibabel as nib
    from uroseg.models.base import SegModel
    from uroseg.utils.image import Image
    class M(SegModel):
        name = 'x'; description = ''; weights_url = ''; labels = {}
    data = np.zeros((5, 5, 5), dtype=np.float32)
    img = Image(data=data, affine=np.eye(4), header=nib.Nifti1Image(data, np.eye(4)).header)
    with pytest.raises(NotImplementedError):
        M().predict_image(None, img)


def test_segmodel_predict_raises_via_init_predictor(tmp_path):
    """predict() propagates NotImplementedError from init_predictor()."""
    import numpy as np
    import nibabel as nib
    from uroseg.models.base import SegModel
    inp = tmp_path / 'a.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((5, 5, 5), dtype=np.float32), np.eye(4)), str(inp))
    class M(SegModel):
        name = 'x'; description = ''; weights_url = ''; labels = {}
    with pytest.raises(NotImplementedError):
        M().predict(inp, tmp_path / 'out')


def test_segmodel_predict_applies_largest_component(tmp_path):
    """predict() calls keep_largest_component when post_largest_component=True."""
    import numpy as np
    import nibabel as nib
    from unittest.mock import MagicMock
    from uroseg.models.base import SegModel
    from uroseg.utils.image import Image

    data = np.ones((8, 8, 8), dtype=np.float32)
    inp = tmp_path / 'img.nii.gz'
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(inp))

    seg_data = np.zeros((8, 8, 8), dtype=np.uint8)
    seg_data[3:5, 3:5, 3:5] = 1

    class M(SegModel):
        name = 'x'; description = ''; weights_url = ''; labels = {}
        post_largest_component = True
        def init_predictor(self, model_dir, fold=0, device='cuda'):
            return MagicMock()
        def predict_image(self, predictor, img):
            return Image(data=seg_data.copy(), affine=np.eye(4),
                         header=nib.Nifti1Image(seg_data, np.eye(4)).header)

    out = M().predict(inp, tmp_path / 'out')
    assert out.exists()


def test_segmodel_predict_iso_false_resamples_back(tmp_path):
    """predict() with iso=False resamples seg back to original affine."""
    import numpy as np
    import nibabel as nib
    from unittest.mock import MagicMock
    from uroseg.models.base import SegModel
    from uroseg.utils.image import Image

    data = np.ones((8, 8, 8), dtype=np.float32)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    inp = tmp_path / 'img.nii.gz'
    nib.save(nib.Nifti1Image(data, affine), str(inp))

    class M(SegModel):
        name = 'x'; description = ''; weights_url = ''; labels = {}
        def init_predictor(self, model_dir, fold=0, device='cuda'):
            return MagicMock()
        def predict_image(self, predictor, img):
            # img is already 1mm — return seg in that space
            seg_data = np.zeros_like(img.data, dtype=np.uint8)
            return Image(data=seg_data, affine=img.affine, header=img.header)

    out = M().predict(inp, tmp_path / 'out', iso=False)
    result = Image.load(out)
    np.testing.assert_allclose(np.abs(result.affine[0, 0]), 2.0, atol=0.2)


def test_segmodel_predict_iso_true_keeps_model_space(tmp_path):
    """predict() with iso=True leaves seg in 1mm space."""
    import numpy as np
    import nibabel as nib
    from unittest.mock import MagicMock
    from uroseg.models.base import SegModel
    from uroseg.utils.image import Image

    data = np.ones((8, 8, 8), dtype=np.float32)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    inp = tmp_path / 'img.nii.gz'
    nib.save(nib.Nifti1Image(data, affine), str(inp))

    class M(SegModel):
        name = 'x'; description = ''; weights_url = ''; labels = {}
        def init_predictor(self, model_dir, fold=0, device='cuda'):
            return MagicMock()
        def predict_image(self, predictor, img):
            seg_data = np.zeros_like(img.data, dtype=np.uint8)
            return Image(data=seg_data, affine=img.affine, header=img.header)

    out = M().predict(inp, tmp_path / 'out', iso=True)
    result = Image.load(out)
    np.testing.assert_allclose(np.abs(result.affine[0, 0]), 1.0, atol=0.2)


def test_nnunet_predict_image_calls_run_inference(tmp_path):
    """NNUNetSegModel.predict_image calls _run_inference and returns Image."""
    import numpy as np
    import nibabel as nib
    from unittest.mock import patch, MagicMock
    from uroseg.models.base import NNUNetSegModel
    from uroseg.utils.image import Image

    class TestModel(NNUNetSegModel):
        name = 'test'; description = ''; weights_url = ''
        labels = {}; nnunet_task = 'Dataset999_Test'

    data = np.ones((8, 8, 8), dtype=np.float32)
    img = Image(data=data, affine=np.eye(4), header=nib.Nifti1Image(data, np.eye(4)).header)
    fake_seg = np.zeros((8, 8, 8), dtype=np.uint8)

    with patch('uroseg.nnunet.helpers._run_inference', return_value=fake_seg) as mock_infer:
        result = TestModel().predict_image(MagicMock(), img)

    assert isinstance(result, Image)
    mock_infer.assert_called_once()
    np.testing.assert_array_equal(result.data, fake_seg)
```

- [ ] **Step 2: Run the new tests to verify they fail**

```
pytest tests/test_models.py::test_segmodel_init_predictor_raises_not_implemented \
       tests/test_models.py::test_segmodel_predict_image_raises_not_implemented \
       tests/test_models.py::test_segmodel_predict_raises_via_init_predictor \
       tests/test_models.py::test_segmodel_predict_applies_largest_component \
       tests/test_models.py::test_segmodel_predict_iso_false_resamples_back \
       tests/test_models.py::test_segmodel_predict_iso_true_keeps_model_space \
       tests/test_models.py::test_nnunet_predict_image_calls_run_inference -v
```
Expected: FAIL (various — `init_predictor` not defined, wrong `predict_image` signature)

- [ ] **Step 3: Rewrite `SegModel` and `NNUNetSegModel` in `uroseg/models/base.py`**

Replace the `SegModel` class body (keep `install`, `predict_dir` unchanged):

```python
class SegModel:
    name: str
    description: str
    weights_url: str
    labels: dict
    post_largest_component: bool = False

    def install(self, data_dir: Path) -> None:
        # unchanged — do not modify
        ...

    def init_predictor(self, model_dir: Path, fold: int = 0, device: str = 'cuda'):
        raise NotImplementedError(f"{self.__class__.__name__} does not implement init_predictor()")

    def predict_image(self, predictor, img: Image) -> Image:
        """Run inference on img (canonical 1mm, already preprocessed). Return seg in model space."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement predict_image()")

    def predict(self, input: Path | str, output_dir: Path | str,
                fold: int = 0, device: str = 'cuda', iso: bool = False) -> Path:
        """Load → canonical+1mm → init_predictor → predict_image → post-process → save."""
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

    def predict_dir(self, input_dir: Path, output_dir: Path,
                    n_jobs: int = 1, **kwargs) -> None:
        # unchanged — do not modify
        ...
```

Replace the entire `NNUNetSegModel` class body:

```python
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
```

- [ ] **Step 4: Run the 7 new tests**

```
pytest tests/test_models.py::test_segmodel_init_predictor_raises_not_implemented \
       tests/test_models.py::test_segmodel_predict_image_raises_not_implemented \
       tests/test_models.py::test_segmodel_predict_raises_via_init_predictor \
       tests/test_models.py::test_segmodel_predict_applies_largest_component \
       tests/test_models.py::test_segmodel_predict_iso_false_resamples_back \
       tests/test_models.py::test_segmodel_predict_iso_true_keeps_model_space \
       tests/test_models.py::test_nnunet_predict_image_calls_run_inference -v
```
Expected: all PASS

- [ ] **Step 5: Run full suite**

```
pytest tests/ -q --ignore=tests/test_install.py
```
Expected: all pass (some old tests removed, new ones added — net count may change)

- [ ] **Step 6: Commit**

```bash
git add uroseg/models/base.py tests/test_models.py
git commit -m "refactor: SegModel.init_predictor + predict_image(predictor, img); canonical+1mm in predict()"
```

---

### Task 3: Rewrite `run_predict_cli`; add `--iso`; update prostate

**Files:**
- Modify: `uroseg/nnunet/predict.py`
- Modify: `uroseg/models/prostate.py`
- Modify: `tests/test_inference.py`

**Interfaces:**
- Consumes: `model.init_predictor`, `model.predict_image`, `model.post_largest_component` from Task 2; `keep_largest_component`, `resample_seg_to_image`, `save_nifti_seg` already available
- Produces: `run_predict_cli(model, args)` — no `largest_component` param; `--iso` in `add_inference_args`

- [ ] **Step 1: Update tests in `tests/test_inference.py`**

**Delete** `test_run_predict_cli_auto_installs_if_missing` (old signature with mock run_predict).

**Update** `test_add_inference_args_defaults` — add assertion for `iso`:

```python
def test_add_inference_args_defaults():
    from uroseg.nnunet.predict import add_inference_args
    parser = argparse.ArgumentParser()
    add_inference_args(parser)
    args = parser.parse_args(['img.nii.gz', 'out/'])
    assert args.fold == 0
    assert args.device == 'cuda'
    assert args.out_suffix == '_seg'
    assert args.out_prefix == ''
    assert args.overwrite is False
    assert args.quiet is False
    assert args.iso is False
```

**Add** new tests:

```python
def test_add_inference_args_iso_flag():
    """--iso flag is parsed correctly."""
    from uroseg.nnunet.predict import add_inference_args
    parser = argparse.ArgumentParser()
    add_inference_args(parser)
    args = parser.parse_args(['img.nii.gz', '--iso'])
    assert args.iso is True


def test_run_predict_cli_no_tempdir(tmp_path):
    """run_predict_cli does not create a temporary directory."""
    import numpy as np
    import nibabel as nib
    import tempfile as tempfile_mod
    from unittest.mock import patch, MagicMock
    from uroseg.nnunet.predict import run_predict_cli
    from uroseg.models.prostate import Prostate

    inp = tmp_path / 'scan.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(inp))

    args = argparse.Namespace(
        img=str(inp), out=str(tmp_path / 'out'),
        fold=0, device='cpu', out_suffix='_seg', out_prefix='',
        data_dir=str(tmp_path), overwrite=True, quiet=True,
        max_workers=1, iso=False,
    )

    import numpy as np
    import nibabel as nib
    from uroseg.utils.image import Image

    fake_seg = np.zeros((4, 4, 4), dtype=np.uint8)

    model = Prostate()
    mock_predictor = MagicMock()

    def fake_init_predictor(model_dir, fold=0, device='cuda'):
        return mock_predictor

    def fake_predict_image(predictor, img):
        return Image(data=fake_seg, affine=img.affine, header=img.header)

    tempdir_calls = []
    original_tempdir = tempfile_mod.TemporaryDirectory
    def tracking_tempdir(*a, **kw):
        tempdir_calls.append(True)
        return original_tempdir(*a, **kw)

    with patch.object(model, 'init_predictor', side_effect=fake_init_predictor), \
         patch.object(model, 'predict_image', side_effect=fake_predict_image), \
         patch('uroseg.models.base._find_model_dir', return_value=tmp_path), \
         patch('tempfile.TemporaryDirectory', side_effect=tracking_tempdir):
        run_predict_cli(model, args)

    assert len(tempdir_calls) == 0, "run_predict_cli must not use TemporaryDirectory"


def test_run_predict_cli_init_predictor_called_once(tmp_path):
    """init_predictor is called once regardless of how many input files."""
    import numpy as np
    import nibabel as nib
    from unittest.mock import patch, MagicMock, call
    from uroseg.nnunet.predict import run_predict_cli
    from uroseg.models.prostate import Prostate
    from uroseg.utils.image import Image

    for name in ('a.nii.gz', 'b.nii.gz'):
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)),
                 str(tmp_path / name))

    args = argparse.Namespace(
        img=str(tmp_path), out=str(tmp_path / 'out'),
        fold=0, device='cpu', out_suffix='_seg', out_prefix='',
        data_dir=str(tmp_path), overwrite=True, quiet=True,
        max_workers=1, iso=False,
    )
    fake_seg = np.zeros((4, 4, 4), dtype=np.uint8)
    model = Prostate()
    init_calls = []

    def fake_init(model_dir, fold=0, device='cuda'):
        init_calls.append(1)
        return MagicMock()

    with patch.object(model, 'init_predictor', side_effect=fake_init), \
         patch.object(model, 'predict_image',
                      return_value=Image(fake_seg, np.eye(4),
                                         nib.Nifti1Image(fake_seg, np.eye(4)).header)), \
         patch('uroseg.models.base._find_model_dir', return_value=tmp_path):
        run_predict_cli(model, args)

    assert len(init_calls) == 1, "init_predictor must be called exactly once"


def test_run_predict_cli_auto_installs_if_missing(tmp_path):
    """run_predict_cli auto-installs when model not found, then proceeds."""
    import numpy as np
    import nibabel as nib
    from unittest.mock import patch, MagicMock
    from uroseg.nnunet.predict import run_predict_cli
    from uroseg.models.prostate import Prostate
    from uroseg.utils.image import Image

    inp = tmp_path / 'scan.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(inp))

    args = argparse.Namespace(
        img=str(inp), out=str(tmp_path / 'out'),
        fold=0, device='cpu', out_suffix='_seg', out_prefix='',
        data_dir=str(tmp_path), overwrite=True, quiet=True,
        max_workers=1, iso=False,
    )
    fake_seg = np.zeros((4, 4, 4), dtype=np.uint8)
    model = Prostate()
    installed = []

    def fake_install(data_dir):
        installed.append(data_dir)
        (data_dir / 'prostate' / 'r0').mkdir(parents=True)

    fake_model_dir = MagicMock()

    with patch.object(model, 'install', side_effect=fake_install), \
         patch('uroseg.models.base._find_model_dir',
               side_effect=[FileNotFoundError('not found'), fake_model_dir, fake_model_dir]), \
         patch.object(model, 'init_predictor', return_value=MagicMock()), \
         patch.object(model, 'predict_image',
                      return_value=Image(fake_seg, np.eye(4),
                                         nib.Nifti1Image(fake_seg, np.eye(4)).header)):
        run_predict_cli(model, args)

    assert len(installed) == 1
```

- [ ] **Step 2: Run new tests to verify they fail**

```
pytest tests/test_inference.py::test_add_inference_args_iso_flag \
       tests/test_inference.py::test_run_predict_cli_no_tempdir \
       tests/test_inference.py::test_run_predict_cli_init_predictor_called_once \
       tests/test_inference.py::test_run_predict_cli_auto_installs_if_missing -v
```
Expected: FAIL (`--iso` unknown; old `run_predict_cli` signature)

- [ ] **Step 3: Rewrite `uroseg/nnunet/predict.py`**

Remove the `import tempfile` line. Remove `import sys` if unused after edits (keep if used in `main()`). Add `from tqdm import tqdm` to top-level imports.

Replace `add_inference_args`:

```python
def add_inference_args(parser: argparse.ArgumentParser) -> None:
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
```

Replace `run_predict_cli` entirely:

```python
def run_predict_cli(model: SegModel, args) -> None:
    """Generic in-memory prediction loop: no tmp dir, single IO per image."""
    from uroseg.models.base import _find_model_dir
    from uroseg.tools.largest_component import keep_largest_component
    from uroseg.tools.transform_seg2image import resample_seg_to_image
    from uroseg.utils.image import save_nifti_seg

    data_path = resolve_data_path(args.data_dir)
    try:
        model_dir = _find_model_dir(model.name, data_path)
    except FileNotFoundError:
        print(f"Model '{model.name}' not installed — downloading...")
        model.install(data_path)
        model_dir = _find_model_dir(model.name, data_path)

    inputs = collect_niftis(args.img)
    if not inputs:
        print(f"No NIfTI files found in {args.img}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor = model.init_predictor(model_dir, fold=args.fold, device=args.device)

    for inp in tqdm(inputs, desc=f'uroseg {model.name}', disable=args.quiet):
        dest = build_output_path(inp, out_dir, args.out_prefix, args.out_suffix)
        if not args.overwrite and dest.exists():
            continue
        img_orig = Image.load(inp)
        img_1mm = img_orig.as_canonical().resample((1.0, 1.0, 1.0))
        seg = model.predict_image(predictor, img_1mm)
        if model.post_largest_component:
            seg.data = keep_largest_component(seg.data, binarize=True)
        if not args.iso:
            seg = resample_seg_to_image(seg, img_orig)
        save_nifti_seg(seg.data, seg.affine, seg.header, str(dest))

    if not args.quiet:
        print(f"Segmentations saved to {out_dir}")
```

- [ ] **Step 4: Update `uroseg/models/prostate.py`**

Add `post_largest_component = True` to `Prostate` class. Change `main()`:

```python
class Prostate(NNUNetSegModel):
    name = "prostate"
    description = "Prostate: whole (1), transition zone (2), peripheral zone (3)"
    weights_url = "https://github.com/yw7/UroSeg/releases/download/20260627/Dataset101_Prostate__nnUNetTrainerDAExtGPU__nnUNetPlans__3d_fullres__fold_0_20260627.zip"
    labels = {"background": 0, "prostate": [1, 2, 3], "prostate_tz": 2, "prostate_pz": 3}
    nnunet_task = "Dataset101_Prostate"
    post_largest_component = True


MODEL = Prostate()
NNUNET_TASK = Prostate.nnunet_task


def main() -> None:
    import argparse
    from uroseg.nnunet.predict import add_inference_args, run_predict_cli
    parser = argparse.ArgumentParser(prog='uroseg prostate', description=Prostate.description)
    add_inference_args(parser)
    args = parser.parse_args()
    run_predict_cli(Prostate(), args)
```

- [ ] **Step 5: Run the 4 new tests**

```
pytest tests/test_inference.py::test_add_inference_args_iso_flag \
       tests/test_inference.py::test_run_predict_cli_no_tempdir \
       tests/test_inference.py::test_run_predict_cli_init_predictor_called_once \
       tests/test_inference.py::test_run_predict_cli_auto_installs_if_missing -v
```
Expected: all PASS

- [ ] **Step 6: Run full suite**

```
pytest tests/ -q --ignore=tests/test_install.py
```
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add uroseg/nnunet/predict.py uroseg/models/prostate.py tests/test_inference.py
git commit -m "refactor: run_predict_cli — in-memory, tqdm, --iso, no tmp dir"
```
