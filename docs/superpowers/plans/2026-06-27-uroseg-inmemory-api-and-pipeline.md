# UroSeg In-Memory API and Inference Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose in-memory data functions in the public API and replace the file-based `predict()` with a full pre/post-processing pipeline that touches the filesystem only once per inference call.

**Architecture:** `SegModel.predict()` becomes a generic IO wrapper (load → `predict_image()` → iso-transform → save); each model implements `predict_image(img: Image) → Image` with no IO. `NNUNetSegModel.predict_image()` reorients, resamples to 1 mm iso, runs nnunet in-memory via `run_predict_array`, then keeps the largest component. The `iso=False` transform-back lives on `SegModel.predict()` so all future models inherit it automatically.

**Tech Stack:** Python 3.10+, nibabel, numpy, nnunetv2 (`nnUNetPredictor.predict_single_npy_array`), unittest.mock for nnunet tests.

## Global Constraints

- `SegModel.predict()` signature: `(self, input: Path | str, output_dir: Path | str, iso: bool = False, **kwargs) → Path`
- `SegModel.predict_image()` signature: `(self, img: Image, **kwargs) → Image` — raises `NotImplementedError`
- `NNUNetSegModel.predict_image()` signature: `(self, img: Image, fold: int = 0, device: str = 'cuda') → Image`
- `run_predict_array()` signature: `(model_dir: Path, img: Image, fold: int = 0, device: str = 'cuda') → Image`
- `iso=False` (default): seg is transformed back to original input space/affine
- `iso=True`: seg left in model's working space (1 mm canonical for nnunet)
- `run_predict_array` and torch/nnunet imports stay **lazy** (inside function body) to avoid loading pytorch at module import time
- `Image`, `save_nifti_seg`, `resample_seg_to_image`, `keep_largest_component` are **module-level** imports in `base.py`
- All existing 150 tests must continue to pass

---

## File Map

| File | Change |
|---|---|
| `uroseg/__init__.py` | Add 5 exports: `Image`, `keep_largest_component`, `resample_seg_to_image`, `apply_map`, `make_preview` |
| `uroseg/nnunet/helpers.py` | Add module-level `Image` import; add `run_predict_array()` |
| `uroseg/models/base.py` | Add module-level imports; add `SegModel.predict_image()`; rewrite `SegModel.predict()` as IO wrapper; replace `NNUNetSegModel.predict()` with `NNUNetSegModel.predict_image()` |
| `tests/test_public_api.py` | Add test for new in-memory exports |
| `tests/test_models.py` | Replace `test_segmodel_predict_raises`; add tests for `predict_image`, `predict` IO wrapper, `run_predict_array` |

---

### Task 1: In-memory API exports

**Files:**
- Modify: `uroseg/__init__.py`
- Test: `tests/test_public_api.py`

**Interfaces:**
- Consumes: `Image` from `uroseg.utils.image`; `keep_largest_component` from `uroseg.tools.largest_component`; `resample_seg_to_image` from `uroseg.tools.transform_seg2image`; `apply_map` from `uroseg.tools.map_labels`; `make_preview` from `uroseg.tools.preview` — all already exist, none are currently exported
- Produces: `uroseg.Image`, `uroseg.keep_largest_component`, `uroseg.resample_seg_to_image`, `uroseg.apply_map`, `uroseg.make_preview` available for Tasks 3 and user code

- [ ] **Step 1: Write the failing test**

Add to `tests/test_public_api.py`:

```python
def test_public_api_inmemory():
    import uroseg
    import numpy as np

    # Image class is exported
    assert hasattr(uroseg, 'Image')
    img = uroseg.Image.load  # has .load classmethod
    assert callable(img)

    # In-memory data functions are exported and callable
    for name in ['keep_largest_component', 'resample_seg_to_image',
                 'apply_map', 'make_preview']:
        assert hasattr(uroseg, name), f"uroseg.{name} missing"
        assert callable(getattr(uroseg, name))

    # apply_map works on a numpy array (smoke test)
    data = np.array([[[0, 1, 2]]], dtype=np.uint8)
    result = uroseg.apply_map(data, {1: 10, 2: 20})
    assert result[0, 0, 1] == 10
    assert result[0, 0, 2] == 20
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_public_api.py::test_public_api_inmemory -v
```
Expected: FAIL — `AttributeError: module 'uroseg' has no attribute 'Image'`

- [ ] **Step 3: Add exports to `uroseg/__init__.py`**

Insert after the existing `# Models` block and before `# Tools — single file`:

```python
# In-memory data layer
from uroseg.utils.image import Image
from uroseg.tools.largest_component import keep_largest_component
from uroseg.tools.transform_seg2image import resample_seg_to_image
from uroseg.tools.map_labels import apply_map
from uroseg.tools.preview import make_preview
```

Add to `__all__`:
```python
'Image',
'keep_largest_component', 'resample_seg_to_image', 'apply_map', 'make_preview',
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_public_api.py -v
```
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add uroseg/__init__.py tests/test_public_api.py
git commit -m "feat: export Image and in-memory data functions in public API"
```

---

### Task 2: `run_predict_array` in nnunet helpers

**Files:**
- Modify: `uroseg/nnunet/helpers.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `Image` from `uroseg.utils.image` (new module-level import in helpers.py); `nnUNetPredictor.predict_single_npy_array`
- Produces: `run_predict_array(model_dir: Path, img: Image, fold: int = 0, device: str = 'cuda') → Image` — used by `NNUNetSegModel.predict_image()` in Task 3

- [ ] **Step 1: Write the failing test**

Add to `tests/test_models.py`:

```python
def test_run_predict_array_returns_image(tmp_path):
    """run_predict_array calls predict_single_npy_array and returns Image in same space."""
    import numpy as np
    import nibabel as nib
    from unittest.mock import patch, MagicMock
    from uroseg.utils.image import Image
    from uroseg.nnunet.helpers import run_predict_array

    # Build a fake model_dir with fold_0 subdir so glob finds it
    fold_dir = tmp_path / 'trainer' / 'fold_0'
    fold_dir.mkdir(parents=True)

    data = np.ones((10, 10, 10), dtype=np.float32)
    affine = np.eye(4)
    img = Image(data=data, affine=affine, header=nib.Nifti1Image(data, affine).header)

    fake_seg = np.zeros((10, 10, 10), dtype=np.uint8)
    fake_seg[3:7, 3:7, 3:7] = 1

    mock_predictor = MagicMock()
    mock_predictor.predict_single_npy_array.return_value = fake_seg

    with patch('nnunetv2.inference.predict_from_raw_data.nnUNetPredictor',
               return_value=mock_predictor):
        result = run_predict_array(tmp_path, img, fold=0, device='cpu')

    assert isinstance(result, Image)
    assert result.data.shape == (10, 10, 10)
    np.testing.assert_array_equal(result.data, fake_seg)
    np.testing.assert_array_equal(result.affine, img.affine)

    # Verify predict_single_npy_array was called with correct shape and spacing
    call_args = mock_predictor.predict_single_npy_array.call_args
    assert call_args[0][0].shape == (1, 10, 10, 10)   # (1, x, y, z)
    assert call_args[0][1]['spacing'] == [1.0, 1.0, 1.0]
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_models.py::test_run_predict_array_returns_image -v
```
Expected: FAIL — `ImportError: cannot import name 'run_predict_array'`

- [ ] **Step 3: Add module-level `Image` import and `run_predict_array` to `uroseg/nnunet/helpers.py`**

Add to the top-level imports (after existing imports, before `if TYPE_CHECKING`):

```python
from uroseg.utils.image import Image
```

Add `run_predict_array` after `run_predict`:

```python
def run_predict_array(
    model_dir: Path,
    img: Image,
    fold: int = 0,
    device: str = 'cuda',
) -> Image:
    """Run nnunet inference on an in-memory Image. Returns seg Image in same space as input."""
    import numpy as np
    import nibabel as nib
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
        allow_tqdm=True,
    )

    fold_matches = sorted(model_dir.glob(f'**/fold_{fold}'))
    fold_dir = fold_matches[0].parent if fold_matches else model_dir
    predictor.initialize_from_trained_model_folder(
        str(fold_dir),
        use_folds=(fold,),
        checkpoint_name='checkpoint_best.pth',
    )

    nib_img = nib.Nifti1Image(img.data, img.affine, img.header)
    spacing = [float(s) for s in nib_img.header.get_zooms()[:3]]

    input_array = img.data[np.newaxis].astype(np.float32)  # (1, x, y, z)
    seg_array = predictor.predict_single_npy_array(
        input_array,
        {'spacing': spacing},
        segmentation_previous_stage=None,
        output_file_truncated=None,
        save_probabilities=False,
    )
    return Image(data=seg_array, affine=img.affine, header=img.header)
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_models.py::test_run_predict_array_returns_image -v
```
Expected: PASS

- [ ] **Step 5: Run full suite to check nothing broke**

```
pytest tests/ -q
```
Expected: 151 passed

- [ ] **Step 6: Commit**

```bash
git add uroseg/nnunet/helpers.py tests/test_models.py
git commit -m "feat: run_predict_array — in-memory nnunet inference via predict_single_npy_array"
```

---

### Task 3: `SegModel.predict_image`, `SegModel.predict` IO wrapper, `NNUNetSegModel.predict_image`

**Files:**
- Modify: `uroseg/models/base.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Consumes: `Image`, `save_nifti_seg` from `uroseg.utils.image`; `resample_seg_to_image` from `uroseg.tools.transform_seg2image`; `keep_largest_component` from `uroseg.tools.largest_component`; `run_predict_array` from `uroseg.nnunet.helpers` (lazy)
- Produces:
  - `SegModel.predict_image(self, img: Image, **kwargs) → Image` — abstract, raises `NotImplementedError`
  - `SegModel.predict(self, input: Path | str, output_dir: Path | str, iso: bool = False, **kwargs) → Path` — IO wrapper
  - `NNUNetSegModel.predict_image(self, img: Image, fold: int = 0, device: str = 'cuda') → Image` — full pipeline

- [ ] **Step 1: Write the failing tests**

In `tests/test_models.py`:

**a) Replace `test_segmodel_predict_raises`** (current test breaks because `predict()` now calls `Image.load()` before `predict_image()`) with two tests:

Remove this entire function from `test_models.py`:
```python
def test_segmodel_predict_raises():
    from uroseg.models.base import SegModel
    class M(SegModel):
        name = 'x'; description = ''; weights_url = ''; labels = {}
    with pytest.raises(NotImplementedError):
        M().predict(Path('x.nii.gz'), Path('/tmp'))
```

Add these replacements:

```python
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
        M().predict_image(img)


def test_segmodel_predict_raises_via_predict_image(tmp_path):
    """predict() delegates to predict_image(); NotImplementedError propagates."""
    import numpy as np
    import nibabel as nib
    from uroseg.models.base import SegModel
    inp = tmp_path / 'a.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((5, 5, 5), dtype=np.float32), np.eye(4)), str(inp))
    class M(SegModel):
        name = 'x'; description = ''; weights_url = ''; labels = {}
    with pytest.raises(NotImplementedError):
        M().predict(inp, tmp_path / 'out')
```

**b) Add IO wrapper tests:**

```python
def test_segmodel_predict_io_wrapper_iso_false(tmp_path):
    """predict() with iso=False transforms seg back to original space."""
    import numpy as np
    import nibabel as nib
    from uroseg.models.base import SegModel
    from uroseg.utils.image import Image

    # 2 mm isotropic input image, shape (8, 8, 8)
    data = np.ones((8, 8, 8), dtype=np.float32)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    inp = tmp_path / 'input.nii.gz'
    nib.save(nib.Nifti1Image(data, affine), str(inp))

    class M(SegModel):
        name = 'x'; description = ''; weights_url = ''; labels = {}
        def predict_image(self, img, **kwargs):
            # Simulate model that returns seg in 1 mm iso space
            resampled = img.resample((1.0, 1.0, 1.0))
            seg_data = np.zeros_like(resampled.data, dtype=np.uint8)
            seg_data[5:11, 5:11, 5:11] = 1
            return Image(seg_data, resampled.affine, resampled.header)

    out_path = M().predict(inp, tmp_path / 'out', iso=False)
    assert out_path.exists()
    assert out_path.name == 'input.nii.gz'
    result = Image.load(out_path)
    # iso=False: affine diagonal should be ~2 mm (original spacing)
    np.testing.assert_allclose(np.abs(result.affine[0, 0]), 2.0, atol=0.2)


def test_segmodel_predict_io_wrapper_iso_true(tmp_path):
    """predict() with iso=True leaves seg in the model's working space."""
    import numpy as np
    import nibabel as nib
    from uroseg.models.base import SegModel
    from uroseg.utils.image import Image

    data = np.ones((8, 8, 8), dtype=np.float32)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    inp = tmp_path / 'input.nii.gz'
    nib.save(nib.Nifti1Image(data, affine), str(inp))

    class M(SegModel):
        name = 'x'; description = ''; weights_url = ''; labels = {}
        def predict_image(self, img, **kwargs):
            resampled = img.resample((1.0, 1.0, 1.0))
            seg_data = np.zeros_like(resampled.data, dtype=np.uint8)
            return Image(seg_data, resampled.affine, resampled.header)

    out_path = M().predict(inp, tmp_path / 'out', iso=True)
    assert out_path.exists()
    result = Image.load(out_path)
    # iso=True: stays in 1 mm space — diagonal should be ~1 mm
    np.testing.assert_allclose(np.abs(result.affine[0, 0]), 1.0, atol=0.2)


def test_nnunet_predict_image_runs_pipeline(tmp_path):
    """NNUNetSegModel.predict_image reorients, resamples, calls run_predict_array, keeps largest CC."""
    import numpy as np
    import nibabel as nib
    from unittest.mock import patch
    from uroseg.models.base import NNUNetSegModel
    from uroseg.utils.image import Image

    class TestModel(NNUNetSegModel):
        name = 'test'; description = ''; weights_url = ''
        labels = {}; nnunet_task = 'Dataset999_Test'

    # 2 mm isotropic input
    data = np.ones((8, 8, 8), dtype=np.float32)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    img = Image(data=data, affine=affine, header=nib.Nifti1Image(data, affine).header)

    def fake_run_predict_array(model_dir, img_1mm, fold, device):
        # Return a seg with same shape as 1 mm resampled input
        seg = np.zeros(img_1mm.data.shape, dtype=np.uint8)
        # Two disconnected blobs — keep_largest_component will reduce to one
        mid = np.array(seg.shape) // 2
        seg[mid[0]-2:mid[0]+2, mid[1]-2:mid[1]+2, mid[2]-2:mid[2]+2] = 1
        seg[0:2, 0:2, 0:2] = 1  # small blob that should be removed
        return Image(data=seg, affine=img_1mm.affine, header=img_1mm.header)

    with patch('uroseg.models.base._find_model_dir', return_value=tmp_path), \
         patch('uroseg.models.base._resolve_data_dir', return_value=tmp_path), \
         patch('uroseg.nnunet.helpers.run_predict_array', side_effect=fake_run_predict_array):
        result = TestModel().predict_image(img, fold=0, device='cpu')

    assert isinstance(result, Image)
    assert result.data.ndim == 3
    # After keep_largest_component, only one blob remains — corner blob gone
    assert result.data[0, 0, 0] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_models.py::test_segmodel_predict_image_raises_not_implemented \
       tests/test_models.py::test_segmodel_predict_raises_via_predict_image \
       tests/test_models.py::test_segmodel_predict_io_wrapper_iso_false \
       tests/test_models.py::test_segmodel_predict_io_wrapper_iso_true \
       tests/test_models.py::test_nnunet_predict_image_runs_pipeline -v
```
Expected: 5 FAILs (AttributeError or similar — `predict_image` doesn't exist yet)

- [ ] **Step 3: Update `uroseg/models/base.py`**

**3a. Add module-level imports** (after existing imports at the top of `base.py`):

```python
from uroseg.utils.image import Image, save_nifti_seg
from uroseg.tools.transform_seg2image import resample_seg_to_image
from uroseg.tools.largest_component import keep_largest_component
```

**3b. Replace `SegModel.predict()` and add `SegModel.predict_image()`.**

Replace the current `SegModel.predict()`:
```python
    def predict(self, input: Path, output_dir: Path, **kwargs) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement predict()")
```

With:
```python
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
```

**3c. Replace `NNUNetSegModel.predict()` with `NNUNetSegModel.predict_image()`.**

Replace the current `NNUNetSegModel.predict()`:
```python
    def predict(self, input: Path, output_dir: Path,
                fold: int = 0, device: str = 'cuda', **kwargs) -> None:
        from uroseg.nnunet.helpers import run_predict
        model_dir = _find_model_dir(self.name, _resolve_data_dir())
        run_predict(model_dir, [Path(input)], Path(output_dir), fold=fold, device=device)
```

With:
```python
    def predict_image(self, img: Image,
                      fold: int = 0, device: str = 'cuda') -> Image:
        """Reorient → 1 mm iso → nnunet → largest_component. Returns seg in 1 mm canonical space."""
        from uroseg.nnunet.helpers import run_predict_array
        img_canon = img.as_canonical()
        img_1mm = img_canon.resample((1.0, 1.0, 1.0))
        model_dir = _find_model_dir(self.name, _resolve_data_dir())
        seg = run_predict_array(model_dir, img_1mm, fold=fold, device=device)
        seg.data = keep_largest_component(seg.data)
        return seg
```

- [ ] **Step 4: Run new tests to verify they pass**

```
pytest tests/test_models.py::test_segmodel_predict_image_raises_not_implemented \
       tests/test_models.py::test_segmodel_predict_raises_via_predict_image \
       tests/test_models.py::test_segmodel_predict_io_wrapper_iso_false \
       tests/test_models.py::test_segmodel_predict_io_wrapper_iso_true \
       tests/test_models.py::test_nnunet_predict_image_runs_pipeline -v
```
Expected: 5 PASSes

- [ ] **Step 5: Run full suite**

```
pytest tests/ -q
```
Expected: all tests pass. If `test_segmodel_predict_raises` is still present, it will now fail (because `predict()` calls `Image.load()` first and raises `FileNotFoundError`, not `NotImplementedError`). That test was deleted in Step 1 — verify it is gone.

- [ ] **Step 6: Commit**

```bash
git add uroseg/models/base.py tests/test_models.py
git commit -m "feat: SegModel.predict_image abstract; predict() IO wrapper with iso flag; NNUNetSegModel pipeline"
```
