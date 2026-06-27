# UroSeg In-Memory API and Inference Pipeline

## Goal

Expose in-memory (no-IO) data functions in the public API, and replace the current file-based `predict()` with a full pre/post-processing pipeline that operates in-memory and touches the filesystem only once (load input, save output).

---

## 1. In-Memory API Exports

Add the following to `uroseg/__init__.py`. All functions already exist in their respective modules — this is purely additive.

| Export | Source | Signature |
|---|---|---|
| `Image` | `uroseg.utils.image` | data carrier; `.load()`, `.save()`, `.resample()`, `.as_canonical()`, `.crop_to_seg()`, `.reorient()` |
| `keep_largest_component` | `uroseg.tools.largest_component` | `(data: np.ndarray, labels=None, dilate=0, binarize=False) → np.ndarray` |
| `resample_seg_to_image` | `uroseg.tools.transform_seg2image` | `(seg: Image, ref: Image, interpolation='nearest') → Image` |
| `apply_map` | `uroseg.tools.map_labels` | `(data: np.ndarray, mapping: dict, keep_unmapped=False) → np.ndarray` |
| `make_preview` | `uroseg.tools.preview` | `(img_data: np.ndarray, seg_data=None, orient='sag', sliceloc=0.5, label_text_right=None, label_text_left=None) → np.ndarray` |

`Image` is the natural carrier for in-memory pipeline work; exporting it makes `resample_seg_to_image` and other `Image`-typed functions usable without internal imports.

### Usage example

```python
import uroseg

img = uroseg.Image.load("t2.nii.gz")
seg = uroseg.Prostate().predict_image(img, iso=True)   # in-memory, no file writes
seg.data = uroseg.apply_map(seg.data, {1: 0, 2: 1, 3: 1})
rgb = uroseg.make_preview(img.data, seg.data)
```

---

## 2. Model Class Changes

### 2a. `SegModel` — `uroseg/models/base.py`

Add abstract `predict_image()` and rewrite `predict()` as a generic IO wrapper.

```python
class SegModel:
    ...
    def predict_image(self, img: Image, **kwargs) -> Image:
        """Process a single image in memory. Return seg in the model's working space."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement predict_image()")

    def predict(self, input: Path | str, output_dir: Path | str,
                iso: bool = False, **kwargs) -> Path:
        """Load → predict_image → (optional) transform back → save.

        iso=False (default): resample seg back to the original input space/affine.
        iso=True: leave seg in the model's working space (e.g. 1 mm canonical).
        """
        # Image, save_nifti_seg, resample_seg_to_image are module-level imports in base.py
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

`predict_dir()` is unchanged — it already passes `**kwargs` through to `predict()`, so `iso` is forwarded automatically.

### 2b. `NNUNetSegModel` — `uroseg/models/base.py`

Replace the current `predict()` override with `predict_image()`. The base-class `predict()` now handles IO and the iso transform.

```python
class NNUNetSegModel(SegModel):
    nnunet_task: str

    def predict_image(self, img: Image,
                      fold: int = 0, device: str = 'cuda') -> Image:
        """Reorient → 1 mm iso → nnunet → largest_component. Returns seg in 1 mm canonical space."""
        from uroseg.nnunet.helpers import run_predict_array
        from uroseg.tools.largest_component import keep_largest_component

        img_canon = img.as_canonical()
        img_1mm   = img_canon.resample((1.0, 1.0, 1.0))
        model_dir = _find_model_dir(self.name, _resolve_data_dir())
        seg       = run_predict_array(model_dir, img_1mm, fold=fold, device=device)
        seg.data  = keep_largest_component(seg.data)
        return seg
```

Contract: `predict_image()` always returns the seg in the model's internal working space (1 mm canonical for nnunet). The base-class `predict()` is responsible for transforming back to the caller's original space when `iso=False`.

---

## 3. nnunet Helper — `uroseg/nnunet/helpers.py`

Add `run_predict_array()` alongside the existing `run_predict()`.

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
    from uroseg.utils.image import Image

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
    spacing = list(float(s) for s in nib_img.header.get_zooms()[:3])

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

The spacing field uses the `(x, y, z)` values from the nibabel header. For the 1 mm isotropic input this is always `[1.0, 1.0, 1.0]`, making the convention irrelevant. nnunet handles its own internal normalization and resampling to training spacing from there.

---

## 4. File Summary

| File | Change |
|---|---|
| `uroseg/__init__.py` | Add exports: `Image`, `keep_largest_component`, `resample_seg_to_image`, `apply_map`, `make_preview` |
| `uroseg/models/base.py` | Add `SegModel.predict_image()` (abstract); rewrite `SegModel.predict()` as IO+iso wrapper; replace `NNUNetSegModel.predict()` with `NNUNetSegModel.predict_image()` |
| `uroseg/nnunet/helpers.py` | Add `run_predict_array(model_dir, img, fold, device) → Image` |
| `tests/test_models.py` | Tests for `run_predict_array` (mock predictor), `NNUNetSegModel.predict_image`, `SegModel.predict` IO wrapper and iso flag |

No changes to CLI, tools, or `predict_dir`.

---

## 5. Invariants

- `predict_image()` never touches the filesystem.
- `SegModel.predict()` is the only place IO happens for model inference.
- The `iso=False` transform-back (`resample_seg_to_image(seg, img_orig)`) is model-agnostic and lives only on `SegModel`. Future models get it for free.
- A model returning seg already in original space (no internal resampling) passes through `resample_seg_to_image` safely — nibabel's affine-based resampling is a near-no-op when source and target spaces match.
