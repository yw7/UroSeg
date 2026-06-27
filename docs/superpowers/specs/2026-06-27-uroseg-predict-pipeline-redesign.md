# UroSeg Predict Pipeline Redesign

## Goal

Replace the file-based, tmp-dir prediction pipeline with a clean in-memory
pipeline: single IO per image, one predictor init per batch, suppressed
nnunet output, tqdm progress bar, and a generic model interface that works
for any future segmentation backend (nnunet, MONAI, …).

---

## 1. Problems with the Current Design

- `run_predict_cli` writes reoriented inputs to a tmp dir, calls nnunet's
  file-based `predict_from_files`, then reads outputs back — 4× IO per image.
- nnunet re-initializes the predictor (loads checkpoint) for every call to
  `run_predict_array`, so a directory of N images does N checkpoint loads.
- nnunet floods stdout/stderr: env-var warnings, "Predicting X…",
  "sending off prediction…", per-tile tqdm bars.
- `NNUNetSegModel.predict_image` mixes preprocessing (canonical→1mm),
  inference, and postprocessing — making the sequence invisible to callers.
- No `--iso` flag on the CLI.

---

## 2. Architecture

### 2.1 Model interface (base.py)

Every `SegModel` subclass implements two methods:

```
init_predictor(model_dir, fold, device) → predictor
predict_image(predictor, img_1mm: Image) → Image   ← pure inference, no pre/post
```

`SegModel` adds one class attribute:

```
post_largest_component: bool = False
```

`SegModel.predict()` (single-image Python API) becomes the canonical pipeline:

```
img_orig = Image.load(input)
img_1mm  = img_orig.as_canonical().resample((1.0, 1.0, 1.0))
predictor = self.init_predictor(model_dir, fold, device)
seg       = self.predict_image(predictor, img_1mm)
if self.post_largest_component:
    seg.data = keep_largest_component(seg.data, binarize=True)
if not iso:
    seg = resample_seg_to_image(seg, img_orig)
save_nifti_seg(...)
```

### 2.2 nnunet helpers (helpers.py)

Three new private helpers:

| Helper | Purpose |
|---|---|
| `_suppress_nnunet()` | context manager — sets dummy env vars, redirects stdout/stderr, suppresses warnings |
| `_init_predictor(model_dir, fold, device)` | creates & initializes nnUNetPredictor under suppression; sets `allow_tqdm=False` |
| `_run_inference(predictor, img)` | calls `predict_single_npy_array` under suppression; returns `np.ndarray` |

`run_predict_array` is refactored to use these (stays in helpers.py for the
Python API; no longer called by the CLI).

### 2.3 CLI pipeline (predict.py)

`run_predict_cli` becomes generic — no nnunet-specific code, no tmp dir:

```
predictor = model.init_predictor(model_dir, fold, device)   ← once

tqdm loop:
  img_orig = Image.load(inp)
  img_1mm  = img_orig.as_canonical().resample((1.0, 1.0, 1.0))
  seg      = model.predict_image(predictor, img_1mm)
  if model.post_largest_component:
      seg.data = keep_largest_component(seg.data, binarize=True)
  if not args.iso:
      seg = resample_seg_to_image(seg, img_orig)
  save_nifti_seg(...)
```

`add_inference_args` gains `--iso` (store_true, default False).
`run_predict_cli` loses the `largest_component` parameter.

### 2.4 NNUNetSegModel (base.py)

```python
class NNUNetSegModel(SegModel):
    def init_predictor(self, model_dir, fold=0, device='cuda'):
        from uroseg.nnunet.helpers import _init_predictor
        return _init_predictor(model_dir, fold=fold, device=device)

    def predict_image(self, predictor, img: Image) -> Image:
        from uroseg.nnunet.helpers import _run_inference
        seg_array = _run_inference(predictor, img)
        return Image(data=seg_array, affine=img.affine, header=img.header)
```

`post_largest_component` is NOT set here — each concrete model sets it:

```python
class Prostate(NNUNetSegModel):
    post_largest_component = True
```

---

## 3. Preprocessing Contract

`canonical → 1mm iso` is applied by BOTH `SegModel.predict()` and
`run_predict_cli` before calling `predict_image`. This is the universal
preprocessing contract for all models (nnunet and future MONAI). Each
model's `predict_image` receives an already-canonical, already-1mm image
and performs only inference.

---

## 4. Suppression Details

`_suppress_nnunet()`:
1. Sets `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results` to `''` if
   not already set (prevents "not defined" prints), restores on exit.
2. Redirects `sys.stdout` and `sys.stderr` to a `StringIO` buffer.
3. Wraps in `warnings.catch_warnings()` + `warnings.simplefilter('ignore')`.

Used around:
- `predictor.initialize_from_trained_model_folder(...)` in `_init_predictor`
- `predictor.predict_single_npy_array(...)` in `_run_inference`

`allow_tqdm=False` on the predictor suppresses nnunet's per-tile progress bar.

---

## 5. File Summary

| File | Change |
|---|---|
| `uroseg/nnunet/helpers.py` | Add `_suppress_nnunet`, `_init_predictor`, `_run_inference`; refactor `run_predict_array` |
| `uroseg/models/base.py` | Add `post_largest_component`, `init_predictor`, new `predict_image` signature; rewrite `predict()` |
| `uroseg/nnunet/predict.py` | Rewrite `run_predict_cli`; add `--iso`; remove `largest_component` param; remove `tempfile` |
| `uroseg/models/prostate.py` | Add `post_largest_component = True`; drop `largest_component` kwarg |
| `tests/test_models.py` | Update tests for new `predict_image(predictor, img)` signature |
| `tests/test_inference.py` | Update `run_predict_cli` tests; add `--iso` and suppression tests |

---

## 6. Invariants

- `predict_image(predictor, img)` never touches the filesystem.
- `predict_image` never applies preprocessing (canonical/resample) or postprocessing (largest_component).
- `run_predict_cli` is model-agnostic: no nnunet imports, no tmp dir.
- `_suppress_nnunet` is the single suppression point — not scattered across callers.
- `init_predictor` is called exactly once per CLI invocation regardless of batch size.
