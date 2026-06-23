# UroSeg Utils TSS Parity

**Date:** 2026-06-23
**Status:** Approved

## Goal

Align UroSeg's utility commands with TotalSpineSeg (TSS) behavior exactly, using UroSeg's existing `Image` abstraction as the home for shared logic.

## Changes

### 1. `image.py` — New/modified `Image` methods

**Add `Image.crop_to_seg(seg: Image, margin: int = 0) -> Image`**
- Computes the bounding box of `seg.data != 0`
- Expands by `margin` voxels, clamped to image shape
- Uses `nib.Nifti1Image.slicer[min_x:max_x+1, ...]` so nibabel handles the affine update internally (matches TSS)
- If seg is all-zeros (no foreground), returns the image unchanged — matches TSS behavior
- Returns a new `Image` containing only the cropped image — no seg output

**Add `Image.as_canonical() -> Image`**
- Wraps `nib.as_closest_canonical(nib_img)` — always reorients to the closest standard axis orientation
- Existing `Image.reorient(orientation)` is kept for internal use (inference pipeline reorients to RAS before prediction)

### 2. `crop_image2seg.py` — Image output only

**CLI args removed:** `--out-img`, `--out-seg`, `--img-suffix`, `--img-prefix`, `--seg-suffix`, `--seg-prefix`
**CLI args added:** `--out/-o` (single output folder), `--out-suffix` (default `_crop`), `--out-prefix`

`process_one` logic:
```
img = Image.load(img_in)
seg = Image.load(seg_in)
cropped = img.crop_to_seg(seg, margin=args.margin)
save_nifti_image(cropped.data, cropped.affine, cropped.header, out_path)
```

Input pairing: `--img -i` (image folder or file), `--seg -s` (seg folder or file), matched by filename.

### 3. `reorient_canonical.py` — Always canonical

**CLI arg removed:** `--orientation`

`process_one` logic:
```
img = Image.load(input_path)
img = img.as_canonical()
save_nifti_image(img.data, img.affine, img.header, output_path)
```

### 4. `largest_component.py` — 6-connectivity dilation

Replace the dilation structure from `np.ones((3,3,3))` with:
```python
ndi.iterate_structure(ndi.generate_binary_structure(3, 1), dilate)
```
This produces a sphere/diamond (face-adjacent only, iterated), matching TSS exactly. The CC labeling step keeps 26-connectivity (`np.ones((3,3,3))`).

### 5. `resample.py` / `transform_seg2image.py` — No changes

`save_nifti_image` already applies:
- Integer dtype rescaling using the same condition as TSS: `(image_min < dtype_min) or (dtype_max < image_max)`
- `set_qform` / `set_sform` from affine
- Preserves original dtype

These match TSS behavior. No changes needed.

## Dtype/affine contract (all image-outputting commands)

Every command that saves an image must use `save_nifti_image` (handles integer rescaling, dtype, qform/sform) or `save_nifti_seg` (always uint8). No raw `nib.save()` calls in command files.

## Tests

- `crop_image2seg`: update arg names; assert only image output written; assert seg file not created; assert output shape matches bounding box + margin
- `reorient_canonical`: remove `--orientation` from test args; assert output has canonical orientation
- `largest_component`: add test that verifies dilation with `dilate=1` uses 6-connectivity (diamond-shaped erosion) rather than full cube

## Out of scope

- `preview_jpg --levels` flag (not relevant for volumetric structures)
- `cpdir` (UroSeg's simpler version is intentional)
- Switching resample/transform_seg2image from nibabel to TorchIO
