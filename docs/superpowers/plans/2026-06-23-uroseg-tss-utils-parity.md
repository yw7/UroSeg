# TSS Utils Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align UroSeg's crop, reorient, and largest_component commands with TotalSpineSeg behavior, centralising logic in the `Image` class.

**Architecture:** New `Image.crop_to_seg()` and `Image.as_canonical()` methods keep command files thin. `crop_image2seg` drops seg output entirely (image only). `reorient_canonical` drops `--orientation` and always uses nibabel's canonical reorientation. `largest_component` makes the 6-connectivity dilation structure explicit.

**Tech Stack:** Python, nibabel, scipy.ndimage, pytest, tqdm.

## Global Constraints

- All image saves must go through `save_nifti_image` or `save_nifti_seg` (no raw `nib.save()` in command files)
- Segmentation outputs always uint8
- All commands keep `-r/--overwrite`, `-w/--max-workers`, `-q/--quiet` via `add_common_args`
- Run `pytest tests/ -q` after each task; all 105+ tests must pass before committing

---

### Task 1: Add `Image.crop_to_seg()` and `Image.as_canonical()` to `image.py`

**Files:**
- Modify: `uroseg/utils/image.py`
- Modify: `tests/test_image.py`

**Interfaces:**
- Produces: `Image.crop_to_seg(seg: Image, margin: int = 0) -> Image`
- Produces: `Image.as_canonical() -> Image`

- [ ] **Step 1: Write failing tests for `crop_to_seg`**

Add to the bottom of `tests/test_image.py`:

```python
def test_crop_to_seg_reduces_shape():
    data = np.zeros((30, 30, 30), dtype=np.int16)
    img = Image(data, np.eye(4), nib.Nifti1Header())
    seg_data = np.zeros((30, 30, 30), dtype=np.uint8)
    seg_data[10:20, 10:20, 10:20] = 1
    seg = Image(seg_data, np.eye(4), nib.Nifti1Header())
    cropped = img.crop_to_seg(seg)
    assert cropped.data.shape == (10, 10, 10)


def test_crop_to_seg_all_zero_returns_original():
    data = np.ones((20, 20, 20), dtype=np.int16)
    img = Image(data, np.eye(4), nib.Nifti1Header())
    seg = Image(np.zeros((20, 20, 20), dtype=np.uint8), np.eye(4), nib.Nifti1Header())
    cropped = img.crop_to_seg(seg)
    assert cropped.data.shape == (20, 20, 20)


def test_crop_to_seg_margin():
    data = np.zeros((30, 30, 30), dtype=np.int16)
    img = Image(data, np.eye(4), nib.Nifti1Header())
    seg_data = np.zeros((30, 30, 30), dtype=np.uint8)
    seg_data[10:20, 10:20, 10:20] = 1  # 10-voxel cube
    seg = Image(seg_data, np.eye(4), nib.Nifti1Header())
    cropped = img.crop_to_seg(seg, margin=2)
    assert cropped.data.shape == (14, 14, 14)


def test_crop_to_seg_updates_affine():
    import pytest
    data = np.zeros((30, 30, 30), dtype=np.int16)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    img = Image(data, affine, nib.Nifti1Header())
    seg_data = np.zeros((30, 30, 30), dtype=np.uint8)
    seg_data[5:10, 5:10, 5:10] = 1  # starts at voxel 5 → world coord 5*2=10mm
    seg = Image(seg_data, affine, nib.Nifti1Header())
    cropped = img.crop_to_seg(seg)
    assert cropped.affine[0, 3] == pytest.approx(10.0)
    assert cropped.affine[1, 3] == pytest.approx(10.0)
    assert cropped.affine[2, 3] == pytest.approx(10.0)


def test_as_canonical_returns_image():
    data = np.ones((10, 10, 10), dtype=np.int16)
    img = Image(data, np.eye(4), nib.Nifti1Header())
    canonical = img.as_canonical()
    assert isinstance(canonical, Image)
    assert canonical.data.ndim == 3
    assert canonical.data.shape == (10, 10, 10)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_image.py -q -k "crop_to_seg or as_canonical"
```
Expected: 5 failures with `AttributeError: 'Image' object has no attribute 'crop_to_seg'`

- [ ] **Step 3: Implement `Image.crop_to_seg()` and `Image.as_canonical()`**

In `uroseg/utils/image.py`, add these two methods to the `Image` class (after `bounding_box`):

```python
    def crop_to_seg(self, seg: 'Image', margin: int = 0) -> 'Image':
        seg_data = np.round(seg.data).astype(np.uint8)
        coords = np.argwhere(seg_data != 0)
        if len(coords) == 0:
            return self.copy()
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        shape = np.array(seg_data.shape)
        lo = np.maximum(mins - margin, 0)
        hi = np.minimum(maxs + margin, shape - 1)
        slices = tuple(slice(int(lo[i]), int(hi[i]) + 1) for i in range(3))
        nib_img = nib.Nifti1Image(self.data, self.affine, self.header)
        cropped = nib_img.slicer[slices]
        return Image(np.asanyarray(cropped.dataobj), cropped.affine, cropped.header)

    def as_canonical(self) -> 'Image':
        nib_img = nib.Nifti1Image(self.data, self.affine, self.header)
        canonical = nib.as_closest_canonical(nib_img)
        return Image(np.asanyarray(canonical.dataobj), canonical.affine, canonical.header)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_image.py -q
```
Expected: all pass

- [ ] **Step 5: Run full suite**

```bash
pytest tests/ -q
```
Expected: 105+ passed

- [ ] **Step 6: Commit**

```bash
git add uroseg/utils/image.py tests/test_image.py
git commit -m "feat: Image.crop_to_seg() and Image.as_canonical() methods"
```

---

### Task 2: Rewrite `crop_image2seg.py` — image output only

**Files:**
- Rewrite: `uroseg/commands/crop_image2seg.py`
- Modify: `tests/test_commands.py` (crop section, lines ~411–492)
- Modify: `README.md`

**Interfaces:**
- Consumes: `Image.crop_to_seg(seg, margin)` from Task 1
- Consumes: `save_nifti_image`, `add_common_args`, `collect_niftis`, `build_output_path` from existing utils

- [ ] **Step 1: Update crop tests to match new interface**

In `tests/test_commands.py`, replace the entire crop section (from `# ── crop_image2seg` to `test_crop_cli`):

```python
# ── crop_image2seg ────────────────────────────────────────────────────────────

def test_crop_to_seg_reduces_image(img_file, seg_file, tmp_path):
    """crop_to_seg returns cropped image smaller than input."""
    img = Image.load(img_file)
    seg = Image.load(seg_file)
    cropped = img.crop_to_seg(seg)
    assert cropped.data.shape[0] <= img.data.shape[0]


def test_crop_margin_expands_bbox(img_file, seg_file, tmp_path):
    """margin N expands the bounding box by N voxels on each side."""
    img = Image.load(img_file)
    seg = Image.load(seg_file)
    no_margin = img.crop_to_seg(seg, margin=0)
    with_margin = img.crop_to_seg(seg, margin=2)
    for i in range(3):
        assert with_margin.data.shape[i] >= no_margin.data.shape[i]


def test_crop_margin_clamped_to_image(img_file, seg_file, tmp_path):
    """Very large margin is clamped to image boundary."""
    img = Image.load(img_file)
    seg = Image.load(seg_file)
    cropped = img.crop_to_seg(seg, margin=100)
    for i in range(3):
        assert cropped.data.shape[i] <= img.data.shape[i]


def test_crop_cli_image_only(img_file, seg_file, tmp_path):
    """CLI writes only the image output; no seg file is created."""
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'crop',
         '--img', str(img_file), '--seg', str(seg_file),
         '--out', str(out), '--out-suffix', '_crop'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out / 'img_crop.nii.gz').exists()
    # Seg must NOT be saved
    assert not (out / 'seg_crop.nii.gz').exists()


def test_crop_margin_cli(img_file, seg_file, tmp_path):
    """CLI --margin flag is accepted and produces valid output."""
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'crop',
         '--img', str(img_file), '--seg', str(seg_file),
         '--out', str(out), '--out-suffix', '_crop', '--margin', '2'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out / 'img_crop.nii.gz').exists()
```

- [ ] **Step 2: Run crop tests to confirm they fail**

```bash
pytest tests/test_commands.py -q -k "crop"
```
Expected: failures on `test_crop_cli_image_only` (wrong arg names) and import errors on removed `crop_to_seg`

- [ ] **Step 3: Rewrite `uroseg/commands/crop_image2seg.py`**

Replace the file contents entirely:

```python
from __future__ import annotations
import argparse
import functools
import sys
from pathlib import Path

from tqdm.contrib.concurrent import process_map

from uroseg.utils.image import Image, save_nifti_image
from uroseg.utils.utils import add_common_args, collect_niftis, build_output_path


def process_one(triple: tuple[Path, Path, Path], args: argparse.Namespace) -> None:
    img_in, seg_in, img_out = triple
    img = Image.load(img_in)
    seg = Image.load(seg_in)
    cropped = img.crop_to_seg(seg, margin=args.margin)
    img_out.parent.mkdir(parents=True, exist_ok=True)
    save_nifti_image(cropped.data, cropped.affine, cropped.header, str(img_out))


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Crop image to the bounding box of a segmentation.'
    )
    parser.add_argument('--img', '-i', required=True, help='Input image file or folder')
    parser.add_argument('--seg', '-s', required=True, help='Input seg file or folder')
    parser.add_argument('--out', '-o', required=True, help='Output folder')
    parser.add_argument('--out-suffix', default='_crop', help='Output filename suffix')
    parser.add_argument('--out-prefix', default='', help='Output filename prefix')
    parser.add_argument('--margin', '-m', type=int, default=0, metavar='N',
                        help='Voxels of margin to add around bounding box (default: 0)')
    add_common_args(parser)
    args = parser.parse_args()

    imgs = collect_niftis(args.img)
    segs = collect_niftis(args.seg)

    if len(imgs) != len(segs):
        print(f"Mismatch: {len(imgs)} images vs {len(segs)} segs.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    triples = [
        (img, seg, build_output_path(img, out_dir, args.out_prefix, args.out_suffix))
        for img, seg in zip(imgs, segs)
        if args.overwrite
        or not build_output_path(img, out_dir, args.out_prefix, args.out_suffix).exists()
    ]

    process_map(
        functools.partial(process_one, args=args),
        triples,
        max_workers=args.max_workers,
        disable=args.quiet,
        desc='uroseg crop',
    )
```

- [ ] **Step 4: Run crop tests to confirm they pass**

```bash
pytest tests/test_commands.py -q -k "crop"
```
Expected: all pass

- [ ] **Step 5: Update README — crop command docs**

In `README.md`, the Utilities section, update the crop line from:

```
uroseg crop                 -i/--img PATH  -s/--seg PATH  --out-img PATH  --out-seg PATH  [-m/--margin N]
```

to:

```
uroseg crop                 -i/--img PATH  -s/--seg PATH  -o/--out PATH  [-m/--margin N]
```

Also update the usage example block — change:
```bash
uroseg crop -i img.nii.gz -s seg.nii.gz --out-img img_crop/ --out-seg seg_crop/
```
to:
```bash
uroseg crop -i img.nii.gz -s seg.nii.gz -o img_crop/
```

- [ ] **Step 6: Run full suite**

```bash
pytest tests/ -q
```
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add uroseg/commands/crop_image2seg.py tests/test_commands.py README.md
git commit -m "feat: crop_image2seg outputs image only, matching TSS; use Image.crop_to_seg()"
```

---

### Task 3: Fix `reorient_canonical.py` — drop `--orientation`, always canonical

**Files:**
- Modify: `uroseg/commands/reorient_canonical.py`
- Modify: `tests/test_commands.py` (reorient section, lines ~262–286)

**Interfaces:**
- Consumes: `Image.as_canonical()` from Task 1

- [ ] **Step 1: Update reorient tests to remove `--orientation`**

In `tests/test_commands.py`, replace the reorient section:

```python
# ── reorient ──────────────────────────────────────────────────────────────────

def test_reorient_produces_output(img_file, tmp_path):
    from uroseg.commands.reorient_canonical import process_one
    import argparse
    out = tmp_path / 'reoriented.nii.gz'
    args = argparse.Namespace(overwrite=True)
    process_one((img_file, out), args)
    assert out.exists()
    img = Image.load(out)
    assert img.data.ndim == 3


def test_reorient_cli(img_file, tmp_path):
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'reorient',
         '--img', str(img_file), '--out', str(out),
         '--out-suffix', '_ras'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out / 'img_ras.nii.gz').exists()
```

- [ ] **Step 2: Run reorient tests to confirm they fail**

```bash
pytest tests/test_commands.py -q -k "reorient"
```
Expected: failures (current `process_one` still requires `args.orientation`)

- [ ] **Step 3: Update `uroseg/commands/reorient_canonical.py`**

Replace the file contents:

```python
from __future__ import annotations
import argparse
import functools
from pathlib import Path

from tqdm.contrib.concurrent import process_map

from uroseg.utils.image import Image, save_nifti_image
from uroseg.utils.utils import add_common_args, build_pairs


def process_one(pair: tuple[Path, Path], args: argparse.Namespace) -> None:
    input_path, output_path = pair
    img = Image.load(input_path)
    img = img.as_canonical()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_nifti_image(img.data, img.affine, img.header, str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Reorient NIfTI images to the closest canonical orientation.'
    )
    parser.add_argument('--img', '-i', required=True, help='Input image file or folder')
    parser.add_argument('--out', '-o', required=True, help='Output file or folder')
    parser.add_argument('--out-suffix', default='_reoriented', help='Output filename suffix')
    parser.add_argument('--out-prefix', default='', help='Output filename prefix')
    add_common_args(parser)
    args = parser.parse_args()

    pairs = build_pairs(args.img, args.out, args.out_suffix, args.out_prefix, args.overwrite)
    process_map(
        functools.partial(process_one, args=args),
        pairs,
        max_workers=args.max_workers,
        disable=args.quiet,
        desc='uroseg reorient',
    )
```

- [ ] **Step 4: Run reorient tests to confirm they pass**

```bash
pytest tests/test_commands.py -q -k "reorient"
```
Expected: all pass

- [ ] **Step 5: Run full suite**

```bash
pytest tests/ -q
```
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add uroseg/commands/reorient_canonical.py tests/test_commands.py
git commit -m "feat: reorient_canonical always uses nib.as_closest_canonical(), drop --orientation"
```

---

### Task 4: `largest_component.py` — explicit 6-connectivity dilation structure

**Files:**
- Modify: `uroseg/commands/largest_component.py`
- Modify: `tests/test_commands.py` (add one regression test)

**Context:** The current code calls `ndi.binary_dilation(mask, iterations=dilate)` which already uses scipy's default 6-connectivity structure. This task makes the structure explicit (matching TSS exactly) and adds a regression test proving that diagonal gaps are not bridged.

- [ ] **Step 1: Add regression test for 6-connectivity dilation**

Add this test to `tests/test_commands.py` in the `largest_component` section (after `test_keep_largest_component_dilate`):

```python
def test_dilate_uses_6conn_not_26conn():
    """dilate=1 uses 6-connectivity: a 1-voxel diagonal gap is NOT bridged.

    Two equal-size blobs separated by a 1-voxel diagonal gap:
    - Under 6-conn dilation (correct): gap not bridged → 2 CCs, one is removed.
    - Under 26-conn dilation (wrong): corner-adjacent dilation bridges gap → 1 CC.
    """
    from uroseg.commands.largest_component import keep_largest_component
    data = np.zeros((25, 25, 25), dtype=np.int16)
    data[3:8, 3:8, 3:8] = 1    # blob A (5x5x5 = 125 voxels)
    data[9:14, 9:14, 9:14] = 1  # blob B (5x5x5 = 125 voxels), 1-voxel diagonal gap
    # Blob A corner: (7,7,7). Blob B corner: (9,9,9). Gap voxel: (8,8,8).
    # 6-conn: dilation from A reaches (8,7,7),(7,8,7),(7,7,8) but NOT (8,8,8).
    # 26-conn: dilation from A would reach (8,8,8) — bridging the diagonal gap.
    result = keep_largest_component(data, dilate=1)
    # With 6-conn: two disconnected CCs of equal size → tie-break keeps blob A (first labeled)
    assert result[5, 5, 5] == 1    # blob A center kept
    assert result[11, 11, 11] == 0  # blob B removed (not merged)
```

- [ ] **Step 2: Run the new test to confirm it passes (verifying current behavior is already correct)**

```bash
pytest tests/test_commands.py -q -k "test_dilate_uses_6conn"
```
Expected: PASS (current code already uses 6-conn via scipy default)

- [ ] **Step 3: Make the dilation structure explicit in `largest_component.py`**

In `uroseg/commands/largest_component.py`, replace the module-level constant and the dilation call:

Remove:
```python
# Full 26-connectivity structure for 3-D images
_STRUCT26 = np.ones((3, 3, 3), dtype=np.int8)
```

Replace with:
```python
# 26-connectivity for CC labeling (matches TSS); 6-connectivity for dilation (matches TSS)
_STRUCT26 = np.ones((3, 3, 3), dtype=np.int8)
```

In `_largest_component_for_label`, replace:
```python
    if dilate > 0:
        work = ndi.binary_dilation(mask, iterations=dilate).astype(np.uint8)
    else:
        work = mask
```

with:
```python
    if dilate > 0:
        struct = ndi.iterate_structure(ndi.generate_binary_structure(3, 1), dilate)
        work = ndi.binary_dilation(mask, structure=struct).astype(np.uint8)
    else:
        work = mask
```

- [ ] **Step 4: Run full suite**

```bash
pytest tests/ -q
```
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add uroseg/commands/largest_component.py tests/test_commands.py
git commit -m "refactor: explicit 6-conn dilation structure in largest_component, matching TSS"
```

---

### Task 5: Push

- [ ] **Step 1: Verify final test count**

```bash
pytest tests/ -q
```
Expected: all pass (count ≥ 110 due to new tests added)

- [ ] **Step 2: Push**

```bash
git push
```
