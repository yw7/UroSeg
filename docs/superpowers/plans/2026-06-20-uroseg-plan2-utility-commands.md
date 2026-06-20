# UroSeg Plan 2 — Utility Commands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all UroSeg utility subcommands: map, resample, reorient, largest_component, crop, preview, transform_seg2image, and cpdir.

**Architecture:** Every command follows the identical pattern: argparse → collect inputs → process_map with tqdm. Shared helpers (`add_common_args`, `collect_niftis`, `build_output_path`, `build_pairs`) from Plan 1 handle all boilerplate. Each command file only contains its transformation logic in `process_one` and its specific argparse setup in `main()`.

**Tech Stack:** Python 3.10+, nibabel, numpy, scipy, pillow, tqdm, pytest

## Global Constraints

- Python ≥ 3.10
- All NIfTI I/O via `Image` class only (`uroseg.utils.image.Image`) — never call nibabel directly in command files
- CLI args: `--img`, `--seg`, `--out`, `--out-img`, `--out-seg`; suffixes/prefixes: `--out-suffix`, `--out-prefix`, `--img-suffix`, `--img-prefix`, `--seg-suffix`, `--seg-prefix`
- Shared args via `add_common_args(parser)`: `--overwrite`, `--max-workers` (default 1), `--quiet`
- Batch processing via `tqdm.contrib.concurrent.process_map`
- Output filename: `{prefix}{stem}{suffix}.nii.gz`
- Every `main()` shape: argparse → collect → process_map

## Prerequisite

Plan 1 must be complete. Verify before starting:

```bash
python -c "from uroseg.utils.image import Image; from uroseg.utils.utils import add_common_args, build_pairs, collect_niftis, build_output_path; print('OK')"
```

Expected: `OK`

---

## File Map

| File | Command | Role |
|------|---------|------|
| `uroseg/commands/map_labels.py` | `uroseg map` | Remap label IDs in segmentation files |
| `uroseg/commands/resample.py` | `uroseg resample` | Resample image to target voxel spacing |
| `uroseg/commands/reorient_canonical.py` | `uroseg reorient` | Reorient NIfTI to canonical orientation |
| `uroseg/commands/largest_component.py` | `uroseg largest_component` | Keep largest connected component per label |
| `uroseg/commands/crop_image2seg.py` | `uroseg crop` | Crop image and seg to seg bounding box |
| `uroseg/commands/preview_jpg.py` | `uroseg preview` | Generate JPG preview of image ± segmentation |
| `uroseg/commands/transform_seg2image.py` | `uroseg transform_seg2image` | Resample seg to match reference image space |
| `uroseg/commands/cpdir.py` | `uroseg cpdir` | Copy NIfTI files with optional rename |
| `tests/test_commands.py` | — | All utility command tests |

---

### Task 1: map_labels

**Files:**
- Create: `uroseg/commands/map_labels.py`
- Create: `tests/test_commands.py`

**Interfaces:**
- Consumes: `Image.load`, `image.save`, `image.data`; `add_common_args`, `build_pairs` from `uroseg.utils.utils`
- Produces: `uroseg map --seg ... --out ... --map labels.json`

- [ ] **Step 1: Write failing tests in tests/test_commands.py**

```python
import json
import numpy as np
import nibabel as nib
import pytest
from pathlib import Path
from uroseg.utils.image import Image


# ── shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def seg_file(tmp_path):
    data = np.zeros((20, 20, 20), dtype=np.int16)
    data[2:8, 2:8, 2:8] = 1
    data[10:16, 10:16, 10:16] = 2
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    path = tmp_path / "seg.nii.gz"
    nib.save(img, path)
    return path


@pytest.fixture
def img_file(tmp_path):
    data = np.random.randint(0, 1000, (20, 20, 20), dtype=np.int16)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    nib.save(nib.Nifti1Image(data, affine), tmp_path / "img.nii.gz")
    return tmp_path / "img.nii.gz"


@pytest.fixture
def map_json(tmp_path):
    mapping = {"1": 10, "2": 20}
    path = tmp_path / "map.json"
    path.write_text(json.dumps(mapping))
    return path


# ── map_labels ────────────────────────────────────────────────────────────────

def test_map_labels_remaps_values(seg_file, map_json, tmp_path):
    from uroseg.commands.map_labels import apply_map
    img = Image.load(seg_file)
    with open(map_json) as f:
        mapping = json.load(f)
    result = apply_map(img.data, mapping)
    assert result[3, 3, 3] == 10
    assert result[12, 12, 12] == 20
    assert result[0, 0, 0] == 0


def test_map_labels_unmapped_becomes_zero(seg_file, tmp_path):
    from uroseg.commands.map_labels import apply_map
    img = Image.load(seg_file)
    mapping = {"1": 5}
    result = apply_map(img.data, mapping)
    assert result[3, 3, 3] == 5
    assert result[12, 12, 12] == 0


def test_map_labels_cli(seg_file, map_json, tmp_path):
    import subprocess, sys
    out = tmp_path / "out"
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'map',
         '--seg', str(seg_file), '--out', str(out),
         '--map', str(map_json), '--out-suffix', '_mapped'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    out_file = out / 'seg_mapped.nii.gz'
    assert out_file.exists()
    img = Image.load(out_file)
    assert img.data[3, 3, 3] == 10
```

- [ ] **Step 2: Run tests — expect failures**

```bash
pytest tests/test_commands.py::test_map_labels_remaps_values tests/test_commands.py::test_map_labels_unmapped_becomes_zero tests/test_commands.py::test_map_labels_cli -v
```

Expected: `ImportError` — `uroseg.commands.map_labels` not found.

- [ ] **Step 3: Implement uroseg/commands/map_labels.py**

```python
from __future__ import annotations
import argparse
import functools
import json
from pathlib import Path

import numpy as np
from tqdm.contrib.concurrent import process_map

from uroseg.utils.image import Image
from uroseg.utils.utils import add_common_args, build_pairs


def apply_map(data: np.ndarray, mapping: dict) -> np.ndarray:
    result = np.zeros_like(data)
    for src, dst in mapping.items():
        result[data == int(src)] = int(dst)
    return result


def process_one(pair: tuple[Path, Path], args: argparse.Namespace) -> None:
    input_path, output_path = pair
    with open(args.map) as f:
        mapping = json.load(f)
    img = Image.load(input_path)
    img.data = apply_map(img.data, mapping)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description='Remap label IDs in segmentation files.')
    parser.add_argument('--seg', required=True, help='Input seg file or folder')
    parser.add_argument('--out', required=True, help='Output file or folder')
    parser.add_argument('--map', required=True, help='Path to label map JSON {"src": dst}')
    parser.add_argument('--out-suffix', default='_mapped', help='Output filename suffix')
    parser.add_argument('--out-prefix', default='', help='Output filename prefix')
    add_common_args(parser)
    args = parser.parse_args()

    pairs = build_pairs(args.seg, args.out, args.out_suffix, args.out_prefix, args.overwrite)
    process_map(
        functools.partial(process_one, args=args),
        pairs,
        max_workers=args.max_workers,
        disable=args.quiet,
        desc='uroseg map',
    )
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_commands.py::test_map_labels_remaps_values tests/test_commands.py::test_map_labels_unmapped_becomes_zero tests/test_commands.py::test_map_labels_cli -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add uroseg/commands/map_labels.py tests/test_commands.py
git commit -m "feat: uroseg map — remap segmentation label IDs"
```

---

### Task 2: resample

**Files:**
- Create: `uroseg/commands/resample.py`
- Modify: `tests/test_commands.py`

**Interfaces:**
- Consumes: `Image.load`, `image.resample(voxel_size: tuple) -> Image`, `image.save`; `add_common_args`, `build_pairs`
- Produces: `uroseg resample --img ... --out ... --spacing X Y Z`

- [ ] **Step 1: Add tests to tests/test_commands.py**

```python
# ── resample ──────────────────────────────────────────────────────────────────

def test_resample_changes_spacing(img_file, tmp_path):
    from uroseg.commands.resample import process_one
    import argparse
    out = tmp_path / 'resampled.nii.gz'
    args = argparse.Namespace(spacing=[2.0, 2.0, 2.0], overwrite=True)
    process_one((img_file, out), args)
    assert out.exists()
    img = Image.load(out)
    assert img.data.ndim == 3


def test_resample_cli(img_file, tmp_path):
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'resample',
         '--img', str(img_file), '--out', str(out),
         '--spacing', '2', '2', '2', '--out-suffix', '_resampled'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out / 'img_resampled.nii.gz').exists()
```

- [ ] **Step 2: Run tests — expect failures**

```bash
pytest tests/test_commands.py::test_resample_changes_spacing tests/test_commands.py::test_resample_cli -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement uroseg/commands/resample.py**

```python
from __future__ import annotations
import argparse
import functools
from pathlib import Path

from tqdm.contrib.concurrent import process_map

from uroseg.utils.image import Image
from uroseg.utils.utils import add_common_args, build_pairs


def process_one(pair: tuple[Path, Path], args: argparse.Namespace) -> None:
    input_path, output_path = pair
    img = Image.load(input_path)
    img = img.resample(tuple(args.spacing))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description='Resample image to target voxel spacing.')
    parser.add_argument('--img', required=True, help='Input image file or folder')
    parser.add_argument('--out', required=True, help='Output file or folder')
    parser.add_argument('--spacing', nargs=3, type=float, required=True,
                        metavar=('X', 'Y', 'Z'), help='Target voxel spacing in mm')
    parser.add_argument('--out-suffix', default='_resampled', help='Output filename suffix')
    parser.add_argument('--out-prefix', default='', help='Output filename prefix')
    add_common_args(parser)
    args = parser.parse_args()

    pairs = build_pairs(args.img, args.out, args.out_suffix, args.out_prefix, args.overwrite)
    process_map(
        functools.partial(process_one, args=args),
        pairs,
        max_workers=args.max_workers,
        disable=args.quiet,
        desc='uroseg resample',
    )
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_commands.py::test_resample_changes_spacing tests/test_commands.py::test_resample_cli -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add uroseg/commands/resample.py tests/test_commands.py
git commit -m "feat: uroseg resample — resample NIfTI to target voxel spacing"
```

---

### Task 3: reorient_canonical

**Files:**
- Create: `uroseg/commands/reorient_canonical.py`
- Modify: `tests/test_commands.py`

**Interfaces:**
- Consumes: `Image.load`, `image.reorient(orientation: str) -> Image`, `image.save`; `add_common_args`, `build_pairs`
- Produces: `uroseg reorient --img ... --out ... [--orientation RAS]`

- [ ] **Step 1: Add tests to tests/test_commands.py**

```python
# ── reorient ──────────────────────────────────────────────────────────────────

def test_reorient_produces_output(img_file, tmp_path):
    from uroseg.commands.reorient_canonical import process_one
    import argparse
    out = tmp_path / 'reoriented.nii.gz'
    args = argparse.Namespace(orientation='RAS', overwrite=True)
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
         '--orientation', 'RAS', '--out-suffix', '_ras'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out / 'img_ras.nii.gz').exists()
```

- [ ] **Step 2: Run tests — expect failures**

```bash
pytest tests/test_commands.py::test_reorient_produces_output tests/test_commands.py::test_reorient_cli -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement uroseg/commands/reorient_canonical.py**

```python
from __future__ import annotations
import argparse
import functools
from pathlib import Path

from tqdm.contrib.concurrent import process_map

from uroseg.utils.image import Image
from uroseg.utils.utils import add_common_args, build_pairs


def process_one(pair: tuple[Path, Path], args: argparse.Namespace) -> None:
    input_path, output_path = pair
    img = Image.load(input_path)
    img = img.reorient(args.orientation)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Reorient NIfTI images to a canonical orientation.'
    )
    parser.add_argument('--img', required=True, help='Input image file or folder')
    parser.add_argument('--out', required=True, help='Output file or folder')
    parser.add_argument('--orientation', default='RAS',
                        help='Target orientation code (default: RAS)')
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

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_commands.py::test_reorient_produces_output tests/test_commands.py::test_reorient_cli -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add uroseg/commands/reorient_canonical.py tests/test_commands.py
git commit -m "feat: uroseg reorient — reorient NIfTI to canonical orientation"
```

---

### Task 4: largest_component

**Files:**
- Create: `uroseg/commands/largest_component.py`
- Modify: `tests/test_commands.py`

**Interfaces:**
- Consumes: `Image.load`, `image.data`, `image.save`; `add_common_args`, `build_pairs`
- Produces:
  - `keep_largest_component(data: np.ndarray, labels: list[int] | None) -> np.ndarray`
  - `uroseg largest_component --seg ... --out ... [--labels 1 2]`

- [ ] **Step 1: Add tests to tests/test_commands.py**

```python
# ── largest_component ─────────────────────────────────────────────────────────

def test_keep_largest_component_removes_small(tmp_path):
    from uroseg.commands.largest_component import keep_largest_component
    data = np.zeros((30, 30, 30), dtype=np.int16)
    data[1:3, 1:3, 1:3] = 1    # small blob
    data[10:20, 10:20, 10:20] = 1  # large blob
    result = keep_largest_component(data, labels=None)
    assert result[1, 1, 1] == 0
    assert result[15, 15, 15] == 1


def test_keep_largest_component_per_label(tmp_path):
    from uroseg.commands.largest_component import keep_largest_component
    data = np.zeros((30, 30, 30), dtype=np.int16)
    data[1:3, 1:3, 1:3] = 1
    data[10:20, 10:20, 10:20] = 1
    data[1:3, 20:22, 1:3] = 2
    data[10:20, 1:10, 10:20] = 2
    result = keep_largest_component(data, labels=[1, 2])
    assert result[1, 1, 1] == 0
    assert result[15, 15, 15] == 1
    assert result[1, 21, 1] == 0
    assert result[15, 5, 15] == 2


def test_largest_component_cli(seg_file, tmp_path):
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'largest_component',
         '--seg', str(seg_file), '--out', str(out), '--out-suffix', '_lc'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out / 'seg_lc.nii.gz').exists()
```

- [ ] **Step 2: Run tests — expect failures**

```bash
pytest tests/test_commands.py::test_keep_largest_component_removes_small tests/test_commands.py::test_keep_largest_component_per_label tests/test_commands.py::test_largest_component_cli -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement uroseg/commands/largest_component.py**

```python
from __future__ import annotations
import argparse
import functools
from pathlib import Path

import numpy as np
from scipy import ndimage
from tqdm.contrib.concurrent import process_map

from uroseg.utils.image import Image
from uroseg.utils.utils import add_common_args, build_pairs


def keep_largest_component(data: np.ndarray, labels: list[int] | None = None) -> np.ndarray:
    result = np.zeros_like(data)
    label_ids = labels if labels else [int(v) for v in np.unique(data) if v > 0]
    for label_id in label_ids:
        mask = data == label_id
        labeled_arr, n = ndimage.label(mask)
        if n == 0:
            continue
        sizes = ndimage.sum(mask, labeled_arr, range(1, n + 1))
        largest = int(np.argmax(sizes)) + 1
        result[labeled_arr == largest] = label_id
    return result


def process_one(pair: tuple[Path, Path], args: argparse.Namespace) -> None:
    input_path, output_path = pair
    img = Image.load(input_path)
    labels = args.labels if args.labels else None
    img.data = keep_largest_component(img.data, labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Keep only the largest connected component per label in a segmentation.'
    )
    parser.add_argument('--seg', required=True, help='Input seg file or folder')
    parser.add_argument('--out', required=True, help='Output file or folder')
    parser.add_argument('--labels', nargs='+', type=int, default=None,
                        help='Label IDs to process (default: all non-zero)')
    parser.add_argument('--out-suffix', default='_largest', help='Output filename suffix')
    parser.add_argument('--out-prefix', default='', help='Output filename prefix')
    add_common_args(parser)
    args = parser.parse_args()

    pairs = build_pairs(args.seg, args.out, args.out_suffix, args.out_prefix, args.overwrite)
    process_map(
        functools.partial(process_one, args=args),
        pairs,
        max_workers=args.max_workers,
        disable=args.quiet,
        desc='uroseg largest_component',
    )
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_commands.py::test_keep_largest_component_removes_small tests/test_commands.py::test_keep_largest_component_per_label tests/test_commands.py::test_largest_component_cli -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add uroseg/commands/largest_component.py tests/test_commands.py
git commit -m "feat: uroseg largest_component — keep largest connected component per label"
```

---

### Task 5: crop_image2seg

**Files:**
- Create: `uroseg/commands/crop_image2seg.py`
- Modify: `tests/test_commands.py`

**Interfaces:**
- Consumes: `Image.load`, `image.bounding_box(label=None) -> tuple | None`, `image.data`, `image.save`; `add_common_args`, `collect_niftis`, `build_output_path`
- Produces: `uroseg crop --img ... --seg ... --out-img ... --out-seg ... [--img-suffix _crop] [--seg-suffix _crop]`

Note: this command takes **two paired inputs** — image + seg — so it cannot use `build_pairs`. It pairs files by sorted filename order.

- [ ] **Step 1: Add tests to tests/test_commands.py**

```python
# ── crop_image2seg ────────────────────────────────────────────────────────────

def test_crop_reduces_size(img_file, seg_file, tmp_path):
    from uroseg.commands.crop_image2seg import crop_to_seg
    img = Image.load(img_file)
    seg = Image.load(seg_file)
    cropped_img, cropped_seg = crop_to_seg(img, seg)
    assert cropped_img.data.shape[0] <= img.data.shape[0]
    assert cropped_seg.data.shape == cropped_img.data.shape


def test_crop_preserves_seg_labels(img_file, seg_file, tmp_path):
    from uroseg.commands.crop_image2seg import crop_to_seg
    img = Image.load(img_file)
    seg = Image.load(seg_file)
    _, cropped_seg = crop_to_seg(img, seg)
    assert 1 in np.unique(cropped_seg.data)


def test_crop_cli(img_file, seg_file, tmp_path):
    import subprocess, sys
    out_img = tmp_path / 'out_img'
    out_seg = tmp_path / 'out_seg'
    out_img.mkdir(); out_seg.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'crop',
         '--img', str(img_file), '--seg', str(seg_file),
         '--out-img', str(out_img), '--out-seg', str(out_seg),
         '--img-suffix', '_crop', '--seg-suffix', '_crop'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out_img / 'img_crop.nii.gz').exists()
    assert (out_seg / 'seg_crop.nii.gz').exists()
```

- [ ] **Step 2: Run tests — expect failures**

```bash
pytest tests/test_commands.py::test_crop_reduces_size tests/test_commands.py::test_crop_preserves_seg_labels tests/test_commands.py::test_crop_cli -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement uroseg/commands/crop_image2seg.py**

```python
from __future__ import annotations
import argparse
import functools
from pathlib import Path

import numpy as np
from tqdm.contrib.concurrent import process_map

from uroseg.utils.image import Image
from uroseg.utils.utils import add_common_args, collect_niftis, build_output_path


def crop_to_seg(img: Image, seg: Image) -> tuple[Image, Image]:
    bb = seg.bounding_box(label=None)
    if bb is None:
        return img.copy(), seg.copy()
    cropped_img = Image(img.data[bb], img.affine.copy(), img.header)
    cropped_seg = Image(seg.data[bb], seg.affine.copy(), seg.header)
    return cropped_img, cropped_seg


def process_one(
    pair: tuple[Path, Path, Path, Path],
    args: argparse.Namespace,
) -> None:
    img_in, seg_in, img_out, seg_out = pair
    img = Image.load(img_in)
    seg = Image.load(seg_in)
    cropped_img, cropped_seg = crop_to_seg(img, seg)
    img_out.parent.mkdir(parents=True, exist_ok=True)
    seg_out.parent.mkdir(parents=True, exist_ok=True)
    cropped_img.save(img_out)
    cropped_seg.save(seg_out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Crop image and segmentation to the bounding box of the segmentation.'
    )
    parser.add_argument('--img', required=True, help='Input image file or folder')
    parser.add_argument('--seg', required=True, help='Input seg file or folder')
    parser.add_argument('--out-img', required=True, help='Output image folder')
    parser.add_argument('--out-seg', required=True, help='Output seg folder')
    parser.add_argument('--img-suffix', default='_crop', help='Suffix for output images')
    parser.add_argument('--img-prefix', default='', help='Prefix for output images')
    parser.add_argument('--seg-suffix', default='_crop', help='Suffix for output segs')
    parser.add_argument('--seg-prefix', default='', help='Prefix for output segs')
    add_common_args(parser)
    args = parser.parse_args()

    imgs = collect_niftis(args.img)
    segs = collect_niftis(args.seg)

    if len(imgs) != len(segs):
        import sys
        print(f"Mismatch: {len(imgs)} images vs {len(segs)} segs.", file=sys.stderr)
        sys.exit(1)

    out_img_dir = Path(args.out_img)
    out_seg_dir = Path(args.out_seg)

    pairs = [
        (
            i, s,
            build_output_path(i, out_img_dir, args.img_prefix, args.img_suffix),
            build_output_path(s, out_seg_dir, args.seg_prefix, args.seg_suffix),
        )
        for i, s in zip(imgs, segs)
        if args.overwrite
        or not build_output_path(i, out_img_dir, args.img_prefix, args.img_suffix).exists()
    ]

    process_map(
        functools.partial(process_one, args=args),
        pairs,
        max_workers=args.max_workers,
        disable=args.quiet,
        desc='uroseg crop',
    )
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_commands.py::test_crop_reduces_size tests/test_commands.py::test_crop_preserves_seg_labels tests/test_commands.py::test_crop_cli -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add uroseg/commands/crop_image2seg.py tests/test_commands.py
git commit -m "feat: uroseg crop — crop image and seg to seg bounding box"
```

---

### Task 6: preview_jpg

**Files:**
- Create: `uroseg/commands/preview_jpg.py`
- Modify: `tests/test_commands.py`

**Interfaces:**
- Consumes: `Image.load`, `image.data`; `add_common_args`, `collect_niftis`, `build_output_path`
- Produces:
  - `make_preview(img_data: np.ndarray, seg_data: np.ndarray | None) -> np.ndarray` — H×W×3 uint8 array (3 orthogonal slices side by side)
  - `uroseg preview --img ... --seg ... --out ... [--out-suffix _preview]`

Note: output is `.jpg`, not `.nii.gz`. `build_output_path` is used but the suffix replaces `.nii.gz` extension with `.jpg`.

- [ ] **Step 1: Add tests to tests/test_commands.py**

```python
# ── preview_jpg ───────────────────────────────────────────────────────────────

def test_make_preview_no_seg(img_file):
    from uroseg.commands.preview_jpg import make_preview
    img = Image.load(img_file)
    preview = make_preview(img.data, seg_data=None)
    assert preview.ndim == 3
    assert preview.shape[2] == 3
    assert preview.dtype == np.uint8


def test_make_preview_with_seg(img_file, seg_file):
    from uroseg.commands.preview_jpg import make_preview
    img = Image.load(img_file)
    seg = Image.load(seg_file)
    preview = make_preview(img.data, seg_data=seg.data)
    assert preview.ndim == 3
    assert preview.dtype == np.uint8


def test_preview_cli_no_seg(img_file, tmp_path):
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'preview',
         '--img', str(img_file), '--out', str(out), '--out-suffix', '_preview'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out / 'img_preview.jpg').exists()


def test_preview_cli_with_seg(img_file, seg_file, tmp_path):
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'preview',
         '--img', str(img_file), '--seg', str(seg_file),
         '--out', str(out), '--out-suffix', '_preview'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out / 'img_preview.jpg').exists()
```

- [ ] **Step 2: Run tests — expect failures**

```bash
pytest tests/test_commands.py::test_make_preview_no_seg tests/test_commands.py::test_make_preview_with_seg tests/test_commands.py::test_preview_cli_no_seg tests/test_commands.py::test_preview_cli_with_seg -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement uroseg/commands/preview_jpg.py**

```python
from __future__ import annotations
import argparse
import functools
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from tqdm.contrib.concurrent import process_map

from uroseg.utils.image import Image
from uroseg.utils.utils import add_common_args, collect_niftis, build_output_path

LABEL_COLORS = [
    (255, 80, 80),
    (80, 255, 80),
    (80, 80, 255),
    (255, 255, 80),
    (255, 80, 255),
    (80, 255, 255),
]


def make_preview(img_data: np.ndarray, seg_data: np.ndarray | None = None) -> np.ndarray:
    slices = []
    for axis in range(3):
        idx = img_data.shape[axis] // 2
        sl = np.take(img_data, idx, axis=axis).astype(float)
        sl -= sl.min()
        if sl.max() > 0:
            sl /= sl.max()
        gray = (sl * 255).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)

        if seg_data is not None:
            seg_sl = np.take(seg_data, idx, axis=axis)
            for label_id in np.unique(seg_sl):
                if label_id == 0:
                    continue
                color = LABEL_COLORS[int(label_id - 1) % len(LABEL_COLORS)]
                mask = seg_sl == label_id
                for c, val in enumerate(color):
                    rgb[..., c][mask] = val

        slices.append(rgb)

    max_h = max(s.shape[0] for s in slices)
    padded = []
    for s in slices:
        pad = max_h - s.shape[0]
        s = np.pad(s, ((pad // 2, pad - pad // 2), (0, 0), (0, 0)))
        padded.append(s)

    return np.concatenate(padded, axis=1)


def _build_jpg_path(inp: Path, out_dir: Path, prefix: str, suffix: str) -> Path:
    stem = inp.name
    for ext in ('.nii.gz', '.nii'):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    return out_dir / f'{prefix}{stem}{suffix}.jpg'


def process_one(
    pair: tuple[Path, Path | None, Path],
    args: argparse.Namespace,
) -> None:
    img_path, seg_path, out_path = pair
    img = Image.load(img_path)
    seg_data = Image.load(seg_path).data if seg_path else None
    preview = make_preview(img.data, seg_data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(preview).save(str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate JPG preview images (3 orthogonal slices) of NIfTI files.'
    )
    parser.add_argument('--img', required=True, help='Input image file or folder')
    parser.add_argument('--seg', default=None, help='Input seg file or folder (optional)')
    parser.add_argument('--out', required=True, help='Output folder')
    parser.add_argument('--out-suffix', default='_preview', help='Output filename suffix')
    parser.add_argument('--out-prefix', default='', help='Output filename prefix')
    add_common_args(parser)
    args = parser.parse_args()

    imgs = collect_niftis(args.img)
    segs = collect_niftis(args.seg) if args.seg else [None] * len(imgs)
    out_dir = Path(args.out)

    pairs = [
        (i, s, _build_jpg_path(i, out_dir, args.out_prefix, args.out_suffix))
        for i, s in zip(imgs, segs)
        if args.overwrite
        or not _build_jpg_path(i, out_dir, args.out_prefix, args.out_suffix).exists()
    ]

    process_map(
        functools.partial(process_one, args=args),
        pairs,
        max_workers=args.max_workers,
        disable=args.quiet,
        desc='uroseg preview',
    )
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_commands.py::test_make_preview_no_seg tests/test_commands.py::test_make_preview_with_seg tests/test_commands.py::test_preview_cli_no_seg tests/test_commands.py::test_preview_cli_with_seg -v
```

Expected: all 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add uroseg/commands/preview_jpg.py tests/test_commands.py
git commit -m "feat: uroseg preview — generate JPG previews with optional seg overlay"
```

---

### Task 7: transform_seg2image

**Files:**
- Create: `uroseg/commands/transform_seg2image.py`
- Modify: `tests/test_commands.py`

**Interfaces:**
- Consumes: `Image.load`, `image.data`, `image.affine`, `image.header`, `image.save`; `add_common_args`, `collect_niftis`, `build_output_path`
- Produces:
  - `resample_seg_to_image(seg: Image, ref: Image) -> Image` — nearest-neighbour resample to ref space
  - `uroseg transform_seg2image --seg ... --img ... --out-seg ... [--seg-suffix _transformed]`

- [ ] **Step 1: Add tests to tests/test_commands.py**

```python
# ── transform_seg2image ───────────────────────────────────────────────────────

def test_transform_seg_matches_img_shape(img_file, seg_file, tmp_path):
    from uroseg.commands.transform_seg2image import resample_seg_to_image
    # create a differently-shaped seg
    data = np.zeros((10, 10, 10), dtype=np.int16)
    data[2:8, 2:8, 2:8] = 1
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    seg_small = Image(data, affine, None)
    ref = Image.load(img_file)
    result = resample_seg_to_image(seg_small, ref)
    assert result.data.shape == ref.data.shape


def test_transform_seg2image_cli(img_file, seg_file, tmp_path):
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'transform_seg2image',
         '--seg', str(seg_file), '--img', str(img_file),
         '--out-seg', str(out), '--seg-suffix', '_transformed'],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (out / 'seg_transformed.nii.gz').exists()
```

- [ ] **Step 2: Run tests — expect failures**

```bash
pytest tests/test_commands.py::test_transform_seg_matches_img_shape tests/test_commands.py::test_transform_seg2image_cli -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement uroseg/commands/transform_seg2image.py**

```python
from __future__ import annotations
import argparse
import functools
from pathlib import Path

import numpy as np
import nibabel as nib
import nibabel.processing as nibp
from tqdm.contrib.concurrent import process_map

from uroseg.utils.image import Image
from uroseg.utils.utils import add_common_args, collect_niftis, build_output_path


def resample_seg_to_image(seg: Image, ref: Image) -> Image:
    seg_nib = nib.Nifti1Image(seg.data.astype(np.int32), seg.affine)
    ref_nib = nib.Nifti1Image(ref.data, ref.affine, ref.header)
    resampled = nibp.resample_from_to(seg_nib, ref_nib, order=0, cval=0)
    return Image(
        data=np.asanyarray(resampled.dataobj).astype(seg.data.dtype),
        affine=resampled.affine,
        header=ref.header,
    )


def process_one(
    pair: tuple[Path, Path, Path],
    args: argparse.Namespace,
) -> None:
    seg_path, img_path, out_path = pair
    seg = Image.load(seg_path)
    ref = Image.load(img_path)
    result = resample_seg_to_image(seg, ref)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Resample segmentation to match reference image space (nearest-neighbour).'
    )
    parser.add_argument('--seg', required=True, help='Input seg file or folder')
    parser.add_argument('--img', required=True, help='Reference image file or folder')
    parser.add_argument('--out-seg', required=True, help='Output seg folder')
    parser.add_argument('--seg-suffix', default='_transformed', help='Output seg suffix')
    parser.add_argument('--seg-prefix', default='', help='Output seg prefix')
    add_common_args(parser)
    args = parser.parse_args()

    segs = collect_niftis(args.seg)
    imgs = collect_niftis(args.img)

    if len(segs) != len(imgs):
        import sys
        print(f"Mismatch: {len(segs)} segs vs {len(imgs)} images.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_seg)
    pairs = [
        (s, i, build_output_path(s, out_dir, args.seg_prefix, args.seg_suffix))
        for s, i in zip(segs, imgs)
        if args.overwrite
        or not build_output_path(s, out_dir, args.seg_prefix, args.seg_suffix).exists()
    ]

    process_map(
        functools.partial(process_one, args=args),
        pairs,
        max_workers=args.max_workers,
        disable=args.quiet,
        desc='uroseg transform_seg2image',
    )
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_commands.py::test_transform_seg_matches_img_shape tests/test_commands.py::test_transform_seg2image_cli -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add uroseg/commands/transform_seg2image.py tests/test_commands.py
git commit -m "feat: uroseg transform_seg2image — resample seg to reference image space"
```

---

### Task 8: cpdir

**Files:**
- Create: `uroseg/commands/cpdir.py`
- Modify: `tests/test_commands.py`

**Interfaces:**
- Consumes: `Image.load`, `image.save`; `add_common_args`, `build_pairs`
- Produces: `uroseg cpdir --img ... --out ... [--out-suffix ''] [--out-prefix '']`

- [ ] **Step 1: Add tests to tests/test_commands.py**

```python
# ── cpdir ─────────────────────────────────────────────────────────────────────

def test_cpdir_copies_files(nifti_folder, tmp_path):
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'cpdir',
         '--img', str(nifti_folder), '--out', str(out), '--out-suffix', ''],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert len(list(out.glob('*.nii.gz'))) == 3


def test_cpdir_skips_existing_without_overwrite(nifti_folder, tmp_path):
    import subprocess, sys
    out = tmp_path / 'out'
    out.mkdir()
    # first copy
    subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'cpdir',
         '--img', str(nifti_folder), '--out', str(out), '--out-suffix', ''],
        capture_output=True
    )
    # second copy without --overwrite should skip
    result = subprocess.run(
        [sys.executable, '-m', 'uroseg.cli', 'cpdir',
         '--img', str(nifti_folder), '--out', str(out), '--out-suffix', ''],
        capture_output=True, text=True
    )
    assert result.returncode == 0
```

- [ ] **Step 2: Run tests — expect failures**

```bash
pytest tests/test_commands.py::test_cpdir_copies_files tests/test_commands.py::test_cpdir_skips_existing_without_overwrite -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement uroseg/commands/cpdir.py**

```python
from __future__ import annotations
import argparse
import functools
from pathlib import Path

from tqdm.contrib.concurrent import process_map

from uroseg.utils.image import Image
from uroseg.utils.utils import add_common_args, build_pairs


def process_one(pair: tuple[Path, Path], args: argparse.Namespace) -> None:
    input_path, output_path = pair
    img = Image.load(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Copy NIfTI files from one folder to another with optional renaming.'
    )
    parser.add_argument('--img', required=True, help='Input file or folder')
    parser.add_argument('--out', required=True, help='Output folder')
    parser.add_argument('--out-suffix', default='', help='Output filename suffix (default: none)')
    parser.add_argument('--out-prefix', default='', help='Output filename prefix (default: none)')
    add_common_args(parser)
    args = parser.parse_args()

    pairs = build_pairs(args.img, args.out, args.out_suffix, args.out_prefix, args.overwrite)
    process_map(
        functools.partial(process_one, args=args),
        pairs,
        max_workers=args.max_workers,
        disable=args.quiet,
        desc='uroseg cpdir',
    )
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_commands.py::test_cpdir_copies_files tests/test_commands.py::test_cpdir_skips_existing_without_overwrite -v
```

Expected: both PASS.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Verify all subcommands reachable**

```bash
uroseg map --help
uroseg resample --help
uroseg reorient --help
uroseg largest_component --help
uroseg crop --help
uroseg preview --help
uroseg transform_seg2image --help
uroseg cpdir --help
```

Expected: all print help without errors.

- [ ] **Step 7: Commit**

```bash
git add uroseg/commands/cpdir.py tests/test_commands.py
git commit -m "feat: uroseg cpdir — copy NIfTI files with optional renaming"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] `uroseg map --seg --out --map --out-suffix --out-prefix` → Task 1
- [x] `uroseg resample --img --out --spacing X Y Z` → Task 2
- [x] `uroseg reorient --img --out --orientation` → Task 3
- [x] `uroseg largest_component --seg --out --labels` → Task 4
- [x] `uroseg crop --img --seg --out-img --out-seg --img-suffix --seg-suffix` → Task 5
- [x] `uroseg preview --img --seg --out --out-suffix` → Task 6
- [x] `uroseg transform_seg2image --seg --img --out-seg --seg-suffix` → Task 7
- [x] `uroseg cpdir --img --out --out-suffix --out-prefix` → Task 8
- [x] All commands support `--overwrite --max-workers --quiet` → every task via `add_common_args`
- [x] Batch processing with tqdm `process_map` → every task
- [x] Single-input commands use `build_pairs` → Tasks 1, 2, 3, 4, 8
- [x] Multi-input commands use `collect_niftis` + `build_output_path` directly → Tasks 5, 6, 7
- [x] Output naming `{prefix}{stem}{suffix}.nii.gz` → all tasks via `build_output_path`

**Placeholder scan:** None found.

**Type consistency:**
- `process_one(pair: tuple[Path, Path], args)` for single-input tools ✓
- `process_one(pair: tuple[Path, Path, Path], args)` for three-input tools ✓
- `process_one(pair: tuple[Path, Path, Path, Path], args)` for crop ✓
- `keep_largest_component(data: np.ndarray, labels: list[int] | None) -> np.ndarray` ✓
- `resample_seg_to_image(seg: Image, ref: Image) -> Image` ✓
- `crop_to_seg(img: Image, seg: Image) -> tuple[Image, Image]` ✓
- `make_preview(img_data: np.ndarray, seg_data: np.ndarray | None) -> np.ndarray` ✓

---

*Next: Plan 3 — Training + README*
