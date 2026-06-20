# UroSeg Design Spec
**Date:** 2026-06-20
**Status:** Approved

---

## Overview

UroSeg is a Python package for automated segmentation of urological anatomy from medical images. It is inspired by [TotalSpineSeg](https://github.com/neuropoly/totalspineseg) but adapted for urological targets (prostate, bladder, etc.). It uses [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) as the deep-learning backbone and [AugLab](https://github.com/neuropoly/AugLab) for online data augmentation during training.

**Key design goals:**
- Single `uroseg` CLI entry point with subcommands
- Independent per-organ models, each capable of multiple sub-labels
- Modality-agnostic (MRI, CT, or any — declared per model config)
- User creates a model JSON first; `uroseg train` auto-generates nnU-Net's `dataset.json` from it
- Consistent, readable tool code — unity over abstraction
- No bundled dataset download/prepare scripts

---

## Package Structure

```
uroseg/
├── __init__.py
├── cli.py                        # single entry point: dispatches all subcommands
├── commands/
│   ├── inference.py              # uroseg <organ>
│   ├── train.py                  # uroseg train
│   ├── install.py                # uroseg install
│   ├── map_labels.py             # uroseg map
│   ├── resample.py               # uroseg resample
│   ├── preview_jpg.py            # uroseg preview
│   ├── crop_image2seg.py         # uroseg crop
│   ├── largest_component.py      # uroseg largest_component
│   ├── reorient_canonical.py     # uroseg reorient
│   ├── cpdir.py                  # uroseg cpdir
│   ├── transform_seg2image.py    # uroseg transform_seg2image
│   └── predict_nnunet.py         # uroseg predict_nnunet (low-level)
├── utils/
│   ├── image.py                  # Image class — shared I/O foundation
│   └── utils.py                  # add_common_args, build_pairs, collect_niftis, build_output_path
└── resources/
    └── models/
        ├── prostate.json
        ├── bladder.json
        └── ...                   # one JSON per organ model
```

**`pyproject.toml` — single console_scripts entry:**
```toml
[project.scripts]
uroseg = "uroseg.cli:main"
```

---

## CLI — Single Entry Point

`uroseg` dispatches based on its first argument.

**Known subcommands** (registered via argparse subparsers):
```
uroseg train
uroseg install
uroseg map
uroseg resample
uroseg preview
uroseg crop
uroseg largest_component
uroseg reorient
uroseg cpdir
uroseg transform_seg2image
uroseg predict_nnunet
uroseg list
```

**Organ inference** — if the first argument is not a known subcommand, it is treated as an organ name looked up in the model registry:
```
uroseg prostate --img input/ --out output/
uroseg bladder  --img input/ --out output/
```

`uroseg list` prints all available organ models from `resources/models/`.

---

## Model Registry

Each organ model is a JSON file in `uroseg/resources/models/`:

```json
{
  "name": "prostate",
  "description": "Prostate zones: peripheral zone (PZ), central zone (CZ), anterior fibromuscular stroma (AFS)",
  "modality": "MRI-T2",
  "nnunet_task": "Dataset001_Prostate",
  "labels": {
    "1": "prostate_pz",
    "2": "prostate_cz",
    "3": "prostate_afs"
  },
  "weights_filename": "prostate.zip"    // GitHub Release asset name; omit for community/unreleased models
}
```

Fields:
- `name` — matches the filename stem and the CLI subcommand token
- `modality` — channel names passed to nnU-Net's `dataset.json` (e.g. `["MRI-T2"]`); used by `uroseg train` to auto-generate `dataset.json`
- `nnunet_task` — nnU-Net dataset/task identifier used for predict and train
- `labels` — label ID → anatomical name mapping; used to auto-generate `dataset.json` and for `uroseg map`
- `weights_filename` — filename of the weights archive on the GitHub Release (e.g. `prostate_v1.0.0.zip`); URL is constructed at runtime from the package version (see Weight URLs section)

**Training workflow — model JSON first:**
The user creates `resources/models/<organ>.json` before training. `uroseg train` reads it and auto-generates the nnU-Net `dataset.json` (label map, channel names, num_training) so the user never writes `dataset.json` by hand. The user only needs to place raw images and segmentations in `nnUNet_raw/DatasetXXX/imagesTr/` and `labelsTr/`.

Adding a new organ requires only: creating the model JSON, placing data, running `uroseg train`, and opening a PR.

---

## `Image` Class (`uroseg/utils/image.py`)

Shared I/O foundation used by every command and utility. No command calls nibabel or SimpleITK directly.

```python
class Image:
    data: np.ndarray       # voxel data
    affine: np.ndarray     # 4x4 affine matrix
    header: nib.Header     # NIfTI header

    @staticmethod
    def load(path: str | Path) -> Image    # .nii or .nii.gz
    def save(path: str | Path) -> None     # always saves as .nii.gz
    def copy() -> Image
    def reorient(orientation: str = 'RAS') -> Image
    def resample(voxel_size: tuple) -> Image
    def bounding_box(label: int = None) -> tuple
```

Adding support for a new format (e.g., DICOM) only requires changes here.

---

## Argument Naming Convention

All tools follow the same vocabulary — no exceptions.

### Input arguments
| Arg | Meaning |
|-----|---------|
| `--img` | Input image file or folder (`.nii` / `.nii.gz`) |
| `--seg` | Input segmentation file or folder |

### Output arguments
| Arg | Meaning |
|-----|---------|
| `--out` | Output file or folder (single output type) |
| `--out-img` | Output image (when tool produces both image + seg) |
| `--out-seg` | Output segmentation (when tool produces both) |

### Suffix / prefix arguments
| Arg | Meaning |
|-----|---------|
| `--out-suffix` | Suffix for single output type (e.g. `_mapped`) |
| `--out-prefix` | Prefix for single output type |
| `--img-suffix` | Suffix for image outputs in multi-output tools |
| `--img-prefix` | Prefix for image outputs in multi-output tools |
| `--seg-suffix` | Suffix for seg outputs in multi-output tools |
| `--seg-prefix` | Prefix for seg outputs in multi-output tools |

**Output filename:** `{prefix}{stem}{suffix}.nii.gz`

### Shared args registered by `add_common_args(parser)` — identical across all tools
| Arg | Default | Meaning |
|-----|---------|---------|
| `--overwrite` | False | Overwrite existing outputs; skip if exists by default |
| `--max-workers` | 1 | Number of parallel workers |
| `--quiet` | False | Suppress progress bar and non-error output |

---

## Tool Argument Signatures

```
uroseg prostate          --img input/ --out output/
                         [--fold N] [--out-suffix _seg]

uroseg map               --seg input/ --out output/
                         --map labels.json [--out-suffix _mapped]

uroseg resample          --img input/ --out output/
                         --spacing X Y Z [--out-suffix _resampled]

uroseg reorient          --img input/ --out output/
                         [--orientation RAS] [--out-suffix _reoriented]

uroseg largest_component --seg input/ --out output/
                         [--labels 1 2 3] [--out-suffix _largest]

uroseg preview           --img input/ --seg input_seg/
                         --out output/ [--out-suffix _preview]

uroseg crop              --img input/ --seg input_seg/
                         --out-img img_out/ --out-seg seg_out/
                         [--img-suffix _crop] [--seg-suffix _crop]

uroseg transform_seg2image --seg input/ --img input_img/
                           --out-seg output/ [--seg-suffix _transformed]

uroseg cpdir             --img input/ --out output/

uroseg install           --model prostate bladder  # one or more models
                         --all                     # download all available models

uroseg list              # print available organ models from resources/models/
```

---

## Batch Processing Pattern

Every command that operates on files follows this identical `main()` structure. The only thing that differs between tools is `process_one` and the tool-specific argparse arguments.

```python
# Example: uroseg/commands/map_labels.py
import functools
import argparse
from tqdm.contrib.concurrent import process_map
from uroseg.utils.utils import add_common_args, build_pairs
from uroseg.utils.image import Image

def process_one(pair, args):
    input_path, output_path = pair
    img = Image.load(input_path)
    img.data = apply_map(img.data, args.map)
    img.save(output_path)

def main():
    parser = argparse.ArgumentParser(description="Remap label IDs in segmentation files.")
    parser.add_argument('--seg', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--map', required=True, help='Path to label map JSON')
    parser.add_argument('--out-suffix', default='_mapped')
    parser.add_argument('--out-prefix', default='')
    add_common_args(parser)
    args = parser.parse_args()

    pairs = build_pairs(args.seg, args.out, args.out_suffix, args.out_prefix, args.overwrite)
    process_map(
        functools.partial(process_one, args=args),
        pairs,
        max_workers=args.max_workers,
        disable=args.quiet,
        desc="uroseg map",
    )
```

Tools that take multiple inputs (e.g., `uroseg crop` with `--img` and `--seg`) do not use `build_pairs` for the paired collection — they implement their own input collection logic in `main()` using `collect_niftis` and `build_output_path` directly. The `main()` structure (argparse → collect → process_map) remains visually identical.

### Shared helpers in `utils/utils.py`
```python
def add_common_args(parser)                         # --overwrite, --max-workers, --quiet
def collect_niftis(path) -> list[Path]              # file or folder → sorted .nii/.nii.gz list
def build_output_path(inp, out_dir, prefix, suffix) -> Path
def build_pairs(inp, out, suffix, prefix, overwrite) -> list[tuple[Path, Path]]
```

---

## Training (`uroseg train`)

Python replacement for `scripts/train.sh`. Wraps nnU-Net with AugLab online augmentation.

### CLI
```
uroseg train \
  --organ prostate \
  --dataset DATASET_ID \
  --fold N \
  [--auglab-config auglab.json] \
  [--gpus 1] \
  [--nnunet-dir /path/to/nnunet_data]
```

### Steps executed internally
1. Load `resources/models/<organ>.json`
2. Validate that `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results` env vars are set (or use `--nnunet-dir`)
3. Auto-generate `nnUNet_raw/DatasetXXX/dataset.json` from the model JSON (labels, channel names, num_training from imagesTr count)
4. Run `nnUNetv2_plan_and_preprocess -d DATASET_ID` if not already done
5. Run `nnUNetv2_train DATASET_ID 3d_fullres FOLD --trainer nnUNetTrainerDAExt`
6. If `--auglab-config` is provided, set `AUGLAB_CONFIG` env var before invoking nnU-Net
7. On completion, print the `nnUNet_results/` path where the trained model is stored

### Trained model storage
By default, nnU-Net stores trained models in `$nnUNet_results/DatasetXXX_<name>/nnUNetTrainerDAExt__nnUNetPlans__3d_fullres/`. This path is set via the `nnUNet_results` environment variable.

To use a different location, set `nnUNet_results` before training:
```bash
export nnUNet_results=/path/to/my/models
uroseg train --organ prostate --dataset 1 --fold 0
```

For inference, the same `nnUNet_results` variable is used by `uroseg predict_nnunet` to locate the model.

### AugLab integration
AugLab's `nnUNetTrainerDAExt` is specified via nnU-Net's `--trainer` flag — no monkey-patching. AugLab is a required dependency. When `--auglab-config` is provided, `uroseg train` sets the `AUGLAB_CONFIG` environment variable to the JSON path before invoking nnU-Net; the AugLab trainer reads this variable internally. If `--auglab-config` is omitted, AugLab trainer defaults are used.

---

## Installation & Dependencies

### `pyproject.toml` dependencies
```toml
[project]
requires-python = ">=3.10"
dependencies = [
    "tqdm",
    "numpy",
    "nibabel",
    "scipy",
    "pillow",
    "SimpleITK",
    "auglab",
    "nnunetv2",
]
```

nnunetv2 is a required dependency — it is the core inference and training backbone.

### Install flow (from README)
```bash
# 1. Install PyTorch with correct CUDA version (light-the-torch handles detection)
pip install light-the-torch
ltt install torch

# 2. Install UroSeg
pip install uroseg                # standard
pip install -e "."                # development

# 3. Download model weights
uroseg install --model prostate
uroseg install --all
```

light-the-torch is not a dependency of UroSeg — it is a user-side installation helper documented in the README only.

---

## Weight URLs & Versioned Releases

Pre-trained model weights are distributed as GitHub Release assets, version-pinned to the UroSeg package version. This ensures that installed weights always match the model architecture of the installed package.

### How it works
`uroseg/__init__.py` defines the base URL using the package version:
```python
from importlib.metadata import version
__version__ = version("uroseg")
WEIGHTS_BASE_URL = f"https://github.com/<org>/uroseg/releases/download/v{__version__}/"
```

Each model JSON's `weights_filename` field holds the archive name (e.g. `prostate.zip`). `uroseg install` constructs the full URL at runtime:
```python
url = WEIGHTS_BASE_URL + model["weights_filename"]
```

### Downloaded weights storage
Weights are downloaded to `~/.uroseg/weights/` by default. Override with:
```bash
export UROSEG_WEIGHTS_DIR=/path/to/weights
```

`uroseg install` downloads and extracts weights there. `uroseg predict_nnunet` and inference commands check `UROSEG_WEIGHTS_DIR` first, then fall back to `nnUNet_results` for locally trained models.

### Release workflow
Each GitHub release tags a version and attaches weights archives as release assets. The `weights_filename` in each model JSON must match the asset name on that release. Community/unreleased models omit `weights_filename` and are not auto-downloaded.

---

## Inference Data Flow

```
Input NIfTI(s)  (--img)
  → collect_niftis()
  → Image.load() → reorient to canonical (RAS)
  → nnU-Net predict (predict_nnunet command, using resources/models/<organ>.json task ID)
  → output seg NIfTI
  → Image.load(seg) → apply label map from model JSON
  → save to --out with --out-suffix
```

Weights are checked on first inference call; auto-downloaded if missing (same pattern as TotalSpineSeg).

---

## README Structure

```
# UroSeg
One-line description.

## Available Models
Table: command | anatomy | labels

## Installation
1. Prerequisites
2. Install PyTorch (light-the-torch, two lines)
3. Install UroSeg (pip install)
4. Download weights (uroseg install)

## Usage
### Inference (uroseg prostate, uroseg bladder, ...)
### Utilities (uroseg map, uroseg resample, ...)
### Full CLI reference

## Training
1. Create resources/models/<organ>.json (labels, modality, nnunet_task)
2. Place images in nnUNet_raw/DatasetXXX/imagesTr/ and labelsTr/
3. Set env vars (nnUNet_raw, nnUNet_preprocessed, nnUNet_results)
4. uroseg train --organ <organ> --dataset XXX --fold 0

## Contributing — Adding a New Organ Model
1. Add resources/models/<organ>.json
2. Train with uroseg train
3. Upload weights archive to GitHub Release
4. Submit PR
```

---

## Out of Scope

- Bundled dataset download/prepare scripts
- Offline augmentation tool (AugLab handles augmentation online during training)
- DICOM support in v1 (Image class designed to add it later in one place)
- Plugin/entry-point system for third-party organ models
