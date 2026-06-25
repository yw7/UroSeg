# UroSeg

Automated segmentation of urological anatomy from medical images. Built on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) with [AugLab](https://github.com/neuropoly/AugLab) online augmentation.

---

## Available Models

| Command | Anatomy | Modality | Labels |
|---------|---------|----------|--------|
| `uroseg prostate` | Prostate | MRI T2 | whole prostate, peripheral zone, central zone, anterior fibromuscular stroma |
| `uroseg bladder` | Bladder | CT | bladder |

---

## Installation

### 1. Install PyTorch

Use [light-the-torch](https://github.com/pmeier/light-the-torch) to automatically detect your CUDA version and install the correct PyTorch build:

```bash
pip install light-the-torch
ltt install torch
```

### 2. Install UroSeg

```bash
pip install uroseg
```

Development install:

```bash
git clone https://github.com/yw7/uroseg
cd uroseg
pip install -e .
```

### 3. Download model weights

```bash
uroseg install --model prostate
uroseg install --model bladder
uroseg install --all          # download all available models
```

By default, weights are stored in `~/uroseg/nnUNet/results/`. Override with `--data-dir PATH` or by setting `UROSEG_DATA=/path/to/data`.

---

## Usage

### Inference

```bash
# Single image — pass a folder as --out (output named automatically)
uroseg prostate -i subject01_T2.nii.gz -o segs/
uroseg bladder  -i subject01_CT.nii.gz  -o segs/

# Batch (folder input → folder output)
uroseg prostate -i /data/mri/ -o /data/segs/ -w 4
```

### Utilities

```bash
# Remap label IDs — JSON file or direct key:value pairs
uroseg map -s seg.nii.gz -o remapped/ -m labels.json
uroseg map -s seg.nii.gz -o remapped/ -m 1:2 3:0

# Resample to 1×1×1 mm isotropic
uroseg resample -i img.nii.gz -o img_1mm/ --mm 1

# Reorient to RAS canonical
uroseg reorient -i img.nii.gz -o img_ras/

# Keep only the largest connected component per label
uroseg largest_component -s seg.nii.gz -o seg_lc/

# Crop image and seg to segmentation bounding box
uroseg crop -i img.nii.gz -s seg.nii.gz -o img_crop/

# Generate JPG preview (single slice, optional seg overlay)
uroseg preview -i img.nii.gz -s seg.nii.gz -o previews/ --orient sag --sliceloc 0.5

# Resample segmentation to match reference image space (nearest-neighbour)
uroseg transform_seg2image -s seg.nii.gz -i ref.nii.gz --out-seg seg_transformed/

# Copy NIfTI files with optional renaming
uroseg cpdir -i /data/mri/ -o /data/mri_copy/ --out-suffix _copy

# List available organ models
uroseg list
```

### CLI reference

All commands support `-r`/`--overwrite`, `-w`/`--max-workers N`, and `-q`/`--quiet`.

```
uroseg <organ>              -i/--img PATH  -o/--out PATH  [-f/--fold N] [-d/--device cuda|cpu|mps]
uroseg map                  -s/--seg PATH  -o/--out PATH  -m/--map JSON|KEY:VAL ...
uroseg resample             -i/--img PATH  -o/--out PATH  -m/--mm X [Y Z]
uroseg reorient             -i/--img PATH  -o/--out PATH
uroseg largest_component    -s/--seg PATH  -o/--out PATH  [-l/--labels 1 2 3]
uroseg preview              -i/--img PATH [-s/--seg PATH] -o/--out PATH [-t/--orient sag|ax|cor] [-l/--sliceloc 0.5]
uroseg crop                 -i/--img PATH  -s/--seg PATH  -o/--out PATH  [-m/--margin N]
uroseg transform_seg2image  -s/--seg PATH  -i/--img PATH  --out-seg PATH  [-x/--interpolation nearest|linear|label]
uroseg cpdir                -i/--img PATH  -o/--out PATH  [--out-suffix SUFFIX] [--out-prefix PREFIX]
uroseg install              --model NAME [NAME ...] | --all  [--data-dir PATH]
uroseg train nnunet ORGAN   [-f/--fold N]  [--auglab-config JSON]  [--gpus N]  [--data-dir PATH]
uroseg list
```

---

## Training

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

### 2. Place training data

```
~/uroseg/nnUNet/raw/Dataset020_Kidney/
├── dataset.json          ← auto-generated by uroseg train nnunet
├── imagesTr/
│   ├── kidney_001_0000.nii.gz   ← image channel 0
│   └── ...
└── labelsTr/
    ├── kidney_001.nii.gz        ← ground truth segmentation
    └── ...
```

nnU-Net filename convention: images end in `_0000.nii.gz` (channel 0), labels have no channel suffix.

### 3. Run training

```bash
# Basic
uroseg train nnunet kidney

# With AugLab augmentation config
uroseg train nnunet kidney --auglab-config auglab.json

# Custom data directory
uroseg train nnunet kidney --data-dir /scratch/uroseg_data
```

`uroseg train nnunet` automatically:
- Generates `dataset.json` from the model JSON
- Sets `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results` environment variables
- Copies `nnUNetTrainerDAExt` from the auglab package into nnunetv2's trainer directory (required so nnU-Net can discover it)
- Runs `nnUNetv2_plan_and_preprocess` (skipped if already done)
- Runs `nnUNetv2_train` with `nnUNetTrainerDAExt` (AugLab's trainer subclass)

Trained model location:

```
~/uroseg/nnUNet/results/<nnunet_task>/nnUNetTrainerDAExt__nnUNetPlans__3d_fullres/fold_<N>/
├── checkpoint_best.pth
└── checkpoint_final.pth
```

---

## Data Storage

All UroSeg data lives under a single configurable root:

| Priority | Method | Value |
|----------|--------|-------|
| 1 (highest) | CLI flag | `--data-dir /path/to/data` |
| 2 | Environment variable | `export UROSEG_DATA=/path/to/data` |
| 3 (default) | Default | `~/uroseg/` |

```
~/uroseg/
└── nnUNet/
    ├── raw/                        # user training images (imagesTr/, labelsTr/)
    ├── preprocessed/               # nnU-Net preprocessing cache (auto-created)
    ├── results/
    │   ├── r20260101/              # versioned by release date tag
    │   │   └── Dataset001_Prostate/
    │   └── r20261001/              # newer release coexists
    │       └── Dataset001_Prostate/
    └── exports/                    # downloaded zip archives (removed after extraction)
```

---

## Contributing — Adding a New Organ Model

1. Create `uroseg/resources/models/<organ>.py` per the Python format above
2. Place training data in `~/uroseg/nnUNet/raw/<nnunet_task>/imagesTr/` and `labelsTr/`
3. Train: `uroseg train nnunet <organ>`
4. Archive the trained model: `Dataset###_<Name>_r<YYYYMMDD>.zip`
5. Upload as a GitHub Release asset and set `weights_url` in the model JSON
6. Open a pull request
