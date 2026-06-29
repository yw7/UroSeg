# UroSeg

Automated segmentation of urological anatomy from MRI/CT.

<img width="488" height="415" alt="Image" src="https://github.com/user-attachments/assets/1ccad12b-7dc2-45bc-8399-ccecb591437b" />

---

## Models

| Model | Modality | Labels |
|-------|----------|--------|
| `prostate` | CT/MRI | whole prostate (1+2+3), transition zone (2), peripheral zone (3) |

---

## Install

```bash
# 1. PyTorch — auto-detects CUDA version
pip install light-the-torch
ltt install torch

# 2. UroSeg
pip install uroseg

# 3. Download model weights
uroseg install --model prostate
uroseg install --model bladder
uroseg install --all       # all models
```

Weights are stored in `~/uroseg/` by default. Override with `--data-dir PATH` or `export UROSEG_DATA=/path`.

Development install:

```bash
git clone https://github.com/yw7/uroseg && cd uroseg
pip install -e .
```

---

## Predict

```bash
# Single file — output written to segs/scan.nii.gz
uroseg prostate scan.nii.gz segs/
uroseg bladder  scan.nii.gz segs/

# Whole folder
uroseg prostate mri_dir/ segs/

# Keep output in 1mm isotropic space instead of resampling back to input
uroseg prostate scan.nii.gz segs/ --iso

# CPU / Apple Silicon
uroseg prostate scan.nii.gz segs/ --device cpu
uroseg prostate scan.nii.gz segs/ --device mps
```

---

## Tools

```bash
# Volume — prints mm³ per label, or saves CSV for a folder
uroseg volume --seg seg.nii.gz  --model prostate
uroseg volume --seg segs/       --model prostate  --out volumes.csv

# Generate JPG slice preview with optional seg overlay
uroseg preview --img scan.nii.gz --seg seg.nii.gz --out previews/ --orient sag

# Crop image to segmentation bounding box
uroseg crop --img scan.nii.gz --seg seg.nii.gz --out cropped/

# Resample to target voxel size
uroseg resample --img scan.nii.gz --out out/ --mm 1         # isotropic 1mm
uroseg resample --img scan.nii.gz --out out/ --mm 1 1 3     # anisotropic

# Reorient to RAS canonical
uroseg reorient --img scan.nii.gz --out out/

# Keep only largest connected component
uroseg largest_component --seg seg.nii.gz --out out/
uroseg largest_component --seg seg.nii.gz --out out/ --binarize  # across all labels

# Remap label IDs
uroseg map --seg seg.nii.gz --out out/ --map 1:10 2:20
uroseg map --seg seg.nii.gz --out out/ --map labels.json

# Resample segmentation to reference image space
uroseg transform_seg2image --seg seg.nii.gz --img ref.nii.gz --out-seg out/

# Copy/rename NIfTI files
uroseg cpdir --img dir/ --out copy/ --out-prefix sub01_

# List available models and commands
uroseg list
```

All tools accept `--overwrite`, `--max-workers N`, and `--quiet`.

---

## Python API

```python
import uroseg

# Predict
uroseg.Prostate().predict('scan.nii.gz', 'segs/')
uroseg.Bladder().predict('scan.nii.gz', 'segs/')

# Volume
vols = uroseg.volume('seg.nii.gz', {'prostate': [1, 2, 3], 'prostate_tz': 2})
# → {'prostate': 28540.0, 'prostate_tz': 12340.0}  (mm³)

# Tools
uroseg.resample('scan.nii.gz', 'out/', mm=1.0)
uroseg.crop('scan.nii.gz', 'seg.nii.gz', 'out/')
uroseg.reorient('scan.nii.gz', 'out/')
uroseg.preview('scan.nii.gz', 'previews/', seg='seg.nii.gz', orient='sag')
```

---

## Add a New Model

### 1. Create `uroseg/models/<name>.py`

```python
from uroseg.models.base import NNUNetSegModel

class Kidney(NNUNetSegModel):
    name = "kidney"
    description = "Kidney (CT)"
    weights_url = "https://github.com/yw7/UroSeg/releases/download/<tag>/Dataset020_Kidney_<tag>.zip"
    labels = {"background": 0, "kidney": 1}
    nnunet_task = "Dataset020_Kidney"

MODEL = Kidney()
NNUNET_TASK = Kidney.nnunet_task

def main():
    Kidney.cli_main()
```

For region-based models (sigmoid per region) use a list value: `"prostate": [1, 2, 3]`.

### 2. Register in `uroseg/models/__init__.py`

```python
from uroseg.models.kidney import Kidney

_REGISTRY = {cls.name: cls for cls in [Prostate, Bladder, Kidney]}
```

### 3. Place training data

```
~/uroseg/nnUNet/raw/Dataset020_Kidney/
├── imagesTr/
│   ├── kidney_001_0000.nii.gz   ← image (channel 0)
│   └── ...
└── labelsTr/
    ├── kidney_001.nii.gz        ← ground truth
    └── ...
```

### 4. Train

```bash
uroseg train nnunet kidney

# With AugLab augmentation
uroseg train nnunet kidney --auglab-config auglab.json

# Custom data path
uroseg train nnunet kidney --data-dir /scratch/data
```

Generates `dataset.json`, runs `nnUNetv2_plan_and_preprocess` and `nnUNetv2_train`. Trained weights land in:

```
~/uroseg/<name>/<release_tag>/
```

### 5. Contribute

1. Upload trained weights zip to a GitHub Release
2. Set `weights_url` in the model class
3. Open a pull request
