# UroSeg CLI UX

**Date:** 2026-06-24
**Status:** Approved

## Goal

Three focused CLI improvements: canonical help output, consistent "model" terminology, and a generic `train <engine>` subcommand pattern that makes adding new backends (monad, etc.) a one-file change.

## Changes

### 1. Canonical help — `uroseg`, `uroseg -h`, `uroseg --help`, `uroseg list`

All four entry points call the same `show_help()` function in `list_models.py`. The output:

```
uroseg — urological anatomy segmentation

Models:
  prostate    Prostate: whole prostate (1), transition zone (2), peripheral zone (3)
  bladder     Urinary bladder (CT)

Commands:
  map                  Remap label IDs
  resample             Resample image to target voxel size
  reorient             Reorient image to RAS canonical
  largest_component    Keep largest connected component per label
  crop                 Crop image to segmentation bounding box
  preview              Generate JPG slice preview
  transform_seg2image  Resample segmentation to reference image space
  cpdir                Copy NIfTI files with optional renaming
  install              Download model weights
  train nnunet         Train with nnU-Net + AugLab

Run 'uroseg <model|command> --help' for per-command usage.
```

All four entry points exit 0 after printing help (consistent with `git`, `docker`).

**Models section:** loaded at runtime from `get_all_models()` — new model JSONs appear automatically.

**Commands section:** hardcoded dict in `list_models.py` keyed by subcommand name. `train` is listed as `train nnunet` (the only currently available engine). When a new engine is added, this dict is updated.

### 2. Terminology: "organ" → "model"

Error message in `cli.py` (`_dispatch_model`) changes from:
```
Unknown organ or subcommand: 'X'
Available organs: ...
```
to:
```
Unknown model or subcommand: 'X'
Run 'uroseg --help' to see available models and commands.
```

No changes to any command files. No changes to model JSON keys.

### 3. `uroseg train <engine>` — generic dispatcher

**Dispatch chain:**

```
cli.py          strips 'train', calls train.main()
train.py        parses engine positional, dispatches to engine module
train_nnunet.py actual nnU-Net + AugLab training logic (current train.py, renamed)
```

**`sys.argv` flow:** `cli.py` strips `'train'` before calling `train.main()`, so `train.py` sees `['uroseg', 'nnunet', 'ORGAN', ...]`. `train.py` reads `sys.argv[1]` as the engine, strips it (`sys.argv = sys.argv[:1] + sys.argv[2:]`), then calls the engine module's `main()`.

**`train.py` behaviour:**
- `uroseg train nnunet ORGAN [opts]` → strips `nnunet`, calls `train_nnunet.main()`
- `uroseg train` (no engine) → prints engine help and exits 1
- `uroseg train -h` / `uroseg train --help` → prints engine help and exits 0
- `uroseg train unknown` → prints "Unknown engine: 'unknown'" + engine help, exits 1

Engine help text:
```
Usage: uroseg train <engine> ORGAN [options]
Engines:
  nnunet    Train with nnU-Net + AugLab
```

**`train_nnunet.py`:** content-identical to current `train.py` (rename only). `cli.py` dispatch for `train` is unchanged in logic — it still strips 'train' from `sys.argv` and calls `train.main()`.

**Adding a new engine later:**
1. Create `train_<engine>.py`
2. Add one `elif engine == '<engine>'` branch in `train.py`
3. Update the command description dict in `list_models.py`

`cli.py` is not touched.

## Files

| File | Change |
|------|--------|
| `uroseg/cli.py` | Handle `-h`/`--help`/no-args → `show_help()`; `list` → `show_help()`; rename "organ"→"model" in `_dispatch_model` |
| `uroseg/commands/list_models.py` | Add `show_help()` with models + hardcoded command descriptions; `main()` calls `show_help()` |
| `uroseg/commands/train.py` | New generic engine dispatcher |
| `uroseg/commands/train_nnunet.py` | Renamed from current `train.py` (no content changes) |
| `README.md` | `uroseg train kidney` → `uroseg train nnunet kidney` in usage examples and CLI reference |
| `tests/test_commands.py` | Update train invocations: `'train', 'kidney'` → `'train', 'nnunet', 'kidney'` |

## Tests

- `test_help_no_args`: `uroseg` (no args) prints help to stdout, exits 0
- `test_help_flag`: `uroseg -h` and `uroseg --help` print help, exit 0
- `test_list_redirects_to_help`: `uroseg list` output matches `uroseg --help` output
- `test_help_contains_models`: help output contains `prostate` and `bladder`
- `test_help_contains_commands`: help output contains `resample`, `train nnunet`
- `test_train_no_engine_exits_1`: `uroseg train` exits 1 and prints engine help
- `test_train_unknown_engine_exits_1`: `uroseg train foo` exits 1 and prints "Unknown engine"
- Update existing train tests: `['train', 'kidney', ...]` → `['train', 'nnunet', 'kidney', ...]`

## Out of scope

- Changing any command file other than `list_models.py`, `train.py`, `train_nnunet.py`
- Converting commands to expose `get_parser()` (argparse subparsers refactor)
- Model definition format changes (separate spec)
- `uroseg train nnunet --help` showing train-specific help (already works via train_nnunet.py's own argparse)
