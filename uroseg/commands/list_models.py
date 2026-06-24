from __future__ import annotations
from uroseg.utils.utils import get_all_models

_COMMANDS = {
    'map':                 'Remap label IDs',
    'resample':            'Resample image to target voxel size',
    'reorient':            'Reorient image to RAS canonical',
    'largest_component':   'Keep largest connected component per label',
    'crop':                'Crop image to segmentation bounding box',
    'preview':             'Generate JPG slice preview',
    'transform_seg2image': 'Resample segmentation to reference image space',
    'cpdir':               'Copy NIfTI files with optional renaming',
    'install':             'Download model weights',
    'train nnunet':        'Train with nnU-Net + AugLab',
}


def show_help() -> None:
    models = get_all_models()
    lines = ['uroseg — urological anatomy segmentation', '']
    lines.append('Models:')
    if models:
        name_w = max(len(n) for n in models) + 2
        for name, info in sorted(models.items()):
            lines.append(f'  {name:<{name_w}}{info.get("description", "")}')
    else:
        lines.append('  (no models installed)')
    lines.append('')
    lines.append('Commands:')
    cmd_w = max(len(c) for c in _COMMANDS) + 2
    for cmd, desc in _COMMANDS.items():
        lines.append(f'  {cmd:<{cmd_w}}{desc}')
    lines.append('')
    lines.append("Run 'uroseg <model|command> --help' for per-command usage.")
    print('\n'.join(lines))


def main() -> None:
    show_help()
