from __future__ import annotations
from uroseg.models import list_models as _list_models, get_model

_COMMANDS = {
    'map':                 'Remap label IDs',
    'resample':            'Resample image to target voxel size',
    'reorient':            'Reorient image to RAS canonical',
    'largest_component':   'Keep largest connected component per label',
    'crop':                'Crop image to segmentation bounding box',
    'preview':             'Generate JPG slice preview',
    'transform_seg2image': 'Resample segmentation to reference image space',
    'cpdir':               'Copy NIfTI files with optional renaming',
    'volume':              'Compute label volumes in mm³',
    'install':             'Download model weights',
    'train nnunet':        'Train with nnU-Net + AugLab',
}


def show_help() -> None:
    names = _list_models()
    lines = ['uroseg — urological anatomy segmentation', '']
    lines.append('Models:')
    if names:
        name_w = max(len(n) for n in names) + 2
        for name in sorted(names):
            model = get_model(name)
            lines.append(f'  {name:<{name_w}}{model.description}')
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
