from __future__ import annotations
import sys

_ENGINES = {
    'nnunet': 'Train with nnU-Net + AugLab',
}

_HELP = (
    'Usage: uroseg train <engine> ORGAN [options]\n'
    'Engines:\n'
    '  nnunet    Train with nnU-Net + AugLab'
)


def main() -> None:
    argv = sys.argv[1:]

    if argv and argv[0] in ('-h', '--help'):
        print(_HELP)
        sys.exit(0)

    if not argv:
        print(_HELP, file=sys.stderr)
        sys.exit(1)

    engine = argv[0]
    if engine not in _ENGINES:
        print(f"Unknown engine: '{engine}'\n{_HELP}", file=sys.stderr)
        sys.exit(1)

    sys.argv = sys.argv[:1] + sys.argv[2:]

    if engine == 'nnunet':
        from uroseg.commands.train_nnunet import main as run
        run()
