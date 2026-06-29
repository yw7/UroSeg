from __future__ import annotations
import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        from uroseg.tools.list_models import show_help
        show_help()
        return

    cmd = sys.argv[1]

    if cmd == 'list':
        from uroseg.tools.list_models import show_help
        show_help()
        return
    elif cmd == 'install':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        _run_install()
    elif cmd == 'train':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        _train_dispatch_wrapper()
    elif cmd == 'map':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.tools.map_labels import main as run
        run()
    elif cmd == 'resample':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.tools.resample import main as run
        run()
    elif cmd == 'preview':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.tools.preview import main as run
        run()
    elif cmd == 'crop':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.tools.crop import main as run
        run()
    elif cmd == 'largest_component':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.tools.largest_component import main as run
        run()
    elif cmd == 'reorient':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.tools.reorient import main as run
        run()
    elif cmd == 'cpdir':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.tools.cpdir import main as run
        run()
    elif cmd == 'transform_seg2image':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.tools.transform_seg2image import main as run
        run()
    elif cmd == 'volume':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.tools.volume import main as run
        run()
    elif cmd.startswith('-'):
        print(f"Unknown option: {cmd}. Run 'uroseg --help' for help.", file=sys.stderr)
        sys.exit(1)
    else:
        _dispatch_model(cmd)


def _run_install() -> None:
    import argparse
    from uroseg.models import get_model, list_models
    from uroseg.utils.utils import resolve_data_path, data_dir_help
    parser = argparse.ArgumentParser(description='Download and install UroSeg model weights.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', nargs='+', metavar='MODEL',
                       help='One or more organ model names (e.g. prostate bladder)')
    group.add_argument('--all', action='store_true', help='Install all available models')
    parser.add_argument('--data-dir', default=None, help=data_dir_help())
    args = parser.parse_args()
    data_path = resolve_data_path(args.data_dir)
    names = list_models() if args.all else args.model
    print(f"Installing {len(names)} model(s) to {data_path}...")
    for name in names:
        get_model(name).install(data_path)


def _train_dispatch_wrapper() -> None:
    _ENGINES = {'nnunet': 'Train with nnU-Net + AugLab'}
    _HELP = 'Usage: uroseg train <engine> ORGAN [options]\nEngines:\n  nnunet    Train with nnU-Net + AugLab'
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
        from uroseg.nnunet.train import main as run
        run()


def _dispatch_model(model: str) -> None:
    import importlib
    from uroseg.models import list_models
    if model not in list_models():
        print(
            f"Unknown model or subcommand: '{model}'\n"
            f"Run 'uroseg --help' to see available models and commands.",
            file=sys.stderr,
        )
        sys.exit(1)
    sys.argv = sys.argv[:1] + sys.argv[2:]
    mod = importlib.import_module(f'uroseg.models.{model}')
    mod.main()


if __name__ == '__main__':
    main()
