from __future__ import annotations
import sys

SUBCOMMANDS = {
    'train', 'install', 'map', 'resample', 'preview',
    'crop', 'largest_component', 'reorient', 'cpdir',
    'transform_seg2image', 'predict_nnunet', 'list',
}


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uroseg <organ|subcommand> [options]", file=sys.stderr)
        print("Run 'uroseg list' to see available organ models and subcommands.", file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'list':
        _cmd_list()
    elif cmd == 'install':
        from uroseg.commands.install import main as run
        run()
    elif cmd == 'train':
        from uroseg.commands.train import main as run
        run()
    elif cmd == 'map':
        from uroseg.commands.map_labels import main as run
        run()
    elif cmd == 'resample':
        from uroseg.commands.resample import main as run
        run()
    elif cmd == 'preview':
        from uroseg.commands.preview_jpg import main as run
        run()
    elif cmd == 'crop':
        from uroseg.commands.crop_image2seg import main as run
        run()
    elif cmd == 'largest_component':
        from uroseg.commands.largest_component import main as run
        run()
    elif cmd == 'reorient':
        from uroseg.commands.reorient_canonical import main as run
        run()
    elif cmd == 'cpdir':
        from uroseg.commands.cpdir import main as run
        run()
    elif cmd == 'transform_seg2image':
        from uroseg.commands.transform_seg2image import main as run
        run()
    elif cmd == 'predict_nnunet':
        from uroseg.commands.predict_nnunet import main as run
        run()
    elif cmd.startswith('-'):
        print(f"Unknown option: {cmd}. Run 'uroseg list' to see available commands.", file=sys.stderr)
        sys.exit(1)
    else:
        _dispatch_organ(cmd)


def _cmd_list() -> None:
    from uroseg.utils.utils import get_all_models
    models = get_all_models()
    print(f"{'Model':<22} {'Description'}")
    print('-' * 70)
    for name, m in sorted(models.items()):
        labels = ', '.join(f"{k}={v}" for k, v in m['labels'].items() if k != '0')
        print(f"  uroseg {name:<16} {m['description']}")
        print(f"  {'':16}   labels: {labels}")


def _dispatch_organ(organ: str) -> None:
    from uroseg.utils.utils import get_all_models
    available = get_all_models()
    if organ not in available:
        print(
            f"Unknown organ or subcommand: '{organ}'\n"
            f"Available organs: {', '.join(sorted(available))}\n"
            f"Subcommands: {', '.join(sorted(SUBCOMMANDS))}",
            file=sys.stderr,
        )
        sys.exit(1)
    from uroseg.commands.inference import main as run
    run()


if __name__ == '__main__':
    main()
