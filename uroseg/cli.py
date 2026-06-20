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
        from uroseg.commands.list_models import main as run
        run()
    elif cmd == 'install':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.commands.install import main as run
        run()
    elif cmd == 'train':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.commands.train import main as run
        run()
    elif cmd == 'map':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.commands.map_labels import main as run
        run()
    elif cmd == 'resample':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.commands.resample import main as run
        run()
    elif cmd == 'preview':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.commands.preview_jpg import main as run
        run()
    elif cmd == 'crop':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.commands.crop_image2seg import main as run
        run()
    elif cmd == 'largest_component':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.commands.largest_component import main as run
        run()
    elif cmd == 'reorient':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.commands.reorient_canonical import main as run
        run()
    elif cmd == 'cpdir':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.commands.cpdir import main as run
        run()
    elif cmd == 'transform_seg2image':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.commands.transform_seg2image import main as run
        run()
    elif cmd == 'predict_nnunet':
        sys.argv = sys.argv[:1] + sys.argv[2:]
        from uroseg.commands.predict_nnunet import main as run
        run()
    elif cmd.startswith('-'):
        print(f"Unknown option: {cmd}. Run 'uroseg list' to see available commands.", file=sys.stderr)
        sys.exit(1)
    else:
        _dispatch_organ(cmd)


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
