from __future__ import annotations
import argparse
import functools
from pathlib import Path

from tqdm.contrib.concurrent import process_map

from uroseg.utils.image import Image, save_nifti_image
from uroseg.utils.utils import add_common_args, build_pairs, build_output_path


def resample(img: Image, mm: float | list[float] = 1.0) -> Image:
    """Resample an Image to target voxel spacing (in-memory)."""
    mm_list = [mm] if isinstance(mm, (int, float)) else list(mm)
    mm_tuple = tuple(mm_list if len(mm_list) == 3 else [mm_list[0]] * 3)
    return img.resample(mm_tuple)


def resample_file(
    input: Path | str,
    output: Path | str,
    mm: float | list[float] = 1.0,
    out_suffix: str = "_resampled",
    out_prefix: str = "",
    overwrite: bool = False,
) -> Path:
    input_path, output_path = Path(input), Path(output)
    if not output_path.suffix:
        output_path = build_output_path(input_path, output_path, out_prefix, out_suffix)
    img = resample(Image.load(input_path), mm=mm)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_nifti_image(img.data, img.affine, img.header, str(output_path))
    return output_path


def resample_dir(
    input_dir: Path | str,
    output_dir: Path | str,
    mm: float | list[float] = 1.0,
    out_suffix: str = "_resampled",
    out_prefix: str = "",
    overwrite: bool = False,
    n_jobs: int = 1,
) -> None:
    pairs = build_pairs(input_dir, output_dir, out_suffix, out_prefix, overwrite)
    in_paths = [p[0] for p in pairs]
    out_paths = [p[1] for p in pairs]
    process_map(
        functools.partial(resample_file, mm=mm, overwrite=overwrite),
        in_paths, out_paths,
        max_workers=n_jobs,
        disable=False,
        desc='uroseg resample',
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Resample image to target voxel spacing.')
    parser.add_argument('--img', '-i', required=True, help='Input image file or folder')
    parser.add_argument('--out', '-o', required=True, help='Output file or folder')
    parser.add_argument('--mm', '-m', nargs='+', type=float, default=[1.0],
                        metavar='MM',
                        help='Target voxel size in mm. One value for isotropic (default: 1), '
                             'or three values for X Y Z.')
    parser.add_argument('--spacing', nargs='+', type=float, dest='mm',
                        help=argparse.SUPPRESS)  # backwards compat alias
    parser.add_argument('--out-suffix', default='_resampled', help='Output filename suffix')
    parser.add_argument('--out-prefix', default='', help='Output filename prefix')
    add_common_args(parser)
    args = parser.parse_args()

    pairs = build_pairs(args.img, args.out, args.out_suffix, args.out_prefix, args.overwrite)
    in_paths = [p[0] for p in pairs]
    out_paths = [p[1] for p in pairs]
    process_map(
        functools.partial(resample_file, mm=args.mm, overwrite=args.overwrite),
        in_paths, out_paths,
        max_workers=args.max_workers,
        disable=args.quiet,
        desc='uroseg resample',
    )
