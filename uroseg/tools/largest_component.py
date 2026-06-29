from __future__ import annotations
import argparse
import functools
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from tqdm.contrib.concurrent import process_map

from uroseg.utils.image import Image, save_nifti_seg
from uroseg.utils.utils import add_common_args, build_pairs, build_output_path

# Full 26-connectivity structure for 3-D images
_STRUCT26 = np.ones((3, 3, 3), dtype=np.int8)


def _largest_component_for_label(
    data: np.ndarray,
    lbl: int,
    dilate: int = 0,
    binarize: bool = False,
) -> np.ndarray:
    """Return a boolean mask containing only the largest CC for *lbl*.

    Parameters
    ----------
    data:
        Integer label array.
    lbl:
        Label value to process (ignored when *binarize* is True).
    dilate:
        Number of dilation iterations applied before CC analysis.
    binarize:
        When True the whole non-zero region is treated as one binary mask and
        the largest CC is found across all labels at once.

    Returns
    -------
    A mask (bool array) that is True where the largest component lies within
    the *original* (pre-dilation) extent of the relevant region.
    """
    if binarize:
        mask = (data != 0).astype(np.uint8)
    else:
        mask = (data == lbl).astype(np.uint8)

    if dilate > 0:
        # Explicit 6-connectivity structure (self-documenting; matches scipy default)
        struct = ndi.iterate_structure(ndi.generate_binary_structure(3, 1), dilate)
        work = ndi.binary_dilation(mask, structure=struct).astype(np.uint8)
    else:
        work = mask

    labeled, num_features = ndi.label(work, structure=_STRUCT26)
    if num_features == 0:
        return np.zeros(data.shape, dtype=bool)

    sizes = ndi.sum(mask, labeled, range(1, num_features + 1))
    largest_label = int(np.argmax(sizes)) + 1
    largest_mask = labeled == largest_label

    # Undo dilation: intersect with original (un-dilated) mask
    if dilate > 0:
        largest_mask = largest_mask & (mask > 0)

    return largest_mask


def largest_component(
    data: np.ndarray,
    labels: list[int] | None = None,
    dilate: int = 0,
    binarize: bool = False,
) -> np.ndarray:
    """Keep only the largest connected component per label (in-memory).

    Parameters
    ----------
    data:
        Integer label array (e.g. a segmentation volume).
    labels:
        Label IDs to process.  Defaults to all non-zero values.
    dilate:
        Dilation iterations applied before CC analysis.
    binarize:
        Treat all non-zero voxels as one region; find the single largest CC
        across all labels, then restore original label values within that mask.
    """
    result = np.zeros_like(data)

    if binarize:
        largest_mask = _largest_component_for_label(data, lbl=0, dilate=dilate, binarize=True)
        result = (data * largest_mask).astype(data.dtype)
    else:
        label_ids = labels if labels else [int(v) for v in np.unique(data) if v > 0]
        for label_id in label_ids:
            largest_mask = _largest_component_for_label(data, lbl=label_id, dilate=dilate)
            result[largest_mask] = label_id

    return result


def largest_component_file(
    input: Path | str,
    output: Path | str,
    labels: list[int] | None = None,
    dilate: int = 0,
    binarize: bool = False,
    out_suffix: str = "_largest",
    out_prefix: str = "",
    overwrite: bool = False,
) -> Path:
    input_path, output_path = Path(input), Path(output)
    if not output_path.suffix:
        output_path = build_output_path(input_path, output_path, out_prefix, out_suffix)
    img = Image.load(input_path)
    img.data = largest_component(img.data, labels=labels, dilate=dilate, binarize=binarize)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_nifti_seg(img.data, img.affine, img.header, str(output_path))
    return output_path


def largest_component_dir(
    input_dir: Path | str,
    output_dir: Path | str,
    labels: list[int] | None = None,
    dilate: int = 0,
    binarize: bool = False,
    out_suffix: str = "_largest",
    out_prefix: str = "",
    overwrite: bool = False,
    n_jobs: int = 1,
) -> None:
    pairs = build_pairs(input_dir, output_dir, out_suffix, out_prefix, overwrite)
    in_paths = [p[0] for p in pairs]
    out_paths = [p[1] for p in pairs]
    process_map(
        functools.partial(
            largest_component_file,
            labels=labels, dilate=dilate, binarize=binarize, overwrite=overwrite,
        ),
        in_paths, out_paths,
        max_workers=n_jobs,
        disable=False,
        desc='uroseg largest_component',
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Keep only the largest connected component per label in a segmentation.'
    )
    parser.add_argument('--seg', '-s', required=True, help='Input seg file or folder')
    parser.add_argument('--out', '-o', required=True, help='Output file or folder')
    parser.add_argument('--labels', '-l', nargs='+', type=int, default=None,
                        help='Label IDs to process (default: all non-zero)')
    parser.add_argument('--out-suffix', default='_largest', help='Output filename suffix')
    parser.add_argument('--out-prefix', default='', help='Output filename prefix')
    parser.add_argument('--binarize', action='store_true',
                        help='Binarize seg before finding largest component, then restore '
                             'original label values')
    parser.add_argument('--dilate', type=int, default=0, metavar='N',
                        help='Dilate by N voxels before connected-component analysis, '
                             'then mask back to original (default: 0 = no dilation)')
    add_common_args(parser)
    args = parser.parse_args()

    pairs = build_pairs(args.seg, args.out, args.out_suffix, args.out_prefix, args.overwrite)
    in_paths = [p[0] for p in pairs]
    out_paths = [p[1] for p in pairs]
    process_map(
        functools.partial(
            largest_component_file,
            labels=args.labels, dilate=args.dilate, binarize=args.binarize, overwrite=args.overwrite,
        ),
        in_paths, out_paths,
        max_workers=args.max_workers,
        disable=args.quiet,
        desc='uroseg largest_component',
    )
