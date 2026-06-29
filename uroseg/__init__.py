from importlib.metadata import version, PackageNotFoundError

# Models
from uroseg.models import get_model, list_models
from uroseg.models.prostate import Prostate
from uroseg.models.bladder import Bladder

# In-memory data layer
from uroseg.utils.image import Image
from uroseg.tools.largest_component import largest_component
from uroseg.tools.transform_seg2image import transform_seg2image
from uroseg.tools.map_labels import map_labels
from uroseg.tools.preview import preview
from uroseg.tools.resample import resample
from uroseg.tools.reorient import reorient
from uroseg.tools.crop import crop
from uroseg.tools.volume import volume

# Tools — single file
from uroseg.tools.map_labels import map_labels_file
from uroseg.tools.resample import resample_file
from uroseg.tools.preview import preview_file
from uroseg.tools.crop import crop_file
from uroseg.tools.largest_component import largest_component_file
from uroseg.tools.reorient import reorient_file
from uroseg.tools.transform_seg2image import transform_seg2image_file
from uroseg.tools.cpdir import cpdir_file
from uroseg.tools.volume import volume_file

# Tools — directory (multiprocessing)
from uroseg.tools.map_labels import map_labels_dir
from uroseg.tools.resample import resample_dir
from uroseg.tools.preview import preview_dir
from uroseg.tools.crop import crop_dir
from uroseg.tools.largest_component import largest_component_dir
from uroseg.tools.reorient import reorient_dir
from uroseg.tools.transform_seg2image import transform_seg2image_dir
from uroseg.tools.cpdir import cpdir_dir
from uroseg.tools.volume import volume_dir

try:
    __version__ = version("uroseg")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    'get_model', 'list_models', 'Prostate', 'Bladder',
    'Image',
    # in-memory
    'map_labels', 'resample', 'reorient', 'largest_component',
    'crop', 'preview', 'transform_seg2image', 'volume',
    # single file
    'map_labels_file', 'resample_file', 'reorient_file', 'largest_component_file',
    'crop_file', 'preview_file', 'transform_seg2image_file', 'volume_file', 'cpdir_file',
    # directory
    'map_labels_dir', 'resample_dir', 'reorient_dir', 'largest_component_dir',
    'crop_dir', 'preview_dir', 'transform_seg2image_dir', 'volume_dir', 'cpdir_dir',
]
