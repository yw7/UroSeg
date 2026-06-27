from importlib.metadata import version, PackageNotFoundError

# Models
from uroseg.models import get_model, list_models
from uroseg.models.prostate import Prostate
from uroseg.models.bladder import Bladder

# In-memory data layer
from uroseg.utils.image import Image
from uroseg.tools.largest_component import keep_largest_component
from uroseg.tools.transform_seg2image import resample_seg_to_image
from uroseg.tools.map_labels import apply_map
from uroseg.tools.preview import make_preview

# Tools — single file
from uroseg.tools.map_labels import map_labels
from uroseg.tools.resample import resample
from uroseg.tools.preview import preview
from uroseg.tools.crop import crop
from uroseg.tools.largest_component import largest_component
from uroseg.tools.reorient import reorient
from uroseg.tools.transform_seg2image import transform_seg2image
from uroseg.tools.cpdir import cpdir

# Tools — directory (multiprocessing)
from uroseg.tools.map_labels import map_labels_dir
from uroseg.tools.resample import resample_dir
from uroseg.tools.preview import preview_dir
from uroseg.tools.crop import crop_dir
from uroseg.tools.largest_component import largest_component_dir
from uroseg.tools.reorient import reorient_dir
from uroseg.tools.transform_seg2image import transform_seg2image_dir
from uroseg.tools.cpdir import cpdir_dir

try:
    __version__ = version("uroseg")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    'get_model', 'list_models', 'Prostate', 'Bladder',
    'Image',
    'keep_largest_component', 'resample_seg_to_image', 'apply_map', 'make_preview',
    'map_labels', 'map_labels_dir',
    'resample', 'resample_dir',
    'preview', 'preview_dir',
    'crop', 'crop_dir',
    'largest_component', 'largest_component_dir',
    'reorient', 'reorient_dir',
    'transform_seg2image', 'transform_seg2image_dir',
    'cpdir', 'cpdir_dir',
]
