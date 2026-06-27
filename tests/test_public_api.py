def test_public_api_models():
    import uroseg
    assert hasattr(uroseg, 'Prostate')
    assert hasattr(uroseg, 'Bladder')
    assert hasattr(uroseg, 'get_model')
    assert hasattr(uroseg, 'list_models')
    assert uroseg.get_model('prostate').name == 'prostate'


def test_public_api_tools():
    import uroseg
    for name in ['map_labels', 'map_labels_dir',
                 'resample', 'resample_dir',
                 'reorient', 'reorient_dir',
                 'largest_component', 'largest_component_dir',
                 'crop', 'crop_dir',
                 'transform_seg2image', 'transform_seg2image_dir',
                 'preview', 'preview_dir',
                 'cpdir', 'cpdir_dir']:
        assert hasattr(uroseg, name), f"uroseg.{name} missing"
        assert callable(getattr(uroseg, name))


def test_public_api_inmemory():
    import uroseg
    import numpy as np

    # Image class is exported
    assert hasattr(uroseg, 'Image')
    img = uroseg.Image.load  # has .load classmethod
    assert callable(img)

    # In-memory data functions are exported and callable
    for name in ['keep_largest_component', 'resample_seg_to_image',
                 'apply_map', 'make_preview']:
        assert hasattr(uroseg, name), f"uroseg.{name} missing"
        assert callable(getattr(uroseg, name))

    # apply_map works on a numpy array (smoke test)
    data = np.array([[[0, 1, 2]]], dtype=np.uint8)
    result = uroseg.apply_map(data, {1: 10, 2: 20})
    assert result[0, 0, 1] == 10
    assert result[0, 0, 2] == 20
