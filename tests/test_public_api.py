def test_public_api_models():
    import uroseg
    assert hasattr(uroseg, 'Prostate')
    assert hasattr(uroseg, 'Bladder')
    assert hasattr(uroseg, 'get_model')
    assert hasattr(uroseg, 'list_models')
    assert uroseg.get_model('prostate').name == 'prostate'


def test_public_api_inmemory():
    import uroseg
    import numpy as np

    assert hasattr(uroseg, 'Image')
    assert callable(uroseg.Image.load)

    for name in ['map_labels', 'resample', 'reorient', 'largest_component',
                 'crop', 'transform_seg2image', 'preview', 'volume']:
        assert hasattr(uroseg, name), f"uroseg.{name} missing"
        assert callable(getattr(uroseg, name))

    # map_labels smoke test on a numpy array
    data = np.array([[[0, 1, 2]]], dtype=np.uint8)
    result = uroseg.map_labels(data, {1: 10, 2: 20})
    assert result[0, 0, 1] == 10
    assert result[0, 0, 2] == 20


def test_public_api_file_tools():
    import uroseg
    for name in ['map_labels_file', 'resample_file', 'reorient_file',
                 'largest_component_file', 'crop_file', 'transform_seg2image_file',
                 'preview_file', 'volume_file', 'cpdir_file']:
        assert hasattr(uroseg, name), f"uroseg.{name} missing"
        assert callable(getattr(uroseg, name))


def test_public_api_dir_tools():
    import uroseg
    for name in ['map_labels_dir', 'resample_dir', 'reorient_dir',
                 'largest_component_dir', 'crop_dir', 'transform_seg2image_dir',
                 'preview_dir', 'volume_dir', 'cpdir_dir']:
        assert hasattr(uroseg, name), f"uroseg.{name} missing"
        assert callable(getattr(uroseg, name))
