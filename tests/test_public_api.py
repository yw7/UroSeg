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
