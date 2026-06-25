from dataclasses import fields
from uroseg.models import ModelDef
import uroseg.resources.models.prostate as prostate_mod
import uroseg.resources.models.bladder as bladder_mod


def test_modeldef_has_four_fields():
    names = {f.name for f in fields(ModelDef)}
    assert names == {'name', 'description', 'weights_url', 'labels'}


def test_modeldef_is_dataclass():
    m = ModelDef(name='x', description='d', weights_url='u', labels={})
    assert m.name == 'x'
    assert m.description == 'd'
    assert m.weights_url == 'u'
    assert m.labels == {}


def test_prostate_model_attributes():
    assert isinstance(prostate_mod.MODEL, ModelDef)
    assert prostate_mod.MODEL.name == 'prostate'
    assert isinstance(prostate_mod.NNUNET_TASK, str)
    assert callable(prostate_mod.main)


def test_bladder_model_attributes():
    assert isinstance(bladder_mod.MODEL, ModelDef)
    assert bladder_mod.MODEL.name == 'bladder'
    assert isinstance(bladder_mod.NNUNET_TASK, str)
    assert callable(bladder_mod.main)


def test_prostate_labels_have_background():
    assert 'background' in prostate_mod.MODEL.labels
    assert isinstance(prostate_mod.MODEL.labels['prostate'], list)


def test_bladder_labels_have_background():
    assert 'background' in bladder_mod.MODEL.labels
    assert isinstance(bladder_mod.MODEL.labels['bladder'], int)


def test_prostate_main_is_callable():
    assert callable(prostate_mod.main)
