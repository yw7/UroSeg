from __future__ import annotations
from uroseg.models.base import SegModel, NNUNetSegModel
from uroseg.models.prostate import Prostate
from uroseg.models.bladder import Bladder

_REGISTRY: dict[str, type[SegModel]] = {
    cls.name: cls for cls in [Prostate, Bladder]
}


def get_model(name: str) -> SegModel:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model: {name!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]()


def list_models() -> list[str]:
    return list(_REGISTRY)
