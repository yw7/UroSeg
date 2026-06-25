from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ModelDef:
    name: str
    description: str
    weights_url: str
    labels: dict
