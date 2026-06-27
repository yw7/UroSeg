from __future__ import annotations
from uroseg.models.base import NNUNetSegModel


class Bladder(NNUNetSegModel):
    name = "bladder"
    description = "Urinary bladder (CT)"
    weights_url = "https://github.com/yw7/uroseg/releases/download/r20260101/Dataset010_Bladder_r20260101.zip"
    labels = {"background": 0, "bladder": 1}
    nnunet_task = "Dataset010_Bladder"


MODEL = Bladder()
NNUNET_TASK = Bladder.nnunet_task


def main() -> None:
    Bladder.cli_main()
