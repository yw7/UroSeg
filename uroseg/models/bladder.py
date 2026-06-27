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
    import argparse
    from uroseg.nnunet.predict import add_inference_args, run_predict_cli
    parser = argparse.ArgumentParser(prog='uroseg bladder', description=Bladder.description)
    add_inference_args(parser)
    args = parser.parse_args()
    run_predict_cli(Bladder(), args)
