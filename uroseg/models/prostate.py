from __future__ import annotations
from uroseg.models.base import NNUNetSegModel


class Prostate(NNUNetSegModel):
    name = "prostate"
    description = "Prostate: whole (1), transition zone (2), peripheral zone (3)"
    weights_url = "https://github.com/yw7/UroSeg/releases/download/20260627/Dataset101_Prostate__nnUNetTrainerDAExtGPU__nnUNetPlans__3d_fullres__fold_0_20260627.zip"
    labels = {"background": 0, "prostate": [1, 2, 3], "prostate_tz": 2, "prostate_pz": 3}
    nnunet_task = "Dataset101_Prostate"
    post_largest_component = True


# Backward-compat aliases (used by existing consumers of resources/models/prostate)
MODEL = Prostate()
NNUNET_TASK = Prostate.nnunet_task


def main() -> None:
    import argparse
    from uroseg.nnunet.predict import add_inference_args, run_predict_cli
    parser = argparse.ArgumentParser(prog='uroseg prostate', description=Prostate.description)
    add_inference_args(parser)
    args = parser.parse_args()
    run_predict_cli(Prostate(), args)
