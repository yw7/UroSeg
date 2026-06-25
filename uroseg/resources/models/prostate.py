import argparse
from uroseg.models import ModelDef
from uroseg.utils.inference_utils import add_common_inference_args, run_nnunet_predict

MODEL = ModelDef(
    name="prostate",
    description="Prostate: whole prostate (1), transition zone (2), peripheral zone (3)",
    weights_url="https://github.com/yw7/uroseg/releases/download/r20260101/Dataset101_Prostate_r20260101.zip",
    labels={"background": 0, "prostate": [1, 2, 3], "prostate_tz": 2, "prostate_pz": 3},
)

NNUNET_TASK = "Dataset101_Prostate"


def main():
    parser = argparse.ArgumentParser(prog='uroseg prostate')
    add_common_inference_args(parser)
    args = parser.parse_args()
    run_nnunet_predict(NNUNET_TASK, args)
