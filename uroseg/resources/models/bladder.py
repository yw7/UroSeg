import argparse
from uroseg.models import ModelDef
from uroseg.utils.inference_utils import add_common_inference_args, run_nnunet_predict

MODEL = ModelDef(
    name="bladder",
    description="Urinary bladder (CT)",
    weights_url="https://github.com/yw7/uroseg/releases/download/r20260101/Dataset010_Bladder_r20260101.zip",
    labels={"background": 0, "bladder": 1},
)

NNUNET_TASK = "Dataset010_Bladder"


def main():
    parser = argparse.ArgumentParser(prog='uroseg bladder')
    add_common_inference_args(parser)
    args = parser.parse_args()
    run_nnunet_predict(NNUNET_TASK, args)
