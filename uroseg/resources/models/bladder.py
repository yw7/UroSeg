from uroseg.models import ModelDef

MODEL = ModelDef(
    name="bladder",
    description="Urinary bladder (CT)",
    weights_url="https://github.com/yw7/uroseg/releases/download/r20260101/Dataset010_Bladder_r20260101.zip",
    labels={"background": 0, "bladder": 1},
)

NNUNET_TASK = "Dataset010_Bladder"


def inference(img, predict):
    return predict(img)
