from uroseg.models import ModelDef

MODEL = ModelDef(
    name="prostate",
    description="Prostate: whole prostate (1), transition zone (2), peripheral zone (3)",
    weights_url="https://github.com/yw7/uroseg/releases/download/r20260101/Dataset101_Prostate_r20260101.zip",
    labels={"background": 0, "prostate": [1, 2, 3], "prostate_tz": 2, "prostate_pz": 3},
)

NNUNET_TASK = "Dataset101_Prostate"


def inference(img, predict):
    return predict(img)
