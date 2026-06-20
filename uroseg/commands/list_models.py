from __future__ import annotations
from uroseg.utils.utils import get_all_models


def main() -> None:
    models = get_all_models()
    if not models:
        print("No models found in resources/models/")
        return
    name_w, task_w = 20, 30
    header = f"{'Model':<{name_w}} {'Task':<{task_w}} Description"
    print(header)
    print('-' * (name_w + task_w + 40))
    for name, model in sorted(models.items()):
        task = model.get('nnunet_task', 'N/A')
        desc = model.get('description', '')
        print(f"{name:<{name_w}} {task:<{task_w}} {desc}")
