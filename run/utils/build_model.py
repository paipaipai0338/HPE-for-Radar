from pathlib import Path
from importlib import import_module

from run.utils.load_config import load_config


def build_model(model_name):
    model_dirs = {
        'P4Transformer': 'models/P4Transformer',
    }

    if model_name not in model_dirs:
        raise ValueError(f"Unknown model: {model_name}")

    project_root = Path(__file__).resolve().parents[2]
    model_dir = project_root / model_dirs[model_name]
    model_config_path = model_dir / 'model_config.yaml'
    model_cfg = load_config(model_config_path)

    module_path = '.'.join(model_dirs[model_name].split('/') + [model_name])
    module = import_module(module_path)
    model_cls = getattr(module, model_name)
    model = model_cls(**model_cfg)
    return model
