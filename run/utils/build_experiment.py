from datetime import datetime
from dataclasses import asdict
from pathlib import Path
import inspect
import shutil
import yaml


def save_radar_config(config_dir, radar_config):
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    radar_config_dict = asdict(radar_config)
    virtual_channel_positions = getattr(
        radar_config,
        "virtual_channel_positions",
        None,
    )
    if virtual_channel_positions is not None:
        radar_config_dict["virtual_channel_positions"] = (
            virtual_channel_positions.tolist()
        )
    with (config_dir / "radar_config.yaml").open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            radar_config_dict,
            file,
            allow_unicode=True,
            sort_keys=False,
        )


def build_experiment(output_root, model_name, source_config_path, model):
    # {output_root}/
    # └── {model_name}/
    #     └── 20260626_230202/
    #         ├── checkpoint/
    #         │   ├── best.pth
    #         │   └── last.pth
    #         ├── log/
    #         │   └── log.txt
    #         ├── fig/
    #         │   ├── training_curve.png
    #         │   ├── epoch_001.png
    #         │   ├── epoch_002.png
    #         │   ├── epoch_003.png
    #         │   └── epoch_00{T}.png
    #         └── config/
    #             ├── config.yaml
    #             ├── model_config.yaml
    #             ├── radar_config.yaml
    #             └── (model_name).py
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_dir = (Path(output_root) / model_name / timestamp)

    paths = {
        "root": experiment_dir,
        "checkpoint": experiment_dir / "checkpoint",
        "log": experiment_dir / "log",
        "fig": experiment_dir / "fig",
        "config": experiment_dir / "config",
    }

    for path in paths.values():
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

    source_config_path = Path(source_config_path)

    shutil.copy2(source_config_path, paths["config"] / "config.yaml")

    model_source_path = Path(inspect.getfile(model.__class__))
    shutil.copy2(model_source_path, paths["config"] / model_source_path.name)

    model_config_path = model_source_path.parent / "model_config.yaml"
    if model_config_path.exists():
        shutil.copy2(model_config_path, paths["config"] / "model_config.yaml")

    return paths
