from app.modules.optimizer.dataset_configs.base import DatasetConfig
from app.modules.optimizer.dataset_configs.chiller import CONFIG as CHILLER_CONFIG
from app.modules.optimizer.dataset_configs.cooling_tower import CONFIG as COOLING_TOWER_CONFIG
from app.modules.optimizer.dataset_configs.pump import CONFIG as PUMP_CONFIG
from app.modules.optimizer.schemas import DeviceType

DATASET_CONFIGS: dict[DeviceType, DatasetConfig] = {
    config.device_type: config
    for config in (CHILLER_CONFIG, PUMP_CONFIG, COOLING_TOWER_CONFIG)
}

__all__ = ["DATASET_CONFIGS", "DatasetConfig"]
