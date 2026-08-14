from app.modules.optimizer.model_configs.base import ModelConfig
from app.modules.optimizer.model_configs.chiller import CONFIG as CHILLER_CONFIG
from app.modules.optimizer.model_configs.cooling_tower import CONFIG as COOLING_TOWER_CONFIG
from app.modules.optimizer.model_configs.pump import CONFIG as PUMP_CONFIG
from app.modules.optimizer.schemas import DeviceType

MODEL_CONFIGS: dict[DeviceType, ModelConfig] = {
    config.device_type: config
    for config in (CHILLER_CONFIG, PUMP_CONFIG, COOLING_TOWER_CONFIG)
}

__all__ = ["MODEL_CONFIGS", "ModelConfig"]
