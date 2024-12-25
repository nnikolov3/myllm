from typing import Dict, Any, Optional
import os
import yaml
import logging
from dataclasses import dataclass
from pydantic import BaseModel, validator
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelConfig(BaseModel):
    """Model configuration schema."""
    name: str
    weight: float
    timeout: int

    @validator('weight')
    def weight_must_be_valid(cls, v):
        """Validate weight is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError('Weight must be between 0 and 1')
        return v

    @validator('timeout')
    def timeout_must_be_positive(cls, v):
        """Validate timeout is positive."""
        if v <= 0:
            raise ValueError('Timeout must be positive')
        return v

class SystemConfig(BaseModel):
    """System configuration schema."""
    batch_size: int
    max_tokens: int
    cache_size: int
    gpu_memory_threshold: float

    @validator('batch_size', 'max_tokens', 'cache_size')
    def value_must_be_positive(cls, v):
        """Validate numeric values are positive."""
        if v <= 0:
            raise ValueError('Value must be positive')
        return v

    @validator('gpu_memory_threshold')
    def threshold_must_be_valid(cls, v):
        """Validate GPU memory threshold is between 0 and 1."""
        if not 0 < v < 1:
            raise ValueError('GPU memory threshold must be between 0 and 1')
        return v

class Config(BaseModel):
    """Main configuration schema."""
    models: Dict[str, ModelConfig]
    system: SystemConfig

@dataclass
class ConfigManager:
    """Manages application configuration."""
    
    config_path: str
    _config: Optional[Config] = None

    def __post_init__(self):
        """Initialize configuration after instance creation."""
        self.reload_config()

    def reload_config(self) -> None:
        """Reload configuration from file."""
        try:
            config_path = Path(self.config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)

            self._config = Config(**config_dict)
            self._validate_model_weights()
            logger.info("Configuration loaded successfully")

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def _validate_model_weights(self) -> None:
        """Validate that model weights sum to approximately 1."""
        if self._config:
            weight_sum = sum(model.weight for model in self._config.models.values())
            if not 0.99 <= weight_sum <= 1.01:  # Allow for small floating point differences
                raise ValueError(f"Model weights must sum to 1, got {weight_sum}")

    @property
    def config(self) -> Config:
        """Get the current configuration."""
        if not self._config:
            self.reload_config()
        return self._config

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        return self.config.models.get(model_name)

    def get_system_config(self) -> SystemConfig:
        """Get system configuration."""
        return self.config.system

    def validate_model_compatibility(self, installed_models: Dict[str, str]) -> Dict[str, bool]:
        """Validate installed models against configuration."""
        compatibility = {}
        for model_name, model_config in self.config.models.items():
            compatibility[model_name] = model_name in installed_models
            if not compatibility[model_name]:
                logger.warning(f"Configured model {model_name} not found in installed models")
        return compatibility

    def export_config(self) -> Dict[str, Any]:
        """Export current configuration as dictionary."""
        return self.config.dict()

    def get_active_models(self) -> Dict[str, ModelConfig]:
        """Get all active model configurations."""
        return self.config.models

    @staticmethod
    def create_default_config(path: str) -> None:
        """Create a default configuration file."""
        default_config = {
            'models': {
                'mistral': {
                    'name': 'mistral:latest',
                    'weight': 0.33,
                    'timeout': 30
                },
                'llama': {
                    'name': 'llama3.2:latest',
                    'weight': 0.33,
                    'timeout': 30
                },
                'granite': {
                    'name': 'granite3.1-dense:8b',
                    'weight': 0.34,
                    'timeout': 30
                }
            },
            'system': {
                'batch_size': 10,
                'max_tokens': 2048,
                'cache_size': 1000,
                'gpu_memory_threshold': 0.8
            }
        }

        config_path = Path(path)
        if not config_path.exists():
            with open(config_path, 'w') as f:
                yaml.safe_dump(default_config, f)
            logger.info(f"Created default configuration at {path}")
        else:
            logger.warning(f"Configuration file already exists at {path}")

# Usage example
if __name__ == "__main__":
    # Create default config if it doesn't exist
    ConfigManager.create_default_config("config.yaml")
    
    # Initialize config manager
    config_manager = ConfigManager("config.yaml")
    
    # Get configurations
    system_config = config_manager.get_system_config()
    mistral_config = config_manager.get_model_config("mistral")
    
    # Print configurations
    print(f"System config: {system_config}")
    print(f"Mistral config: {mistral_config}")