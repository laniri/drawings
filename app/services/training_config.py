"""
Training Configuration Management Service for Children's Drawing Anomaly Detection System

This module provides Hydra-based configuration management for training parameters,
validation, and parameter sweep capabilities.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import yaml
from omegaconf import DictConfig, OmegaConf

from app.core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class TrainingEnvironment(str, Enum):
    """Training environment options."""
    LOCAL = "local"
    SAGEMAKER = "sagemaker"


class OptimizerType(str, Enum):
    """Optimizer types for training."""
    ADAM = "adam"
    SGD = "sgd"
    ADAMW = "adamw"


class SchedulerType(str, Enum):
    """Learning rate scheduler types."""
    NONE = "none"
    STEP = "step"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    encoder_dim: int = 512
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    latent_dim: int = 32
    dropout_rate: float = 0.1
    activation: str = "relu"
    batch_norm: bool = True


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""
    type: OptimizerType = OptimizerType.ADAM
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    momentum: float = 0.9  # For SGD
    betas: tuple = field(default_factory=lambda: (0.9, 0.999))  # For Adam/AdamW


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler."""
    type: SchedulerType = SchedulerType.NONE
    step_size: int = 30  # For StepLR
    gamma: float = 0.1  # For StepLR and ExponentialLR
    T_max: int = 100  # For CosineAnnealingLR


@dataclass
class DataConfig:
    """Configuration for data handling."""
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    train_split: float = 0.7
    validation_split: float = 0.2
    test_split: float = 0.1
    stratify_by_age: bool = True
    age_group_size: float = 1.0


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    # Job identification
    job_name: str = "autoencoder_training"
    experiment_name: str = "drawing_anomaly_detection"
    
    # Environment
    environment: TrainingEnvironment = TrainingEnvironment.LOCAL
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    mixed_precision: bool = False
    
    # Data
    dataset_folder: str = ""
    metadata_file: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    
    # Model
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Training
    epochs: int = 100
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    gradient_clip_norm: Optional[float] = 1.0
    
    # Optimization
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Validation and checkpointing
    validation_frequency: int = 1  # Validate every N epochs
    checkpoint_frequency: int = 10  # Save checkpoint every N epochs
    save_best_only: bool = True
    
    # Logging and monitoring
    log_frequency: int = 10  # Log every N batches
    save_plots: bool = True
    plot_frequency: int = 5  # Save plots every N epochs
    
    # SageMaker specific
    sagemaker_instance_type: str = "ml.m5.large"
    sagemaker_instance_count: int = 1
    sagemaker_volume_size: int = 30  # GB
    sagemaker_max_runtime: int = 86400  # seconds (24 hours)
    
    # Output paths
    output_dir: str = "outputs"
    model_save_path: str = "models"
    log_save_path: str = "logs"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate data splits
        total_split = self.data.train_split + self.data.validation_split + self.data.test_split
        if abs(total_split - 1.0) > 0.01:
            raise ValidationError(f"Data splits must sum to 1.0, got {total_split}")
        
        # Validate ranges
        if not (0.0 < self.optimizer.learning_rate <= 1.0):
            raise ValidationError(f"Learning rate must be in (0, 1], got {self.optimizer.learning_rate}")
        
        if not (1 <= self.data.batch_size <= 1024):
            raise ValidationError(f"Batch size must be in [1, 1024], got {self.data.batch_size}")
        
        if not (1 <= self.epochs <= 10000):
            raise ValidationError(f"Epochs must be in [1, 10000], got {self.epochs}")
        
        if not (0.0 <= self.model.dropout_rate < 1.0):
            raise ValidationError(f"Dropout rate must be in [0, 1), got {self.model.dropout_rate}")
        
        # Validate paths
        if self.dataset_folder and not Path(self.dataset_folder).exists():
            raise ValidationError(f"Dataset folder not found: {self.dataset_folder}")
        
        if self.metadata_file and not Path(self.metadata_file).exists():
            raise ValidationError(f"Metadata file not found: {self.metadata_file}")


@dataclass
class ParameterSweepConfig:
    """Configuration for parameter sweeps."""
    learning_rates: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1])
    batch_sizes: List[int] = field(default_factory=lambda: [16, 32, 64])
    hidden_dims: List[List[int]] = field(default_factory=lambda: [
        [256, 128, 64],
        [512, 256, 128],
        [128, 64, 32]
    ])
    latent_dims: List[int] = field(default_factory=lambda: [16, 32, 64])
    dropout_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2])
    
    max_trials: int = 20
    optimization_metric: str = "validation_loss"
    optimization_direction: str = "minimize"  # "minimize" or "maximize"


class TrainingConfigManager:
    """Manager for training configurations with Hydra integration."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the training config manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir or Path("configs")
        self.config_dir.mkdir(exist_ok=True)
        
        # Create default config directories
        (self.config_dir / "model").mkdir(exist_ok=True)
        (self.config_dir / "data").mkdir(exist_ok=True)
        (self.config_dir / "optimizer").mkdir(exist_ok=True)
        (self.config_dir / "experiment").mkdir(exist_ok=True)
        
        logger.info(f"TrainingConfigManager initialized with config_dir: {self.config_dir}")
    
    def create_default_configs(self):
        """Create default configuration files."""
        
        # Main config
        main_config = {
            "defaults": [
                "model: autoencoder",
                "data: default",
                "optimizer: adam",
                "_self_"
            ],
            "job_name": "autoencoder_training",
            "experiment_name": "drawing_anomaly_detection",
            "environment": "local",
            "epochs": 100,
            "device": "auto"
        }
        
        # Model configs
        autoencoder_config = {
            "encoder_dim": 512,
            "hidden_dims": [256, 128, 64],
            "latent_dim": 32,
            "dropout_rate": 0.1,
            "activation": "relu",
            "batch_norm": True
        }
        
        # Data config
        data_config = {
            "batch_size": 32,
            "num_workers": 4,
            "train_split": 0.7,
            "validation_split": 0.2,
            "test_split": 0.1,
            "stratify_by_age": True,
            "age_group_size": 1.0
        }
        
        # Optimizer configs
        adam_config = {
            "type": "adam",
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "betas": [0.9, 0.999]
        }
        
        sgd_config = {
            "type": "sgd",
            "learning_rate": 0.01,
            "weight_decay": 1e-4,
            "momentum": 0.9
        }
        
        # Save configs
        self._save_yaml_config(self.config_dir / "config.yaml", main_config)
        self._save_yaml_config(self.config_dir / "model" / "autoencoder.yaml", autoencoder_config)
        self._save_yaml_config(self.config_dir / "data" / "default.yaml", data_config)
        self._save_yaml_config(self.config_dir / "optimizer" / "adam.yaml", adam_config)
        self._save_yaml_config(self.config_dir / "optimizer" / "sgd.yaml", sgd_config)
        
        logger.info("Default configuration files created")
    
    def _save_yaml_config(self, path: Path, config: Dict[str, Any]):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> TrainingConfig:
        """
        Load training configuration from file or create default.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            TrainingConfig object
        """
        if config_path is None:
            # Create default config
            return TrainingConfig()
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
            else:
                raise ValidationError(f"Unsupported config format: {config_path.suffix}")
            
            return self._dict_to_config(config_dict)
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {str(e)}")
            raise ValidationError(f"Invalid configuration file: {str(e)}")
    
    def save_config(self, config: TrainingConfig, config_path: Union[str, Path]):
        """
        Save training configuration to file.
        
        Args:
            config: TrainingConfig object to save
            config_path: Path where to save the configuration
        """
        config_path = Path(config_path)
        config_dict = self._config_to_dict(config)
        
        try:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                with open(config_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            else:
                raise ValidationError(f"Unsupported config format: {config_path.suffix}")
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {str(e)}")
            raise ValidationError(f"Failed to save configuration: {str(e)}")
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> TrainingConfig:
        """Convert dictionary to TrainingConfig object."""
        # Handle enum conversions
        if 'environment' in config_dict and isinstance(config_dict['environment'], str):
            config_dict['environment'] = TrainingEnvironment(config_dict['environment'])
        
        # Handle nested configurations
        if 'model' in config_dict and isinstance(config_dict['model'], dict):
            config_dict['model'] = ModelConfig(**config_dict['model'])
        
        if 'data' in config_dict and isinstance(config_dict['data'], dict):
            config_dict['data'] = DataConfig(**config_dict['data'])
        
        if 'optimizer' in config_dict and isinstance(config_dict['optimizer'], dict):
            optimizer_dict = config_dict['optimizer'].copy()
            if 'type' in optimizer_dict and isinstance(optimizer_dict['type'], str):
                optimizer_dict['type'] = OptimizerType(optimizer_dict['type'])
            config_dict['optimizer'] = OptimizerConfig(**optimizer_dict)
        
        if 'scheduler' in config_dict and isinstance(config_dict['scheduler'], dict):
            scheduler_dict = config_dict['scheduler'].copy()
            if 'type' in scheduler_dict and isinstance(scheduler_dict['type'], str):
                scheduler_dict['type'] = SchedulerType(scheduler_dict['type'])
            config_dict['scheduler'] = SchedulerConfig(**scheduler_dict)
        
        return TrainingConfig(**config_dict)
    
    def _config_to_dict(self, config: TrainingConfig) -> Dict[str, Any]:
        """Convert TrainingConfig object to dictionary."""
        import json
        
        def serialize_value(value):
            """Recursively serialize values, handling enums properly."""
            if isinstance(value, Enum):
                return value.value
            elif hasattr(value, '__dict__'):
                # Handle dataclass or object with attributes
                result = {}
                for attr_name, attr_value in value.__dict__.items():
                    result[attr_name] = serialize_value(attr_value)
                return result
            elif isinstance(value, (list, tuple)):
                return [serialize_value(item) for item in value]
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            else:
                # Try to JSON serialize to check if it's serializable
                try:
                    json.dumps(value)
                    return value
                except (TypeError, ValueError):
                    # If not serializable, convert to string
                    return str(value)
        
        config_dict = {}
        for field_name, field_value in config.__dict__.items():
            config_dict[field_name] = serialize_value(field_value)
        
        return config_dict
    
    def validate_config(self, config: TrainingConfig) -> Dict[str, Any]:
        """
        Validate training configuration and return validation results.
        
        Args:
            config: TrainingConfig to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        try:
            # This will raise ValidationError if invalid
            config._validate_config()
        except ValidationError as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(str(e))
        
        # Additional warnings and recommendations
        if config.data.batch_size > 128:
            validation_result['warnings'].append(
                f"Large batch size ({config.data.batch_size}) may require more memory"
            )
        
        if config.optimizer.learning_rate > 0.1:
            validation_result['warnings'].append(
                f"High learning rate ({config.optimizer.learning_rate}) may cause instability"
            )
        
        if config.epochs > 500:
            validation_result['warnings'].append(
                f"Many epochs ({config.epochs}) may lead to overfitting"
            )
        
        # Recommendations
        if config.early_stopping_patience < 5:
            validation_result['recommendations'].append(
                "Consider increasing early stopping patience for better convergence"
            )
        
        if not config.gradient_clip_norm:
            validation_result['recommendations'].append(
                "Consider enabling gradient clipping for training stability"
            )
        
        return validation_result
    
    def create_parameter_sweep(
        self, 
        base_config: TrainingConfig, 
        sweep_config: ParameterSweepConfig
    ) -> List[TrainingConfig]:
        """
        Create parameter sweep configurations.
        
        Args:
            base_config: Base configuration to modify
            sweep_config: Parameter sweep configuration
            
        Returns:
            List of TrainingConfig objects for the sweep
        """
        import itertools
        
        # Create parameter combinations
        param_combinations = list(itertools.product(
            sweep_config.learning_rates,
            sweep_config.batch_sizes,
            sweep_config.hidden_dims,
            sweep_config.latent_dims,
            sweep_config.dropout_rates
        ))
        
        # Limit to max_trials
        if len(param_combinations) > sweep_config.max_trials:
            import random
            param_combinations = random.sample(param_combinations, sweep_config.max_trials)
        
        # Create configurations
        sweep_configs = []
        for i, (lr, batch_size, hidden_dims, latent_dim, dropout_rate) in enumerate(param_combinations):
            # Create a copy of base config
            config_dict = self._config_to_dict(base_config)
            
            # Modify parameters
            config_dict['job_name'] = f"{base_config.job_name}_sweep_{i:03d}"
            config_dict['optimizer']['learning_rate'] = lr
            config_dict['data']['batch_size'] = batch_size
            config_dict['model']['hidden_dims'] = hidden_dims
            config_dict['model']['latent_dim'] = latent_dim
            config_dict['model']['dropout_rate'] = dropout_rate
            
            # Create new config
            sweep_config_obj = self._dict_to_config(config_dict)
            sweep_configs.append(sweep_config_obj)
        
        logger.info(f"Created {len(sweep_configs)} configurations for parameter sweep")
        return sweep_configs
    
    def get_hydra_config(self, config: TrainingConfig) -> DictConfig:
        """
        Convert TrainingConfig to Hydra DictConfig.
        
        Args:
            config: TrainingConfig to convert
            
        Returns:
            Hydra DictConfig object
        """
        config_dict = self._config_to_dict(config)
        return OmegaConf.create(config_dict)


# Global config manager instance
_training_config_manager = None


def get_training_config_manager() -> TrainingConfigManager:
    """Get the global training config manager instance."""
    global _training_config_manager
    if _training_config_manager is None:
        _training_config_manager = TrainingConfigManager()
    return _training_config_manager