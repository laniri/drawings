"""
Model Export and Deployment Service for Children's Drawing Anomaly Detection System

This module provides model parameter export in production-compatible format, model validation
and compatibility checking, and deployment API endpoints for model loading.
"""

import json
import logging
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import tempfile
import pickle

import torch
import torch.nn as nn
import numpy as np
from sqlalchemy.orm import Session

from app.models.database import TrainingJob, TrainingReport, AgeGroupModel
from app.services.model_manager import AutoencoderModel
from app.core.exceptions import ValidationError
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class ModelDeploymentError(Exception):
    """Base exception for model deployment errors."""
    pass


class ModelExportError(ModelDeploymentError):
    """Raised when model export fails."""
    pass


class ModelValidationError(ModelDeploymentError):
    """Raised when model validation fails."""
    pass


class ModelCompatibilityError(ModelDeploymentError):
    """Raised when model compatibility check fails."""
    pass


@dataclass
class ModelExportMetadata:
    """Metadata for exported model."""
    model_id: str
    export_timestamp: datetime
    training_job_id: int
    model_type: str
    model_version: str
    architecture_hash: str
    parameter_count: int
    input_dimension: int
    output_dimension: int
    age_group_min: float
    age_group_max: float
    training_metrics: Dict[str, Any]
    compatibility_version: str
    export_format: str
    file_size_bytes: int
    checksum: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['export_timestamp'] = self.export_timestamp.isoformat()
        return result


@dataclass
class ModelDeploymentConfig:
    """Configuration for model deployment."""
    model_export_path: str
    age_group_min: float
    age_group_max: float
    replace_existing: bool = False
    validate_before_deployment: bool = True
    backup_existing: bool = True
    deployment_environment: str = "production"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ModelExporter:
    """Service for exporting trained models in production-compatible format."""
    
    def __init__(self):
        self.settings = get_settings()
        self.export_dir = Path("exports/models")
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported export formats
        self.supported_formats = ["pytorch", "onnx", "pickle"]
        self.default_format = "pytorch"
        
        logger.info("Model Exporter initialized")
    
    def export_model_from_training_job(self, 
                                     training_job_id: int,
                                     age_group_min: float,
                                     age_group_max: float,
                                     export_format: str = "pytorch",
                                     db: Session = None) -> ModelExportMetadata:
        """
        Export model from completed training job.
        
        Args:
            training_job_id: ID of the training job
            age_group_min: Minimum age for the age group
            age_group_max: Maximum age for the age group
            export_format: Export format (pytorch, onnx, pickle)
            db: Database session
            
        Returns:
            Model export metadata
            
        Raises:
            ModelExportError: If export fails
        """
        try:
            logger.info(f"Exporting model from training job {training_job_id}")
            
            if export_format not in self.supported_formats:
                raise ModelExportError(f"Unsupported export format: {export_format}")
            
            # Get training job and report
            if db:
                training_job = db.query(TrainingJob).filter(TrainingJob.id == training_job_id).first()
                if not training_job:
                    raise ModelExportError(f"Training job {training_job_id} not found")
                
                training_report = db.query(TrainingReport).filter(
                    TrainingReport.training_job_id == training_job_id
                ).first()
                if not training_report:
                    raise ModelExportError(f"Training report for job {training_job_id} not found")
            else:
                # For testing without database
                training_job = None
                training_report = None
            
            # Load model from training artifacts
            model, model_info = self._load_model_from_training_artifacts(
                training_job_id, training_job, training_report
            )
            
            # Generate export metadata
            export_metadata = self._create_export_metadata(
                training_job_id=training_job_id,
                model=model,
                model_info=model_info,
                age_group_min=age_group_min,
                age_group_max=age_group_max,
                export_format=export_format,
                training_report=training_report
            )
            
            # Export model in specified format
            export_path = self._export_model_file(model, export_metadata, export_format)
            
            # Update metadata with file information
            export_metadata.file_size_bytes = export_path.stat().st_size
            export_metadata.checksum = self._calculate_file_checksum(export_path)
            
            # Save metadata file
            metadata_path = export_path.parent / f"{export_metadata.model_id}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(export_metadata.to_dict(), f, indent=2)
            
            logger.info(f"Model exported successfully: {export_path}")
            
            return export_metadata
            
        except Exception as e:
            logger.error(f"Failed to export model from training job {training_job_id}: {str(e)}")
            raise ModelExportError(f"Model export failed: {str(e)}")
    
    def export_model_direct(self,
                          model: nn.Module,
                          model_info: Dict[str, Any],
                          age_group_min: float,
                          age_group_max: float,
                          export_format: str = "pytorch") -> ModelExportMetadata:
        """
        Export model directly without training job reference.
        
        Args:
            model: PyTorch model to export
            model_info: Model information dictionary
            age_group_min: Minimum age for the age group
            age_group_max: Maximum age for the age group
            export_format: Export format
            
        Returns:
            Model export metadata
        """
        try:
            logger.info("Exporting model directly")
            
            # Generate export metadata
            export_metadata = self._create_export_metadata(
                training_job_id=0,  # No training job
                model=model,
                model_info=model_info,
                age_group_min=age_group_min,
                age_group_max=age_group_max,
                export_format=export_format,
                training_report=None
            )
            
            # Export model
            export_path = self._export_model_file(model, export_metadata, export_format)
            
            # Update metadata
            export_metadata.file_size_bytes = export_path.stat().st_size
            export_metadata.checksum = self._calculate_file_checksum(export_path)
            
            # Save metadata
            metadata_path = export_path.parent / f"{export_metadata.model_id}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(export_metadata.to_dict(), f, indent=2)
            
            logger.info(f"Model exported directly: {export_path}")
            
            return export_metadata
            
        except Exception as e:
            logger.error(f"Failed to export model directly: {str(e)}")
            raise ModelExportError(f"Direct model export failed: {str(e)}")
    
    def _load_model_from_training_artifacts(self, 
                                          training_job_id: int,
                                          training_job: Optional[Any],
                                          training_report: Optional[Any]) -> Tuple[nn.Module, Dict]:
        """Load model from training artifacts."""
        # Look for model files in various locations
        possible_paths = [
            Path(f"outputs/job_{training_job_id}/model.pth"),
            Path(f"outputs/local_job_{training_job_id}/model.pth"),
            Path(f"outputs/sagemaker_job_{training_job_id}/model.pth"),
            Path(f"static/models/job_{training_job_id}_model.pth")
        ]
        
        if training_report and training_report.model_parameters_path:
            possible_paths.insert(0, Path(training_report.model_parameters_path))
        
        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = path
                break
        
        if not model_path:
            raise ModelExportError(f"Model file not found for training job {training_job_id}")
        
        # Load model
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Extract model information
            if isinstance(checkpoint, dict):
                model_state = checkpoint.get('model_state_dict', checkpoint)
                model_info = checkpoint.get('model_architecture', {})
                training_config = checkpoint.get('training_config', {})
            else:
                model_state = checkpoint
                model_info = {}
                training_config = {}
            
            # Reconstruct model architecture
            input_dim = model_info.get('input_dim', 512)  # Default
            hidden_dims = model_info.get('hidden_dims', [256, 128, 64])  # Default
            
            model = AutoencoderModel(input_dim, hidden_dims)
            model.load_state_dict(model_state)
            model.eval()
            
            # Combine model information
            combined_info = {
                **model_info,
                **training_config,
                'model_path': str(model_path),
                'parameter_count': sum(p.numel() for p in model.parameters())
            }
            
            return model, combined_info
            
        except Exception as e:
            raise ModelExportError(f"Failed to load model from {model_path}: {str(e)}")
    
    def _create_export_metadata(self,
                              training_job_id: int,
                              model: nn.Module,
                              model_info: Dict[str, Any],
                              age_group_min: float,
                              age_group_max: float,
                              export_format: str,
                              training_report: Optional[Any]) -> ModelExportMetadata:
        """Create export metadata."""
        # Generate unique model ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"autoencoder_{age_group_min}_{age_group_max}_{timestamp}"
        
        # Calculate architecture hash
        arch_string = f"{model_info.get('input_dim', 0)}_{model_info.get('hidden_dims', [])}_{model_info.get('latent_dim', 0)}"
        architecture_hash = hashlib.md5(arch_string.encode()).hexdigest()[:8]
        
        # Extract training metrics
        training_metrics = {}
        if training_report:
            try:
                training_metrics = json.loads(training_report.metrics_summary)
            except (json.JSONDecodeError, AttributeError):
                training_metrics = {
                    'final_loss': getattr(training_report, 'final_loss', 0.0),
                    'validation_accuracy': getattr(training_report, 'validation_accuracy', 0.0),
                    'best_epoch': getattr(training_report, 'best_epoch', 0),
                    'training_time_seconds': getattr(training_report, 'training_time_seconds', 0.0)
                }
        
        return ModelExportMetadata(
            model_id=model_id,
            export_timestamp=datetime.now(),
            training_job_id=training_job_id,
            model_type="autoencoder",
            model_version="1.0",
            architecture_hash=architecture_hash,
            parameter_count=sum(p.numel() for p in model.parameters()),
            input_dimension=model_info.get('input_dim', 0),
            output_dimension=model_info.get('input_dim', 0),  # Autoencoder: input = output
            age_group_min=age_group_min,
            age_group_max=age_group_max,
            training_metrics=training_metrics,
            compatibility_version="1.0",
            export_format=export_format,
            file_size_bytes=0,  # Will be updated after export
            checksum=""  # Will be updated after export
        )
    
    def _export_model_file(self, 
                         model: nn.Module, 
                         metadata: ModelExportMetadata, 
                         export_format: str) -> Path:
        """Export model file in specified format."""
        export_path = self.export_dir / f"{metadata.model_id}.{export_format}"
        
        if export_format == "pytorch":
            # Export as PyTorch state dict
            import torch
            export_data = {
                'model_state_dict': model.state_dict(),
                'model_architecture': {
                    'input_dim': metadata.input_dimension,
                    'hidden_dims': getattr(model, 'hidden_dims', []),
                    'model_type': metadata.model_type
                },
                'metadata': metadata.to_dict(),
                'pytorch_version': torch.__version__
            }
            torch.save(export_data, export_path)
            
        elif export_format == "pickle":
            # Export as pickle (includes full model)
            export_data = {
                'model': model,
                'metadata': metadata.to_dict()
            }
            with open(export_path, 'wb') as f:
                pickle.dump(export_data, f)
                
        elif export_format == "onnx":
            # Export as ONNX (requires dummy input)
            try:
                import torch.onnx
                dummy_input = torch.randn(1, metadata.input_dimension)
                torch.onnx.export(
                    model,
                    dummy_input,
                    export_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output']
                )
            except ImportError:
                raise ModelExportError("ONNX export requires torch.onnx")
        
        return export_path
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def list_exported_models(self) -> List[Dict[str, Any]]:
        """List all exported models."""
        exported_models = []
        
        for metadata_file in self.export_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                exported_models.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to read metadata file {metadata_file}: {str(e)}")
        
        # Sort by export timestamp (newest first)
        exported_models.sort(key=lambda x: x.get('export_timestamp', ''), reverse=True)
        
        return exported_models


class ModelValidator:
    """Service for validating model compatibility and integrity."""
    
    def __init__(self):
        self.compatibility_version = "1.0"
        logger.info("Model Validator initialized")
    
    def validate_exported_model(self, export_metadata: ModelExportMetadata) -> Dict[str, Any]:
        """
        Validate exported model for compatibility and integrity.
        
        Args:
            export_metadata: Model export metadata
            
        Returns:
            Validation results dictionary
            
        Raises:
            ModelValidationError: If validation fails
        """
        try:
            logger.info(f"Validating exported model: {export_metadata.model_id}")
            
            validation_result = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'compatibility_checks': {},
                'integrity_checks': {},
                'performance_checks': {}
            }
            
            # Check file existence and integrity
            model_file = Path(f"exports/models/{export_metadata.model_id}.{export_metadata.export_format}")
            if not model_file.exists():
                validation_result['errors'].append(f"Model file not found: {model_file}")
                validation_result['is_valid'] = False
            else:
                # Verify checksum
                actual_checksum = self._calculate_checksum(model_file)
                if actual_checksum != export_metadata.checksum:
                    validation_result['errors'].append("Model file checksum mismatch")
                    validation_result['is_valid'] = False
                else:
                    validation_result['integrity_checks']['checksum'] = 'passed'
            
            # Compatibility checks
            compatibility_result = self._check_compatibility(export_metadata)
            validation_result['compatibility_checks'] = compatibility_result
            
            if not compatibility_result.get('is_compatible', True):
                validation_result['errors'].extend(compatibility_result.get('errors', []))
                validation_result['is_valid'] = False
            
            # Model structure validation
            if validation_result['is_valid']:
                structure_result = self._validate_model_structure(export_metadata, model_file)
                validation_result['integrity_checks'].update(structure_result)
                
                if not structure_result.get('structure_valid', True):
                    validation_result['errors'].extend(structure_result.get('errors', []))
                    validation_result['is_valid'] = False
            
            # Performance validation
            if validation_result['is_valid']:
                performance_result = self._validate_model_performance(export_metadata)
                validation_result['performance_checks'] = performance_result
                
                # Performance issues are warnings, not errors
                validation_result['warnings'].extend(performance_result.get('warnings', []))
            
            logger.info(f"Model validation completed: {'PASSED' if validation_result['is_valid'] else 'FAILED'}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            raise ModelValidationError(f"Validation failed: {str(e)}")
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _check_compatibility(self, metadata: ModelExportMetadata) -> Dict[str, Any]:
        """Check model compatibility."""
        result = {
            'is_compatible': True,
            'errors': [],
            'warnings': [],
            'version_check': 'passed',
            'architecture_check': 'passed',
            'format_check': 'passed'
        }
        
        # Check compatibility version
        if metadata.compatibility_version != self.compatibility_version:
            result['warnings'].append(f"Compatibility version mismatch: {metadata.compatibility_version} vs {self.compatibility_version}")
        
        # Check model type
        if metadata.model_type != "autoencoder":
            result['errors'].append(f"Unsupported model type: {metadata.model_type}")
            result['is_compatible'] = False
        
        # Check dimensions
        if metadata.input_dimension <= 0:
            result['errors'].append("Invalid input dimension")
            result['is_compatible'] = False
        
        if metadata.parameter_count <= 0:
            result['errors'].append("Invalid parameter count")
            result['is_compatible'] = False
        
        # Check age group validity
        if metadata.age_group_min >= metadata.age_group_max:
            result['errors'].append("Invalid age group range")
            result['is_compatible'] = False
        
        if metadata.age_group_min < 2.0 or metadata.age_group_max > 18.0:
            result['warnings'].append("Age group outside typical range (2-18 years)")
        
        return result
    
    def _validate_model_structure(self, metadata: ModelExportMetadata, model_file: Path) -> Dict[str, Any]:
        """Validate model structure."""
        result = {
            'structure_valid': True,
            'errors': [],
            'warnings': [],
            'loadable': False,
            'parameter_count_match': False
        }
        
        try:
            # Try to load the model
            if metadata.export_format == "pytorch":
                checkpoint = torch.load(model_file, map_location='cpu')
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    result['loadable'] = True
                    
                    # Count parameters
                    param_count = sum(p.numel() for p in state_dict.values())
                    if param_count == metadata.parameter_count:
                        result['parameter_count_match'] = True
                    else:
                        result['warnings'].append(f"Parameter count mismatch: {param_count} vs {metadata.parameter_count}")
                else:
                    result['errors'].append("Invalid PyTorch checkpoint format")
                    result['structure_valid'] = False
                    
            elif metadata.export_format == "pickle":
                with open(model_file, 'rb') as f:
                    data = pickle.load(f)
                    if 'model' in data:
                        result['loadable'] = True
                    else:
                        result['errors'].append("Invalid pickle format")
                        result['structure_valid'] = False
                        
            elif metadata.export_format == "onnx":
                try:
                    import onnx
                    model = onnx.load(str(model_file))
                    onnx.checker.check_model(model)
                    result['loadable'] = True
                except ImportError:
                    result['warnings'].append("ONNX validation skipped (onnx not installed)")
                except Exception as e:
                    result['errors'].append(f"ONNX validation failed: {str(e)}")
                    result['structure_valid'] = False
            
        except Exception as e:
            result['errors'].append(f"Failed to load model: {str(e)}")
            result['structure_valid'] = False
        
        return result
    
    def _validate_model_performance(self, metadata: ModelExportMetadata) -> Dict[str, Any]:
        """Validate model performance metrics."""
        result = {
            'performance_acceptable': True,
            'warnings': [],
            'metrics_available': False,
            'loss_reasonable': False,
            'training_time_reasonable': False
        }
        
        metrics = metadata.training_metrics
        if metrics:
            result['metrics_available'] = True
            
            # Check final loss
            final_loss = metrics.get('final_loss', float('inf'))
            if final_loss < 1.0:  # Reasonable loss threshold
                result['loss_reasonable'] = True
            else:
                result['warnings'].append(f"High final loss: {final_loss}")
            
            # Check training time
            training_time = metrics.get('training_time_seconds', 0)
            if 0 < training_time < 86400:  # Between 0 and 24 hours
                result['training_time_reasonable'] = True
            elif training_time >= 86400:
                result['warnings'].append(f"Very long training time: {training_time/3600:.1f} hours")
            
            # Check validation accuracy
            val_accuracy = metrics.get('validation_accuracy', 0)
            if val_accuracy < 0.5:
                result['warnings'].append(f"Low validation accuracy: {val_accuracy:.3f}")
        else:
            result['warnings'].append("No training metrics available")
        
        return result


class ModelDeploymentService:
    """Service for deploying models to production environment."""
    
    def __init__(self):
        self.settings = get_settings()
        self.deployment_dir = Path("static/models")
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir = Path("backups/models")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.exporter = ModelExporter()
        self.validator = ModelValidator()
        
        logger.info("Model Deployment Service initialized")
    
    def deploy_model(self, 
                    deployment_config: ModelDeploymentConfig,
                    db: Session) -> Dict[str, Any]:
        """
        Deploy model to production environment.
        
        Args:
            deployment_config: Deployment configuration
            db: Database session
            
        Returns:
            Deployment result dictionary
            
        Raises:
            ModelDeploymentError: If deployment fails
        """
        try:
            logger.info(f"Deploying model from {deployment_config.model_export_path}")
            
            deployment_result = {
                'success': False,
                'model_id': None,
                'deployment_path': None,
                'backup_path': None,
                'validation_result': None,
                'database_updated': False,
                'warnings': []
            }
            
            # Load export metadata
            export_path = Path(deployment_config.model_export_path)
            if not export_path.exists():
                raise ModelDeploymentError(f"Export file not found: {export_path}")
            
            metadata_path = export_path.parent / f"{export_path.stem}_metadata.json"
            if not metadata_path.exists():
                raise ModelDeploymentError(f"Metadata file not found: {metadata_path}")
            
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            # Reconstruct metadata object
            metadata_dict['export_timestamp'] = datetime.fromisoformat(metadata_dict['export_timestamp'])
            export_metadata = ModelExportMetadata(**metadata_dict)
            
            # Validate model if requested
            if deployment_config.validate_before_deployment:
                validation_result = self.validator.validate_exported_model(export_metadata)
                deployment_result['validation_result'] = validation_result
                
                if not validation_result['is_valid']:
                    raise ModelDeploymentError(f"Model validation failed: {validation_result['errors']}")
                
                deployment_result['warnings'].extend(validation_result.get('warnings', []))
            
            # Check for existing model in same age group
            existing_model = None
            if db:
                existing_model = db.query(AgeGroupModel).filter(
                    AgeGroupModel.age_min == deployment_config.age_group_min,
                    AgeGroupModel.age_max == deployment_config.age_group_max,
                    AgeGroupModel.is_active == True
                ).first()
            
            if existing_model and not deployment_config.replace_existing:
                raise ModelDeploymentError(f"Model already exists for age group {deployment_config.age_group_min}-{deployment_config.age_group_max}")
            
            # Backup existing model if requested
            backup_path = None
            if existing_model and deployment_config.backup_existing:
                backup_path = self._backup_existing_model(existing_model)
                deployment_result['backup_path'] = str(backup_path)
            
            # Deploy model file
            deployment_path = self._deploy_model_file(export_path, export_metadata)
            deployment_result['deployment_path'] = str(deployment_path)
            deployment_result['model_id'] = export_metadata.model_id
            
            # Update database
            if db:
                self._update_database_record(
                    export_metadata, 
                    deployment_config, 
                    deployment_path, 
                    existing_model,
                    db
                )
                deployment_result['database_updated'] = True
            
            deployment_result['success'] = True
            logger.info(f"Model deployed successfully: {deployment_path}")
            
            return deployment_result
            
        except Exception as e:
            logger.error(f"Model deployment failed: {str(e)}")
            raise ModelDeploymentError(f"Deployment failed: {str(e)}")
    
    def _backup_existing_model(self, existing_model: Any) -> Path:
        """Backup existing model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"model_{existing_model.id}_{timestamp}.backup"
        backup_path = self.backup_dir / backup_filename
        
        # Find and backup the model file
        model_files = list(self.deployment_dir.glob(f"*{existing_model.id}*"))
        if model_files:
            shutil.copy2(model_files[0], backup_path)
            logger.info(f"Backed up existing model to: {backup_path}")
        
        return backup_path
    
    def _deploy_model_file(self, export_path: Path, metadata: ModelExportMetadata) -> Path:
        """Deploy model file to production directory."""
        deployment_filename = f"{metadata.model_id}.{metadata.export_format}"
        deployment_path = self.deployment_dir / deployment_filename
        
        # Copy model file
        shutil.copy2(export_path, deployment_path)
        
        # Copy metadata file
        metadata_src = export_path.parent / f"{export_path.stem}_metadata.json"
        metadata_dst = self.deployment_dir / f"{metadata.model_id}_metadata.json"
        if metadata_src.exists():
            shutil.copy2(metadata_src, metadata_dst)
        
        return deployment_path
    
    def _update_database_record(self,
                              metadata: ModelExportMetadata,
                              config: ModelDeploymentConfig,
                              deployment_path: Path,
                              existing_model: Optional[Any],
                              db: Session):
        """Update database with deployment information."""
        try:
            # Deactivate existing model if replacing
            if existing_model and config.replace_existing:
                existing_model.is_active = False
                db.commit()
            
            # Create new age group model record
            model_parameters = {
                'model_id': metadata.model_id,
                'export_metadata': metadata.to_dict(),
                'deployment_path': str(deployment_path),
                'deployment_timestamp': datetime.now().isoformat()
            }
            
            new_model = AgeGroupModel(
                age_min=config.age_group_min,
                age_max=config.age_group_max,
                model_type=metadata.model_type,
                vision_model="vit",  # Assuming ViT is used for embeddings
                parameters=json.dumps(model_parameters),
                sample_count=0,  # Would be updated based on training data
                threshold=0.95,  # Default threshold, would be calculated from validation data
                is_active=True
            )
            
            db.add(new_model)
            db.commit()
            
            logger.info(f"Database updated with new model record: {new_model.id}")
            
        except Exception as e:
            logger.error(f"Failed to update database: {str(e)}")
            db.rollback()
            raise
    
    def list_deployed_models(self, db: Session) -> List[Dict[str, Any]]:
        """List all deployed models."""
        deployed_models = []
        
        if db:
            models = db.query(AgeGroupModel).filter(AgeGroupModel.is_active == True).all()
            
            for model in models:
                try:
                    parameters = json.loads(model.parameters)
                    deployed_models.append({
                        'id': model.id,
                        'age_min': model.age_min,
                        'age_max': model.age_max,
                        'model_type': model.model_type,
                        'vision_model': model.vision_model,
                        'threshold': model.threshold,
                        'created_timestamp': model.created_timestamp.isoformat(),
                        'deployment_info': parameters
                    })
                except Exception as e:
                    logger.warning(f"Failed to parse model parameters for model {model.id}: {str(e)}")
        
        return deployed_models
    
    def undeploy_model(self, model_id: int, db: Session) -> bool:
        """Undeploy (deactivate) a model."""
        try:
            model = db.query(AgeGroupModel).filter(AgeGroupModel.id == model_id).first()
            if not model:
                return False
            
            model.is_active = False
            db.commit()
            
            logger.info(f"Model {model_id} undeployed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to undeploy model {model_id}: {str(e)}")
            db.rollback()
            return False


# Global service instances
_model_exporter = None
_model_validator = None
_model_deployment_service = None


def get_model_exporter() -> ModelExporter:
    """Get the global model exporter instance."""
    global _model_exporter
    if _model_exporter is None:
        _model_exporter = ModelExporter()
    return _model_exporter


def get_model_validator() -> ModelValidator:
    """Get the global model validator instance."""
    global _model_validator
    if _model_validator is None:
        _model_validator = ModelValidator()
    return _model_validator


def get_model_deployment_service() -> ModelDeploymentService:
    """Get the global model deployment service instance."""
    global _model_deployment_service
    if _model_deployment_service is None:
        _model_deployment_service = ModelDeploymentService()
    return _model_deployment_service