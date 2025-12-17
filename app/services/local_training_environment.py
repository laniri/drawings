"""
Local Training Environment Service for Children's Drawing Anomaly Detection System

This module provides local autoencoder training with PyTorch, GPU/CPU device detection,
and comprehensive training progress monitoring and logging.
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import queue

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.database import TrainingJob, TrainingReport
from app.services.training_config import TrainingConfig, TrainingConfigManager
from app.services.dataset_preparation import DatasetPreparationService, SplitConfig
from app.services.embedding_service import get_embedding_service
from app.services.model_manager import AutoencoderModel, AutoencoderTrainer
from app.core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class LocalTrainingError(Exception):
    """Base exception for local training errors."""
    pass


class DeviceDetectionError(LocalTrainingError):
    """Raised when device detection fails."""
    pass


class TrainingProgressError(LocalTrainingError):
    """Raised when training progress monitoring fails."""
    pass


@dataclass
class TrainingProgress:
    """Container for training progress information."""
    job_id: int
    epoch: int
    total_epochs: int
    batch: int
    total_batches: int
    train_loss: float
    val_loss: Optional[float] = None
    learning_rate: float = 0.0
    elapsed_time: float = 0.0
    estimated_remaining: Optional[float] = None
    memory_usage: Optional[Dict] = None
    
    @property
    def epoch_progress(self) -> float:
        """Get epoch progress as percentage."""
        return (self.epoch / self.total_epochs) * 100 if self.total_epochs > 0 else 0.0
    
    @property
    def batch_progress(self) -> float:
        """Get batch progress within current epoch as percentage."""
        return (self.batch / self.total_batches) * 100 if self.total_batches > 0 else 0.0


class DeviceManager:
    """Enhanced device manager with detailed GPU/CPU detection and monitoring."""
    
    def __init__(self):
        self._device = None
        self._device_info = None
        self._memory_monitor = None
        self._detect_device()
    
    def _detect_device(self) -> None:
        """Detect and configure the best available device with detailed information."""
        try:
            if torch.cuda.is_available():
                self._setup_cuda_device()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._setup_mps_device()
            else:
                self._setup_cpu_device()
            
            logger.info(f"Device detection complete: {self._device_info}")
            
        except Exception as e:
            logger.error(f"Device detection failed: {str(e)}")
            raise DeviceDetectionError(f"Failed to detect device: {str(e)}")
    
    def _setup_cuda_device(self) -> None:
        """Setup CUDA device with detailed information."""
        device_id = 0  # Use first GPU
        self._device = torch.device(f"cuda:{device_id}")
        
        props = torch.cuda.get_device_properties(device_id)
        self._device_info = {
            "type": "cuda",
            "device_id": device_id,
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory": props.total_memory,
            "total_memory_gb": props.total_memory / (1024**3),
            "multiprocessor_count": props.multi_processor_count,
            "device_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
            "supports_mixed_precision": True
        }
        
        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.8, device_id)
        
        logger.info(f"CUDA device configured: {props.name} with {self._device_info['total_memory_gb']:.1f}GB memory")
    
    def _setup_mps_device(self) -> None:
        """Setup Apple Metal Performance Shaders device."""
        self._device = torch.device("mps")
        self._device_info = {
            "type": "mps",
            "name": "Apple Metal Performance Shaders",
            "device_id": 0,
            "total_memory": None,  # MPS doesn't expose memory info
            "total_memory_gb": None,
            "device_count": 1,
            "supports_mixed_precision": False  # MPS has limited mixed precision support
        }
        
        logger.info("MPS (Apple Silicon) device configured")
    
    def _setup_cpu_device(self) -> None:
        """Setup CPU device with thread information."""
        self._device = torch.device("cpu")
        
        # Get CPU information
        import psutil
        cpu_count = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        memory_info = psutil.virtual_memory()
        
        self._device_info = {
            "type": "cpu",
            "name": "CPU",
            "device_id": 0,
            "physical_cores": cpu_count,
            "logical_cores": cpu_count_logical,
            "total_memory": memory_info.total,
            "total_memory_gb": memory_info.total / (1024**3),
            "available_memory_gb": memory_info.available / (1024**3),
            "device_count": 1,
            "supports_mixed_precision": False
        }
        
        # Set optimal number of threads for PyTorch
        torch.set_num_threads(min(cpu_count_logical, 8))  # Cap at 8 threads
        
        logger.info(f"CPU device configured: {cpu_count}/{cpu_count_logical} cores, "
                   f"{self._device_info['total_memory_gb']:.1f}GB memory")
    
    @property
    def device(self) -> torch.device:
        """Get the current device."""
        return self._device
    
    @property
    def device_info(self) -> Dict:
        """Get device information."""
        return self._device_info.copy()
    
    def get_memory_usage(self) -> Optional[Dict]:
        """Get current memory usage if available."""
        try:
            if self._device.type == "cuda":
                return {
                    "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                    "cached_mb": torch.cuda.memory_reserved() / (1024**2),
                    "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
                    "total_mb": self._device_info["total_memory"] / (1024**2)
                }
            elif self._device.type == "cpu":
                import psutil
                memory = psutil.virtual_memory()
                return {
                    "used_mb": (memory.total - memory.available) / (1024**2),
                    "available_mb": memory.available / (1024**2),
                    "total_mb": memory.total / (1024**2),
                    "percent": memory.percent
                }
            return None
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {str(e)}")
            return None
    
    def clear_cache(self) -> None:
        """Clear device cache if applicable."""
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
    
    def optimize_for_training(self) -> None:
        """Apply device-specific optimizations for training."""
        if self._device.type == "cuda":
            # Enable cuDNN benchmark for consistent input sizes
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            logger.info("CUDA optimizations enabled")
        elif self._device.type == "cpu":
            # Enable CPU optimizations
            torch.set_num_threads(self._device_info["logical_cores"])
            logger.info("CPU optimizations enabled")


class TrainingProgressMonitor:
    """Monitor and log training progress with real-time updates."""
    
    def __init__(self, job_id: int, total_epochs: int):
        self.job_id = job_id
        self.total_epochs = total_epochs
        self.start_time = time.time()
        self.epoch_start_time = None
        self.progress_queue = queue.Queue()
        self.callbacks = []
        self.history = []
        self._stop_monitoring = False
        
    def add_callback(self, callback: Callable[[TrainingProgress], None]) -> None:
        """Add a callback function to be called on progress updates."""
        self.callbacks.append(callback)
    
    def start_epoch(self, epoch: int, total_batches: int) -> None:
        """Mark the start of a new epoch."""
        self.epoch_start_time = time.time()
        self.current_epoch = epoch
        self.total_batches = total_batches
        logger.info(f"Starting epoch {epoch}/{self.total_epochs} with {total_batches} batches")
    
    def update_batch(self, batch: int, train_loss: float, learning_rate: float = 0.0) -> None:
        """Update progress for current batch."""
        if self.epoch_start_time is None:
            return
        
        elapsed_time = time.time() - self.start_time
        epoch_elapsed = time.time() - self.epoch_start_time
        
        # Estimate remaining time
        if batch > 0:
            batch_time = epoch_elapsed / batch
            remaining_batches = self.total_batches - batch
            epoch_remaining = remaining_batches * batch_time
            
            # Estimate total remaining time
            epochs_remaining = self.total_epochs - self.current_epoch
            if epochs_remaining > 0:
                avg_epoch_time = elapsed_time / self.current_epoch if self.current_epoch > 0 else epoch_elapsed
                total_remaining = epoch_remaining + (epochs_remaining * avg_epoch_time)
            else:
                total_remaining = epoch_remaining
        else:
            total_remaining = None
        
        progress = TrainingProgress(
            job_id=self.job_id,
            epoch=self.current_epoch,
            total_epochs=self.total_epochs,
            batch=batch,
            total_batches=self.total_batches,
            train_loss=train_loss,
            learning_rate=learning_rate,
            elapsed_time=elapsed_time,
            estimated_remaining=total_remaining
        )
        
        # Add memory usage if available
        device_manager = DeviceManager()
        progress.memory_usage = device_manager.get_memory_usage()
        
        # Store in queue and call callbacks
        self.progress_queue.put(progress)
        for callback in self.callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {str(e)}")
    
    def update_epoch(self, epoch: int, train_loss: float, val_loss: float) -> None:
        """Update progress at end of epoch."""
        elapsed_time = time.time() - self.start_time
        
        progress = TrainingProgress(
            job_id=self.job_id,
            epoch=epoch,
            total_epochs=self.total_epochs,
            batch=self.total_batches,
            total_batches=self.total_batches,
            train_loss=train_loss,
            val_loss=val_loss,
            elapsed_time=elapsed_time
        )
        
        self.history.append(progress)
        
        # Log epoch completion
        logger.info(f"Epoch {epoch}/{self.total_epochs} completed: "
                   f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                   f"Time: {elapsed_time:.1f}s")
        
        # Call callbacks
        for callback in self.callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.warning(f"Epoch callback failed: {str(e)}")
    
    def get_latest_progress(self) -> Optional[TrainingProgress]:
        """Get the latest progress update."""
        try:
            return self.progress_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_history(self) -> List[TrainingProgress]:
        """Get complete training history."""
        return self.history.copy()
    
    def stop(self) -> None:
        """Stop monitoring."""
        self._stop_monitoring = True


class LocalTrainingEnvironment:
    """Local training environment with comprehensive monitoring and logging."""
    
    def __init__(self):
        self.device_manager = DeviceManager()
        self.dataset_service = DatasetPreparationService()
        self.embedding_service = get_embedding_service()
        self.config_manager = TrainingConfigManager()
        self.active_jobs = {}  # job_id -> monitor
        self.training_history = {}  # job_id -> history
        
        # Ensure embedding service is initialized
        if not self.embedding_service.is_ready():
            self.embedding_service.initialize()
        
        logger.info("Local Training Environment initialized")
    
    def get_environment_info(self) -> Dict:
        """Get comprehensive environment information."""
        return {
            "device_info": self.device_manager.device_info,
            "memory_usage": self.device_manager.get_memory_usage(),
            "embedding_service": self.embedding_service.get_service_info(),
            "active_jobs": list(self.active_jobs.keys()),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        }
    
    def prepare_training_data(self, 
                            dataset_folder: str, 
                            metadata_file: str,
                            config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data from dataset folder.
        
        Args:
            dataset_folder: Path to folder containing images
            metadata_file: Path to metadata file
            config: Training configuration
            
        Returns:
            Tuple of (train_embeddings, val_embeddings, test_embeddings)
        """
        try:
            logger.info(f"Preparing training data from {dataset_folder}")
            
            # Create split configuration
            split_config = SplitConfig(
                train_ratio=config.data.train_split,
                validation_ratio=config.data.validation_split,
                test_ratio=config.data.test_split,
                stratify_by_age=config.data.stratify_by_age,
                age_group_size=config.data.age_group_size
            )
            
            # Load and split dataset
            dataset_split = self.dataset_service.prepare_dataset(
                dataset_folder, metadata_file, split_config
            )
            
            logger.info(f"Dataset split: train={dataset_split.train_count}, "
                       f"val={dataset_split.validation_count}, test={dataset_split.test_count}")
            
            # Generate embeddings for each split
            train_embeddings = self._generate_embeddings_for_split(
                dataset_split.train_files, dataset_split.train_metadata
            )
            
            val_embeddings = self._generate_embeddings_for_split(
                dataset_split.validation_files, dataset_split.validation_metadata
            )
            
            test_embeddings = self._generate_embeddings_for_split(
                dataset_split.test_files, dataset_split.test_metadata
            ) if dataset_split.test_count > 0 else np.array([])
            
            logger.info(f"Generated embeddings: train={len(train_embeddings)}, "
                       f"val={len(val_embeddings)}, test={len(test_embeddings)}")
            
            return train_embeddings, val_embeddings, test_embeddings
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {str(e)}")
            raise LocalTrainingError(f"Data preparation failed: {str(e)}")
    
    def _generate_embeddings_for_split(self, files: List[Path], metadata: List) -> np.ndarray:
        """Generate embeddings for a dataset split."""
        embeddings = []
        
        for file_path, meta in zip(files, metadata):
            try:
                # Load image
                image = Image.open(file_path).convert('RGB')
                
                # Generate embedding with age information
                embedding = self.embedding_service.generate_embedding(
                    image=image,
                    age=meta.age_years,
                    use_cache=True
                )
                
                embeddings.append(embedding)
                
            except Exception as e:
                logger.warning(f"Failed to generate embedding for {file_path}: {str(e)}")
                continue
        
        if not embeddings:
            raise LocalTrainingError("No valid embeddings generated")
        
        return np.array(embeddings)
    
    def start_training_job(self, 
                          config: TrainingConfig,
                          train_embeddings: np.ndarray,
                          val_embeddings: np.ndarray,
                          db: Session) -> int:
        """
        Start a local training job.
        
        Args:
            config: Training configuration
            train_embeddings: Training embeddings
            val_embeddings: Validation embeddings
            db: Database session
            
        Returns:
            Training job ID
        """
        try:
            # Create training job record
            training_job = TrainingJob(
                job_name=config.job_name,
                environment="local",
                config_parameters=json.dumps(asdict(config)),
                dataset_path=config.dataset_folder,
                status="pending",
                start_timestamp=datetime.utcnow()
            )
            
            db.add(training_job)
            db.commit()
            db.refresh(training_job)
            
            logger.info(f"Created training job {training_job.id}: {config.job_name}")
            
            # Start training in background thread
            training_thread = threading.Thread(
                target=self._run_training_job,
                args=(training_job.id, config, train_embeddings, val_embeddings, db),
                daemon=True
            )
            training_thread.start()
            
            return training_job.id
            
        except Exception as e:
            logger.error(f"Failed to start training job: {str(e)}")
            raise LocalTrainingError(f"Job creation failed: {str(e)}")
    
    def _run_training_job(self, 
                         job_id: int,
                         config: TrainingConfig,
                         train_embeddings: np.ndarray,
                         val_embeddings: np.ndarray,
                         db: Session) -> None:
        """Run training job in background thread."""
        try:
            # Update job status
            training_job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            training_job.status = "running"
            db.commit()
            
            logger.info(f"Starting training for job {job_id}")
            
            # Setup progress monitoring
            monitor = TrainingProgressMonitor(job_id, config.epochs)
            self.active_jobs[job_id] = monitor
            
            # Add logging callback
            monitor.add_callback(self._log_progress)
            
            # Optimize device for training
            self.device_manager.optimize_for_training()
            
            # Create enhanced trainer with monitoring
            trainer = EnhancedAutoencoderTrainer(config, self.device_manager, monitor)
            
            # Train model
            training_result = trainer.train(train_embeddings, val_embeddings)
            
            # Save training results
            self._save_training_results(job_id, training_result, config, db)
            
            # Update job status
            training_job.status = "completed"
            training_job.end_timestamp = datetime.utcnow()
            db.commit()
            
            logger.info(f"Training job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Training job {job_id} failed: {str(e)}")
            
            # Update job status to failed
            try:
                training_job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
                if training_job:
                    training_job.status = "failed"
                    training_job.end_timestamp = datetime.utcnow()
                    db.commit()
            except Exception as db_error:
                logger.error(f"Failed to update job status: {str(db_error)}")
        
        finally:
            # Clean up
            if job_id in self.active_jobs:
                self.active_jobs[job_id].stop()
                del self.active_jobs[job_id]
            
            # Clear device cache
            self.device_manager.clear_cache()
    
    def _log_progress(self, progress: TrainingProgress) -> None:
        """Log training progress."""
        if progress.batch % 10 == 0:  # Log every 10 batches
            memory_str = ""
            if progress.memory_usage:
                if "allocated_mb" in progress.memory_usage:
                    memory_str = f", GPU Memory: {progress.memory_usage['allocated_mb']:.0f}MB"
                elif "used_mb" in progress.memory_usage:
                    memory_str = f", Memory: {progress.memory_usage['percent']:.1f}%"
            
            logger.info(f"Job {progress.job_id} - Epoch {progress.epoch}/{progress.total_epochs}, "
                       f"Batch {progress.batch}/{progress.total_batches}, "
                       f"Loss: {progress.train_loss:.6f}{memory_str}")
    
    def _save_training_results(self, 
                              job_id: int,
                              training_result: Dict,
                              config: TrainingConfig,
                              db: Session) -> None:
        """Save training results to database and files."""
        try:
            # Create output directory
            output_dir = Path(config.output_dir) / f"job_{job_id}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model parameters
            model_path = output_dir / "model.pth"
            # Note: Model saving would be handled by the trainer
            
            # Generate training report
            report_path = output_dir / "training_report.json"
            with open(report_path, 'w') as f:
                json.dump(training_result, f, indent=2, default=str)
            
            # Generate plots if requested
            if config.save_plots:
                self._generate_training_plots(training_result, output_dir)
            
            # Create database record
            training_report = TrainingReport(
                training_job_id=job_id,
                final_loss=training_result["best_val_loss"],
                validation_accuracy=1.0 - training_result["best_val_loss"],  # Approximate
                best_epoch=training_result["epochs_trained"],
                training_time_seconds=training_result.get("training_time", 0),
                model_parameters_path=str(model_path),
                metrics_summary=json.dumps(training_result["metrics"]),
                report_file_path=str(report_path)
            )
            
            db.add(training_report)
            db.commit()
            
            logger.info(f"Training results saved for job {job_id}")
            
        except Exception as e:
            logger.error(f"Failed to save training results: {str(e)}")
            raise LocalTrainingError(f"Results saving failed: {str(e)}")
    
    def _generate_training_plots(self, training_result: Dict, output_dir: Path) -> None:
        """Generate training visualization plots."""
        try:
            history = training_result.get("training_history", [])
            if not history:
                return
            
            # Extract data for plotting
            epochs = [h["epoch"] for h in history]
            train_losses = [h["train_loss"] for h in history]
            val_losses = [h["val_loss"] for h in history]
            
            # Create loss plot
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_losses, label='Training Loss', color='blue')
            plt.plot(epochs, val_losses, label='Validation Loss', color='red')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / "loss_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create metrics plot if available
            metrics = training_result.get("metrics", {})
            if metrics:
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                
                # Reconstruction error distribution
                if "mean_reconstruction_error" in metrics:
                    axes[0, 0].bar(['Mean', 'Std', 'Min', 'Max'], [
                        metrics["mean_reconstruction_error"],
                        metrics["std_reconstruction_error"],
                        metrics["min_reconstruction_error"],
                        metrics["max_reconstruction_error"]
                    ])
                    axes[0, 0].set_title('Reconstruction Error Statistics')
                    axes[0, 0].set_ylabel('Error')
                
                # Percentiles
                if "percentile_95" in metrics:
                    percentiles = [90, 95, 99]
                    values = [
                        metrics.get("percentile_90", 0),
                        metrics.get("percentile_95", 0),
                        metrics.get("percentile_99", 0)
                    ]
                    axes[0, 1].bar([f'{p}th' for p in percentiles], values)
                    axes[0, 1].set_title('Error Percentiles')
                    axes[0, 1].set_ylabel('Error')
                
                # Training progress
                axes[1, 0].plot(epochs, train_losses, label='Train')
                axes[1, 0].plot(epochs, val_losses, label='Validation')
                axes[1, 0].set_title('Loss Progress')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].legend()
                
                # Model architecture info
                arch_info = training_result.get("model_architecture", {})
                if arch_info:
                    info_text = f"Input Dim: {arch_info.get('input_dim', 'N/A')}\n"
                    info_text += f"Hidden Dims: {arch_info.get('hidden_dims', 'N/A')}\n"
                    info_text += f"Parameters: {arch_info.get('total_parameters', 'N/A')}"
                    axes[1, 1].text(0.1, 0.5, info_text, transform=axes[1, 1].transAxes,
                                   fontsize=10, verticalalignment='center')
                    axes[1, 1].set_title('Model Architecture')
                    axes[1, 1].axis('off')
                
                plt.tight_layout()
                plt.savefig(output_dir / "training_metrics.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info(f"Training plots saved to {output_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to generate training plots: {str(e)}")
    
    def get_job_status(self, job_id: int, db: Session) -> Dict:
        """Get status of a training job."""
        training_job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        
        if not training_job:
            raise LocalTrainingError(f"Training job {job_id} not found")
        
        status = {
            "job_id": job_id,
            "job_name": training_job.job_name,
            "status": training_job.status,
            "start_timestamp": training_job.start_timestamp,
            "end_timestamp": training_job.end_timestamp,
            "environment": training_job.environment
        }
        
        # Add progress information if job is active
        if job_id in self.active_jobs:
            monitor = self.active_jobs[job_id]
            latest_progress = monitor.get_latest_progress()
            if latest_progress:
                status["progress"] = {
                    "epoch": latest_progress.epoch,
                    "total_epochs": latest_progress.total_epochs,
                    "epoch_progress": latest_progress.epoch_progress,
                    "batch_progress": latest_progress.batch_progress,
                    "train_loss": latest_progress.train_loss,
                    "val_loss": latest_progress.val_loss,
                    "elapsed_time": latest_progress.elapsed_time,
                    "estimated_remaining": latest_progress.estimated_remaining
                }
        
        return status
    
    def list_training_jobs(self, db: Session) -> List[Dict]:
        """List all training jobs."""
        jobs = db.query(TrainingJob).filter(
            TrainingJob.environment == "local"
        ).order_by(TrainingJob.start_timestamp.desc()).all()
        
        return [self.get_job_status(job.id, db) for job in jobs]
    
    def cancel_training_job(self, job_id: int, db: Session) -> bool:
        """Cancel an active training job."""
        if job_id in self.active_jobs:
            monitor = self.active_jobs[job_id]
            monitor.stop()
            
            # Update database status
            training_job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if training_job:
                training_job.status = "cancelled"
                training_job.end_timestamp = datetime.utcnow()
                db.commit()
            
            logger.info(f"Training job {job_id} cancelled")
            return True
        
        return False


class EnhancedAutoencoderTrainer(AutoencoderTrainer):
    """Enhanced autoencoder trainer with progress monitoring and device management."""
    
    def __init__(self, 
                 config: TrainingConfig, 
                 device_manager: DeviceManager,
                 progress_monitor: TrainingProgressMonitor):
        super().__init__(config)
        self.device_manager = device_manager
        self.progress_monitor = progress_monitor
        self.device = device_manager.device
    
    def train(self, train_embeddings: np.ndarray, val_embeddings: np.ndarray) -> Dict:
        """
        Enhanced training with progress monitoring and validation split.
        
        Args:
            train_embeddings: Training embeddings
            val_embeddings: Validation embeddings
            
        Returns:
            Dictionary containing training results and metrics
        """
        try:
            start_time = time.time()
            logger.info(f"Starting enhanced autoencoder training on {self.device}")
            
            # Initialize model
            input_dim = train_embeddings.shape[1]
            self.model = AutoencoderModel(input_dim, self.config.hidden_dims)
            self.model.to(self.device)
            
            # Initialize optimizer
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.config.learning_rate
            )
            
            # Prepare data loaders
            train_loader = self._create_data_loader(train_embeddings, shuffle=True)
            val_loader = self._create_data_loader(val_embeddings, shuffle=False)
            
            # Initialize early stopping
            from app.services.model_manager import EarlyStopping
            early_stopping = EarlyStopping(
                patience=self.config.early_stopping_patience,
                min_delta=self.config.min_delta
            )
            
            # Training loop with monitoring
            self.training_history = []
            best_val_loss = float('inf')
            best_model_state = None
            
            for epoch in range(self.config.epochs):
                # Start epoch monitoring
                self.progress_monitor.start_epoch(epoch + 1, len(train_loader))
                
                # Training phase
                train_loss = self._train_epoch(train_loader, epoch)
                
                # Validation phase
                val_loss = self._validate_epoch(val_loader)
                
                # Record history
                epoch_info = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss
                }
                self.training_history.append(epoch_info)
                
                # Update progress monitor
                self.progress_monitor.update_epoch(epoch + 1, train_loss, val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                
                # Check early stopping
                if early_stopping(val_loss):
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Load best model
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
            
            # Calculate final metrics
            final_metrics = self._calculate_enhanced_metrics(train_embeddings, val_embeddings)
            
            training_time = time.time() - start_time
            
            training_result = {
                "success": True,
                "epochs_trained": len(self.training_history),
                "best_val_loss": best_val_loss,
                "final_train_loss": self.training_history[-1]["train_loss"],
                "final_val_loss": self.training_history[-1]["val_loss"],
                "training_time": training_time,
                "model_architecture": self.model.get_architecture_info(),
                "training_config": self.config.__dict__,
                "metrics": final_metrics,
                "training_history": self.training_history,
                "device_info": self.device_manager.device_info
            }
            
            logger.info(f"Enhanced training completed in {training_time:.1f}s. "
                       f"Best validation loss: {best_val_loss:.6f}")
            
            return training_result
            
        except Exception as e:
            logger.error(f"Enhanced autoencoder training failed: {str(e)}")
            raise AutoencoderTrainingError(f"Enhanced training failed: {str(e)}")
    
    def _create_data_loader(self, embeddings: np.ndarray, shuffle: bool = True) -> DataLoader:
        """Create data loader for embeddings."""
        X = torch.FloatTensor(embeddings)
        dataset = TensorDataset(X, X)  # Autoencoder: input = target
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,  # Use 0 for compatibility
            pin_memory=self.device.type == "cuda"
        )
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch with progress monitoring."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
            batch_data = batch_data.to(self.device)
            batch_target = batch_target.to(self.device)
            
            self.optimizer.zero_grad()
            reconstructed = self.model(batch_data)
            loss = self.criterion(reconstructed, batch_target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress monitor
            current_lr = self.optimizer.param_groups[0]['lr']
            self.progress_monitor.update_batch(
                batch_idx + 1, 
                loss.item(), 
                current_lr
            )
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_data, batch_target in val_loader:
                batch_data = batch_data.to(self.device)
                batch_target = batch_target.to(self.device)
                
                reconstructed = self.model(batch_data)
                loss = self.criterion(reconstructed, batch_target)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def _calculate_enhanced_metrics(self, 
                                  train_embeddings: np.ndarray, 
                                  val_embeddings: np.ndarray) -> Dict:
        """Calculate enhanced training metrics."""
        self.model.eval()
        
        # Calculate metrics for both train and validation sets
        train_metrics = self._calculate_set_metrics(train_embeddings, "train")
        val_metrics = self._calculate_set_metrics(val_embeddings, "validation")
        
        return {
            **train_metrics,
            **val_metrics,
            "generalization_gap": val_metrics["validation_mean_error"] - train_metrics["train_mean_error"]
        }
    
    def _calculate_set_metrics(self, embeddings: np.ndarray, prefix: str) -> Dict:
        """Calculate metrics for a specific dataset."""
        X = torch.FloatTensor(embeddings).to(self.device)
        
        with torch.no_grad():
            reconstructed = self.model(X)
            reconstruction_errors = torch.mean((X - reconstructed) ** 2, dim=1)
            
            # Calculate statistics
            mean_error = torch.mean(reconstruction_errors).item()
            std_error = torch.std(reconstruction_errors).item()
            min_error = torch.min(reconstruction_errors).item()
            max_error = torch.max(reconstruction_errors).item()
            
            # Calculate percentiles
            sorted_errors, _ = torch.sort(reconstruction_errors)
            n = len(sorted_errors)
            percentiles = {}
            for p in [90, 95, 99]:
                idx = int(p / 100.0 * n)
                percentiles[f"{prefix}_percentile_{p}"] = sorted_errors[min(idx, n-1)].item()
        
        return {
            f"{prefix}_mean_error": mean_error,
            f"{prefix}_std_error": std_error,
            f"{prefix}_min_error": min_error,
            f"{prefix}_max_error": max_error,
            **percentiles
        }


# Global service instance
_local_training_environment = None


def get_local_training_environment() -> LocalTrainingEnvironment:
    """Get the global local training environment instance."""
    global _local_training_environment
    if _local_training_environment is None:
        _local_training_environment = LocalTrainingEnvironment()
    return _local_training_environment