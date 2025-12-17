"""
Model Manager Service for age-based modeling and anomaly detection.

This service handles autoencoder architecture, training pipelines, and model management
for age-group specific anomaly detection in children's drawings.
"""

import logging
import os
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import hashlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.database import AgeGroupModel, Drawing, DrawingEmbedding
from app.services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


class ModelManagerError(Exception):
    """Base exception for model manager errors."""
    pass


class AutoencoderTrainingError(ModelManagerError):
    """Raised when autoencoder training fails."""
    pass


class ModelLoadingError(ModelManagerError):
    """Raised when model loading fails."""
    pass


@dataclass
class TrainingConfig:
    """Configuration for autoencoder training."""
    hidden_dims: List[int]
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    min_delta: float = 1e-4
    device: str = "auto"


class AutoencoderModel(nn.Module):
    """Autoencoder architecture for embedding reconstruction."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        """
        Initialize autoencoder with specified architecture.
        
        Args:
            input_dim: Dimension of input embeddings
            hidden_dims: List of hidden layer dimensions (encoder path)
        """
        super(AutoencoderModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (reverse of encoder)
        decoder_layers = []
        decoder_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        prev_dim = hidden_dims[-1]  # Start from bottleneck
        
        for i, hidden_dim in enumerate(decoder_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            if i < len(decoder_dims) - 1:  # No activation on output layer
                decoder_layers.extend([nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def get_architecture_info(self) -> Dict:
        """Get information about the model architecture."""
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class EarlyStopping:
    """Smart early stopping utility that handles gradient explosion recovery."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.loss_history = []
        self.explosion_recovery_patience = 5  # Extra patience after explosion
        self.in_recovery = False
    
    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop early with explosion recovery logic."""
        self.loss_history.append(val_loss)
        
        # Detect gradient explosion (loss > 1000x best loss)
        if val_loss > self.best_loss * 1000 and self.best_loss < float('inf'):
            self.in_recovery = True
            if len(self.loss_history) >= 2:
                # Reset counter if we're recovering (loss is decreasing)
                if val_loss < self.loss_history[-2]:
                    self.counter = 0
                    return False
        
        # Normal early stopping logic
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.in_recovery = False
        else:
            self.counter += 1
            
            # Give extra patience if we're recovering from explosion
            effective_patience = self.patience + (self.explosion_recovery_patience if self.in_recovery else 0)
            
            if self.counter >= effective_patience:
                self.early_stop = True
        
        return self.early_stop


class AutoencoderTrainer:
    """Trainer for autoencoder models."""
    
    def __init__(self, config: TrainingConfig, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.device = self._get_device()
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.training_history = []
        
        if self.verbose:
            print(f"  🔧 Initializing trainer on device: {self.device}")
            print(f"  📊 Training config: {config.epochs} epochs, LR: {config.learning_rate}, Batch: {config.batch_size}")
    
    def _get_device(self) -> torch.device:
        """Get the appropriate device for training."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.config.device)
    
    def _prepare_data(self, embeddings: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders."""
        if self.verbose:
            print(f"  📦 Preparing data: {len(embeddings)} samples, {embeddings.shape[1]} features")
        
        # Split into train/validation using numpy arrays first
        X_train, X_val = train_test_split(
            embeddings, 
            test_size=self.config.validation_split, 
            random_state=42
        )
        
        if self.verbose:
            print(f"  📊 Data split: {len(X_train)} training, {len(X_val)} validation samples")
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        X_val = torch.FloatTensor(X_val)
        
        # Create datasets and loaders
        train_dataset = TensorDataset(X_train, X_train)  # Autoencoder: input = target
        val_dataset = TensorDataset(X_val, X_val)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False
        )
        
        if self.verbose:
            print(f"  🔄 Created data loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
        
        return train_loader, val_loader
    
    def train(self, embeddings: np.ndarray) -> Dict:
        """
        Train autoencoder on embeddings.
        
        Args:
            embeddings: Array of embeddings for training
            
        Returns:
            Dictionary containing training results and metrics
        """
        try:
            logger.info(f"Starting autoencoder training with {len(embeddings)} samples")
            
            if self.verbose:
                print(f"  🚀 Starting autoencoder training with {len(embeddings)} samples")
            
            # Initialize model
            input_dim = embeddings.shape[1]
            self.model = AutoencoderModel(input_dim, self.config.hidden_dims)
            self.model.to(self.device)
            
            if self.verbose:
                arch_info = self.model.get_architecture_info()
                print(f"  🏗️  Model architecture: {input_dim} → {' → '.join(map(str, self.config.hidden_dims))} → {input_dim}")
                print(f"  📈 Total parameters: {arch_info['total_parameters']:,}")
            
            # Initialize optimizer
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.config.learning_rate
            )
            
            # Add learning rate scheduler for stability
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.5, 
                patience=5, 
                verbose=self.verbose
            )
            
            if self.verbose:
                print(f"  ⚙️  Optimizer: Adam (LR: {self.config.learning_rate})")
                print(f"  📉 Scheduler: ReduceLROnPlateau (factor: 0.5, patience: 5)")
            
            # Prepare data
            train_loader, val_loader = self._prepare_data(embeddings)
            
            # Initialize early stopping
            early_stopping = EarlyStopping(
                patience=self.config.early_stopping_patience,
                min_delta=self.config.min_delta
            )
            
            if self.verbose:
                print(f"  ⏹️  Early stopping: patience={self.config.early_stopping_patience}, min_delta={self.config.min_delta}")
                print(f"  🎯 Training target: {self.config.epochs} epochs maximum")
                print(f"  ⏱️  Starting training loop...\n")
            
            # Training loop
            self.training_history = []
            best_val_loss = float('inf')
            best_model_state = None
            
            for epoch in range(self.config.epochs):
                epoch_start_time = time.time()
                
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_batches = 0
                
                if self.verbose:
                    print(f"    Epoch {epoch + 1:3d}/{self.config.epochs}: Training", end="", flush=True)
                
                for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
                    batch_data = batch_data.to(self.device)
                    batch_target = batch_target.to(self.device)
                    
                    self.optimizer.zero_grad()
                    reconstructed = self.model(batch_data)
                    loss = self.criterion(reconstructed, batch_target)
                    loss.backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    train_loss += loss.item()
                    train_batches += 1
                    
                    # Show batch progress in verbose mode
                    if self.verbose and len(train_loader) > 10:
                        progress_points = [len(train_loader) // 4, len(train_loader) // 2, 3 * len(train_loader) // 4]
                        if (batch_idx + 1) in progress_points:
                            progress_pct = int((batch_idx + 1) / len(train_loader) * 100)
                            print(f" {progress_pct}%", end="", flush=True)
                
                avg_train_loss = train_loss / train_batches
                
                # Check for NaN or infinite loss
                if not torch.isfinite(torch.tensor(avg_train_loss)):
                    if self.verbose:
                        print(f" ✗ Training stopped: Non-finite loss detected")
                    logger.error(f"Training stopped due to non-finite loss: {avg_train_loss}")
                    break
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                val_batches = 0
                
                if self.verbose:
                    print(" → Validating", end="", flush=True)
                
                with torch.no_grad():
                    for batch_data, batch_target in val_loader:
                        batch_data = batch_data.to(self.device)
                        batch_target = batch_target.to(self.device)
                        
                        reconstructed = self.model(batch_data)
                        loss = self.criterion(reconstructed, batch_target)
                        
                        val_loss += loss.item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                epoch_time = time.time() - epoch_start_time
                
                # Check for exploding gradients (validation loss > 1000x training loss)
                if avg_val_loss > avg_train_loss * 1000:
                    if self.verbose:
                        print(f" ⚠ Warning: Possible exploding gradients detected")
                    logger.warning(f"Large validation loss detected: {avg_val_loss} vs train: {avg_train_loss}")
                
                # Record history
                epoch_info = {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss
                }
                self.training_history.append(epoch_info)
                
                # Save best model
                is_best = False
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = self.model.state_dict().copy()
                    is_best = True
                
                # Update learning rate scheduler
                self.scheduler.step(avg_val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Verbose progress output
                if self.verbose:
                    best_indicator = " ⭐" if is_best else ""
                    lr_info = f" (LR: {current_lr:.2e})" if current_lr != self.config.learning_rate else ""
                    
                    # Progress bar for epochs
                    progress = (epoch + 1) / self.config.epochs
                    bar_length = 20
                    filled_length = int(bar_length * progress)
                    bar = "█" * filled_length + "░" * (bar_length - filled_length)
                    
                    # Loss trend indicator
                    if len(self.training_history) > 1:
                        prev_val_loss = self.training_history[-2]["val_loss"]
                        if avg_val_loss < prev_val_loss:
                            trend = "↓"
                        elif avg_val_loss > prev_val_loss:
                            trend = "↑"
                        else:
                            trend = "→"
                    else:
                        trend = ""
                    
                    print(f" [{bar}] {progress*100:5.1f}% | Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f} {trend} ({epoch_time:.1f}s){best_indicator}{lr_info}")
                    
                    # Show early stopping counter if patience is being tested
                    if early_stopping.counter > 0:
                        patience_left = self.config.early_stopping_patience - early_stopping.counter
                        if early_stopping.in_recovery:
                            print(f"    🔄 Recovery mode: {patience_left + early_stopping.explosion_recovery_patience} patience left")
                        else:
                            print(f"    ⏳ Early stopping: {patience_left} patience left")
                
                # Log progress (less frequent in verbose mode)
                log_frequency = 20 if self.verbose else 10
                if (epoch + 1) % log_frequency == 0:
                    logger.info(f"Epoch {epoch + 1}/{self.config.epochs}: "
                              f"Train Loss: {avg_train_loss:.6f}, "
                              f"Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.2e}")
                
                # Check early stopping
                if early_stopping(avg_val_loss):
                    if self.verbose:
                        print(f"    ⏹️  Early stopping triggered after {epoch + 1} epochs")
                        print(f"    🎯 Best validation loss: {best_val_loss:.6f}")
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
                # Emergency stop for exploding gradients
                if avg_val_loss > 100000:  # Threshold for exploding gradients
                    if self.verbose:
                        print(f"    🚨 Emergency stop: Loss too high ({avg_val_loss:.0f})")
                        print(f"    💥 Possible gradient explosion detected")
                    logger.error(f"Emergency stop due to exploding gradients at epoch {epoch + 1}")
                    break
            
            # Load best model
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
                if self.verbose:
                    print(f"\n    ✅ Loaded best model (validation loss: {best_val_loss:.6f})")
            
            # Calculate final metrics
            if self.verbose:
                print(f"    📊 Calculating final metrics on {len(embeddings)} samples...")
            final_metrics = self._calculate_metrics(embeddings)
            
            training_result = {
                "success": True,
                "epochs_trained": len(self.training_history),
                "best_val_loss": best_val_loss,
                "final_train_loss": self.training_history[-1]["train_loss"],
                "final_val_loss": self.training_history[-1]["val_loss"],
                "model_architecture": self.model.get_architecture_info(),
                "training_config": self.config.__dict__,
                "metrics": final_metrics,
                "training_history": self.training_history
            }
            
            if self.verbose:
                print(f"\n    🎉 Training completed successfully!")
                print(f"    📈 Epochs trained: {len(self.training_history)}")
                print(f"    🏆 Best validation loss: {best_val_loss:.6f}")
                print(f"    📊 Final metrics:")
                print(f"      • Mean reconstruction error: {final_metrics['mean_reconstruction_error']:.6f}")
                print(f"      • 95th percentile threshold: {final_metrics['percentile_95']:.6f}")
                print(f"      • 99th percentile threshold: {final_metrics['percentile_99']:.6f}")
            
            logger.info(f"Training completed successfully. Best validation loss: {best_val_loss:.6f}")
            return training_result
            
        except Exception as e:
            logger.error(f"Autoencoder training failed: {str(e)}")
            raise AutoencoderTrainingError(f"Training failed: {str(e)}")
    
    def _calculate_metrics(self, embeddings: np.ndarray) -> Dict:
        """Calculate training metrics on the full dataset."""
        if self.model is None:
            return {}
        
        self.model.eval()
        X = torch.FloatTensor(embeddings).to(self.device)
        
        with torch.no_grad():
            reconstructed = self.model(X)
            mse_loss = self.criterion(reconstructed, X).item()
            
            # Calculate reconstruction errors for each sample
            reconstruction_errors = torch.mean((X - reconstructed) ** 2, dim=1)
            
            # Use torch operations to avoid numpy compatibility issues
            mean_error = torch.mean(reconstruction_errors).item()
            std_error = torch.std(reconstruction_errors).item()
            min_error = torch.min(reconstruction_errors).item()
            max_error = torch.max(reconstruction_errors).item()
            
            # Calculate percentiles using torch
            sorted_errors, _ = torch.sort(reconstruction_errors)
            n = len(sorted_errors)
            percentile_95_idx = int(0.95 * n)
            percentile_99_idx = int(0.99 * n)
            percentile_95 = sorted_errors[min(percentile_95_idx, n-1)].item()
            percentile_99 = sorted_errors[min(percentile_99_idx, n-1)].item()
            
            metrics = {
                "mse_loss": mse_loss,
                "mean_reconstruction_error": mean_error,
                "std_reconstruction_error": std_error,
                "min_reconstruction_error": min_error,
                "max_reconstruction_error": max_error,
                "percentile_95": percentile_95,
                "percentile_99": percentile_99
            }
        
        return metrics


class ModelManager:
    """Manager for age-based autoencoder models."""
    
    def __init__(self):
        self.models_dir = Path("static/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_service = get_embedding_service()
        self._loaded_models = {}  # Cache for loaded models
        self.model = None  # Current active model for testing compatibility
    
    def _get_model_path(self, age_group_model_id: int) -> Path:
        """Get the file path for a model."""
        return self.models_dir / f"autoencoder_{age_group_model_id}.pkl"
    
    def _get_embeddings_for_age_group(self, 
                                    age_min: float, 
                                    age_max: float,
                                    db: Session) -> np.ndarray:
        """Retrieve embeddings for a specific age group."""
        # Query drawings in age range
        drawings = db.query(Drawing).filter(
            Drawing.age_years >= age_min,
            Drawing.age_years <= age_max
        ).all()
        
        if not drawings:
            raise ModelManagerError(f"No drawings found for age range {age_min}-{age_max}")
        
        # Get embeddings for these drawings
        embeddings = []
        for drawing in drawings:
            # Get the most recent embedding for this drawing
            embedding_record = db.query(DrawingEmbedding).filter(
                DrawingEmbedding.drawing_id == drawing.id
            ).order_by(DrawingEmbedding.created_timestamp.desc()).first()
            
            if embedding_record:
                # Deserialize embedding using serialization utilities
                from app.utils.embedding_serialization import deserialize_embedding_from_db
                embedding_data = deserialize_embedding_from_db(embedding_record.embedding_vector)
                embeddings.append(embedding_data)
        
        if not embeddings:
            raise ModelManagerError(f"No embeddings found for age range {age_min}-{age_max}")
        
        return np.array(embeddings)
    
    def train_age_group_model(self, 
                            age_min: float, 
                            age_max: float,
                            config: TrainingConfig,
                            db: Session) -> Dict:
        """
        Train an autoencoder model for a specific age group.
        
        Args:
            age_min: Minimum age for the group
            age_max: Maximum age for the group
            config: Training configuration
            db: Database session
            
        Returns:
            Dictionary containing training results and model info
        """
        try:
            logger.info(f"Training autoencoder for age group {age_min}-{age_max}")
            
            # Get embeddings for age group
            embeddings = self._get_embeddings_for_age_group(age_min, age_max, db)
            logger.info(f"Found {len(embeddings)} embeddings for training")
            
            # Initialize trainer and train model
            trainer = AutoencoderTrainer(config)
            training_result = trainer.train(embeddings)
            
            # Calculate threshold (95th percentile of reconstruction errors)
            threshold = training_result["metrics"]["percentile_95"]
            
            # Save model to database
            model_params = {
                "training_config": config.__dict__,
                "architecture": training_result["model_architecture"],
                "training_metrics": training_result["metrics"],
                "training_history": training_result["training_history"]
            }
            
            age_group_model = AgeGroupModel(
                age_min=age_min,
                age_max=age_max,
                model_type="autoencoder",
                vision_model="vit",
                parameters=json.dumps(model_params),
                sample_count=len(embeddings),
                threshold=threshold
            )
            
            db.add(age_group_model)
            db.commit()
            db.refresh(age_group_model)
            
            # Save model weights
            model_path = self._get_model_path(age_group_model.id)
            torch.save({
                'model_state_dict': trainer.model.state_dict(),
                'model_architecture': trainer.model.get_architecture_info(),
                'training_config': config.__dict__
            }, model_path)
            
            logger.info(f"Model saved successfully with ID {age_group_model.id}")
            
            result = {
                "model_id": age_group_model.id,
                "age_range": (age_min, age_max),
                "sample_count": len(embeddings),
                "threshold": threshold,
                "training_result": training_result,
                "model_path": str(model_path)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to train age group model: {str(e)}")
            raise AutoencoderTrainingError(f"Training failed: {str(e)}")
    
    def load_model(self, age_group_model_id: int, db: Session) -> AutoencoderModel:
        """
        Load a trained autoencoder model.
        
        Args:
            age_group_model_id: ID of the age group model
            db: Database session
            
        Returns:
            Loaded autoencoder model
        """
        # Check cache first
        if age_group_model_id in self._loaded_models:
            return self._loaded_models[age_group_model_id]
        
        try:
            # Get model record from database
            age_group_model = db.query(AgeGroupModel).filter(
                AgeGroupModel.id == age_group_model_id
            ).first()
            
            if not age_group_model:
                raise ModelLoadingError(f"Age group model {age_group_model_id} not found")
            
            # Load model weights
            model_path = self._get_model_path(age_group_model_id)
            if not model_path.exists():
                raise ModelLoadingError(f"Model file not found: {model_path}")
            
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Reconstruct model architecture
            arch_info = checkpoint['model_architecture']
            model = AutoencoderModel(
                input_dim=arch_info['input_dim'],
                hidden_dims=arch_info['hidden_dims']
            )
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Cache the model
            self._loaded_models[age_group_model_id] = model
            
            logger.info(f"Successfully loaded model {age_group_model_id}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {age_group_model_id}: {str(e)}")
            raise ModelLoadingError(f"Loading failed: {str(e)}")
    
    def compute_reconstruction_loss(self, 
                                  embedding: np.ndarray, 
                                  age_group_model_id: int,
                                  db: Session) -> float:
        """
        Compute reconstruction loss for an embedding using a specific model.
        
        Args:
            embedding: Input embedding vector
            age_group_model_id: ID of the age group model to use
            db: Database session
            
        Returns:
            Reconstruction loss value
        """
        try:
            # Load model
            model = self.load_model(age_group_model_id, db)
            
            # Convert embedding to tensor
            X = torch.FloatTensor(embedding).unsqueeze(0)  # Add batch dimension
            
            # Compute reconstruction
            with torch.no_grad():
                reconstructed = model(X)
                loss = torch.mean((X - reconstructed) ** 2).item()
            
            return loss
            
        except Exception as e:
            logger.error(f"Failed to compute reconstruction loss: {str(e)}")
            raise ModelManagerError(f"Reconstruction loss computation failed: {str(e)}")
    
    def get_model_info(self, age_group_model_id: int, db: Session) -> Dict:
        """Get information about a specific model."""
        age_group_model = db.query(AgeGroupModel).filter(
            AgeGroupModel.id == age_group_model_id
        ).first()
        
        if not age_group_model:
            raise ModelLoadingError(f"Age group model {age_group_model_id} not found")
        
        # Parse parameters
        parameters = json.loads(age_group_model.parameters)
        
        return {
            "id": age_group_model.id,
            "age_range": (age_group_model.age_min, age_group_model.age_max),
            "model_type": age_group_model.model_type,
            "vision_model": age_group_model.vision_model,
            "sample_count": age_group_model.sample_count,
            "threshold": age_group_model.threshold,
            "created_timestamp": age_group_model.created_timestamp,
            "is_active": age_group_model.is_active,
            "parameters": parameters
        }
    
    def list_models(self, db: Session) -> List[Dict]:
        """List all age group models."""
        models = db.query(AgeGroupModel).order_by(AgeGroupModel.age_min).all()
        return [self.get_model_info(model.id, db) for model in models]
    
    def _calculate_reconstruction_loss(self, 
                                     original_embedding: np.ndarray, 
                                     reconstructed_embedding: np.ndarray) -> float:
        """Calculate reconstruction loss between original and reconstructed embeddings."""
        # Mean squared error
        mse = np.mean((original_embedding - reconstructed_embedding) ** 2)
        return float(mse)
    
    def _calculate_metrics(self, embeddings: np.ndarray) -> Dict:
        """Calculate training metrics for embeddings."""
        if self.model is None:
            raise ModelManagerError("No model available for metrics calculation")
        
        # Get reconstructions from model
        reconstructions = self.model.predict(embeddings)
        
        # Calculate losses for each sample
        losses = []
        for i in range(len(embeddings)):
            loss = self._calculate_reconstruction_loss(embeddings[i], reconstructions[i])
            losses.append(loss)
        
        losses = np.array(losses)
        
        return {
            'mean_loss': float(np.mean(losses)),
            'std_loss': float(np.std(losses)),
            'min_loss': float(np.min(losses)),
            'max_loss': float(np.max(losses))
        }
    
    def clear_model_cache(self) -> None:
        """Clear the loaded models cache."""
        self._loaded_models.clear()
        logger.info("Model cache cleared")


# Global model manager instance
_model_manager = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager