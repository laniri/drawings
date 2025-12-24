"""
Training Report Generation Service for Children's Drawing Anomaly Detection System

This module provides comprehensive training metrics collection, validation curve plotting,
performance visualization, and summary report generation with model performance analysis.
"""

import json
import logging
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sqlalchemy.orm import Session

from app.core.exceptions import ValidationError
from app.models.database import TrainingJob, TrainingReport

logger = logging.getLogger(__name__)


class TrainingReportError(Exception):
    """Base exception for training report generation errors."""

    pass


class MetricsCalculationError(TrainingReportError):
    """Raised when metrics calculation fails."""

    pass


class VisualizationError(TrainingReportError):
    """Raised when visualization generation fails."""

    pass


@dataclass
class TrainingMetrics:
    """Container for comprehensive training metrics."""

    # Basic metrics
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    best_epoch: int
    total_epochs: int
    training_time_seconds: float

    # Loss statistics
    train_loss_mean: float
    train_loss_std: float
    val_loss_mean: float
    val_loss_std: float

    # Convergence metrics
    convergence_epoch: Optional[int]
    early_stopping_triggered: bool
    overfitting_detected: bool
    generalization_gap: float

    # Model performance
    reconstruction_error_stats: Dict[str, float]
    anomaly_detection_threshold: float
    validation_accuracy_estimate: float

    # Training stability
    loss_variance: float
    gradient_norm_stats: Optional[Dict[str, float]]
    learning_rate_schedule: List[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


@dataclass
class ModelArchitectureInfo:
    """Container for model architecture information."""

    model_type: str
    input_dimension: int
    hidden_dimensions: List[int]
    latent_dimension: int
    total_parameters: int
    trainable_parameters: int
    model_size_mb: float
    activation_functions: List[str]
    dropout_rate: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert architecture info to dictionary."""
        return asdict(self)


@dataclass
class TrainingConfiguration:
    """Container for training configuration details."""

    learning_rate: float
    batch_size: int
    epochs: int
    optimizer: str
    loss_function: str
    early_stopping_patience: int
    min_delta: float
    data_split_ratios: Dict[str, float]
    device_used: str
    random_seed: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)


class TrainingReportGenerator:
    """Generator for comprehensive training reports with visualizations."""

    def __init__(self):
        self.report_template_dir = Path("templates/reports")
        self.output_dir = Path("outputs/training_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configure matplotlib for non-interactive use
        plt.style.use("seaborn-v0_8")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 10
        plt.rcParams["axes.titlesize"] = 12
        plt.rcParams["axes.labelsize"] = 10
        plt.rcParams["xtick.labelsize"] = 9
        plt.rcParams["ytick.labelsize"] = 9
        plt.rcParams["legend.fontsize"] = 9

        logger.info("Training Report Generator initialized")

    def generate_comprehensive_report(
        self,
        training_job_id: int,
        training_result: Dict[str, Any],
        model_info: Optional[Dict] = None,
        db: Session = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive training report with all metrics and visualizations.

        Args:
            training_job_id: ID of the training job
            training_result: Training results dictionary
            model_info: Optional model information
            db: Database session

        Returns:
            Dictionary containing complete report information

        Raises:
            TrainingReportError: If report generation fails
        """
        try:
            logger.info(
                f"Generating comprehensive training report for job {training_job_id}"
            )

            # Create report directory
            report_dir = (
                self.output_dir
                / f"job_{training_job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            report_dir.mkdir(parents=True, exist_ok=True)

            # Extract and calculate metrics
            metrics = self._calculate_comprehensive_metrics(training_result)

            # Extract model architecture information
            architecture_info = self._extract_architecture_info(
                training_result, model_info
            )

            # Extract training configuration
            config_info = self._extract_configuration_info(training_result)

            # Generate visualizations
            visualization_paths = self._generate_all_visualizations(
                training_result, metrics, report_dir
            )

            # Generate summary statistics
            summary_stats = self._generate_summary_statistics(
                metrics, architecture_info
            )

            # Create performance analysis
            performance_analysis = self._analyze_model_performance(
                metrics, training_result
            )

            # Generate recommendations
            recommendations = self._generate_training_recommendations(
                metrics, training_result
            )

            # Create comprehensive report document
            report_content = {
                "report_metadata": {
                    "training_job_id": training_job_id,
                    "generation_timestamp": datetime.now().isoformat(),
                    "report_version": "1.0",
                    "generator": "TrainingReportGenerator",
                },
                "training_metrics": metrics.to_dict(),
                "model_architecture": architecture_info.to_dict(),
                "training_configuration": config_info.to_dict(),
                "summary_statistics": summary_stats,
                "performance_analysis": performance_analysis,
                "recommendations": recommendations,
                "visualizations": visualization_paths,
                "raw_training_data": training_result,
            }

            # Save report to files
            report_paths = self._save_report_files(report_content, report_dir)

            # Update database if provided
            if db:
                self._update_database_record(
                    training_job_id, report_content, report_paths, db
                )

            logger.info(f"Training report generated successfully: {report_dir}")

            return {
                "success": True,
                "report_directory": str(report_dir),
                "report_paths": report_paths,
                "metrics_summary": summary_stats,
                "performance_score": performance_analysis.get("overall_score", 0.0),
            }

        except Exception as e:
            logger.error(f"Failed to generate training report: {str(e)}")
            raise TrainingReportError(f"Report generation failed: {str(e)}")

    def _calculate_comprehensive_metrics(
        self, training_result: Dict[str, Any]
    ) -> TrainingMetrics:
        """Calculate comprehensive training metrics from training results."""
        try:
            history = training_result.get("training_history", [])
            if not history:
                raise MetricsCalculationError("No training history available")

            # Extract loss values
            train_losses = [h["train_loss"] for h in history]
            val_losses = [h["val_loss"] for h in history]

            # Basic metrics
            final_train_loss = train_losses[-1]
            final_val_loss = val_losses[-1]
            best_val_loss = min(val_losses)
            best_epoch = val_losses.index(best_val_loss) + 1
            total_epochs = len(history)
            training_time = training_result.get("training_time", 0.0)

            # Loss statistics
            train_loss_mean = np.mean(train_losses)
            train_loss_std = np.std(train_losses)
            val_loss_mean = np.mean(val_losses)
            val_loss_std = np.std(val_losses)

            # Convergence analysis
            convergence_epoch = self._detect_convergence(val_losses)
            early_stopping_triggered = total_epochs < training_result.get(
                "max_epochs", total_epochs
            )
            overfitting_detected = self._detect_overfitting(train_losses, val_losses)
            generalization_gap = final_val_loss - final_train_loss

            # Reconstruction error statistics
            metrics_data = training_result.get("metrics", {})
            reconstruction_stats = {
                "mean": metrics_data.get("validation_mean_error", 0.0),
                "std": metrics_data.get("validation_std_error", 0.0),
                "min": metrics_data.get("validation_min_error", 0.0),
                "max": metrics_data.get("validation_max_error", 0.0),
                "percentile_95": metrics_data.get("validation_percentile_95", 0.0),
                "percentile_99": metrics_data.get("validation_percentile_99", 0.0),
            }

            # Anomaly detection threshold (95th percentile)
            anomaly_threshold = reconstruction_stats["percentile_95"]

            # Validation accuracy estimate (1 - normalized loss)
            val_accuracy_estimate = max(
                0.0, 1.0 - (final_val_loss / max(train_loss_mean, 1e-6))
            )

            # Training stability
            loss_variance = (
                np.var(val_losses[-10:])
                if len(val_losses) >= 10
                else np.var(val_losses)
            )

            # Learning rate schedule (if available)
            lr_schedule = training_result.get("learning_rate_schedule", [])

            return TrainingMetrics(
                final_train_loss=final_train_loss,
                final_val_loss=final_val_loss,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                total_epochs=total_epochs,
                training_time_seconds=training_time,
                train_loss_mean=train_loss_mean,
                train_loss_std=train_loss_std,
                val_loss_mean=val_loss_mean,
                val_loss_std=val_loss_std,
                convergence_epoch=convergence_epoch,
                early_stopping_triggered=early_stopping_triggered,
                overfitting_detected=overfitting_detected,
                generalization_gap=generalization_gap,
                reconstruction_error_stats=reconstruction_stats,
                anomaly_detection_threshold=anomaly_threshold,
                validation_accuracy_estimate=val_accuracy_estimate,
                loss_variance=loss_variance,
                gradient_norm_stats=None,  # Would need gradient tracking
                learning_rate_schedule=lr_schedule,
            )

        except Exception as e:
            logger.error(f"Failed to calculate metrics: {str(e)}")
            raise MetricsCalculationError(f"Metrics calculation failed: {str(e)}")

    def _detect_convergence(
        self, val_losses: List[float], window_size: int = 10, threshold: float = 1e-4
    ) -> Optional[int]:
        """Detect convergence epoch based on loss stabilization."""
        if len(val_losses) < window_size * 2:
            return None

        for i in range(window_size, len(val_losses) - window_size):
            window_before = val_losses[i - window_size : i]
            window_after = val_losses[i : i + window_size]

            if abs(np.mean(window_after) - np.mean(window_before)) < threshold:
                return i + 1

        return None

    def _detect_overfitting(
        self, train_losses: List[float], val_losses: List[float], threshold: float = 0.1
    ) -> bool:
        """Detect overfitting based on train/validation loss divergence."""
        if len(train_losses) < 10 or len(val_losses) < 10:
            return False

        # Check last 10 epochs
        recent_train = train_losses[-10:]
        recent_val = val_losses[-10:]

        # Overfitting if validation loss increases while training loss decreases
        train_trend = np.polyfit(range(len(recent_train)), recent_train, 1)[0]
        val_trend = np.polyfit(range(len(recent_val)), recent_val, 1)[0]

        return train_trend < -threshold and val_trend > threshold

    def _extract_architecture_info(
        self, training_result: Dict, model_info: Optional[Dict]
    ) -> ModelArchitectureInfo:
        """Extract model architecture information."""
        arch_data = training_result.get("model_architecture", {})

        if model_info:
            arch_data.update(model_info)

        return ModelArchitectureInfo(
            model_type=arch_data.get("model_type", "autoencoder"),
            input_dimension=arch_data.get("input_dim", 0),
            hidden_dimensions=arch_data.get("hidden_dims", []),
            latent_dimension=arch_data.get("latent_dim", 0),
            total_parameters=arch_data.get("total_parameters", 0),
            trainable_parameters=arch_data.get("trainable_parameters", 0),
            model_size_mb=arch_data.get("model_size_mb", 0.0),
            activation_functions=arch_data.get("activation_functions", ["ReLU"]),
            dropout_rate=arch_data.get("dropout_rate", 0.0),
        )

    def _extract_configuration_info(
        self, training_result: Dict
    ) -> TrainingConfiguration:
        """Extract training configuration information."""
        config_data = training_result.get("training_config", {})

        return TrainingConfiguration(
            learning_rate=config_data.get("learning_rate", 0.001),
            batch_size=config_data.get("batch_size", 32),
            epochs=config_data.get("epochs", 100),
            optimizer=config_data.get("optimizer", "Adam"),
            loss_function=config_data.get("loss_function", "MSE"),
            early_stopping_patience=config_data.get("early_stopping_patience", 10),
            min_delta=config_data.get("min_delta", 1e-4),
            data_split_ratios={
                "train": config_data.get("train_split", 0.7),
                "validation": config_data.get("validation_split", 0.2),
                "test": config_data.get("test_split", 0.1),
            },
            device_used=training_result.get("device_info", {}).get("type", "unknown"),
            random_seed=config_data.get("random_seed", None),
        )

    def _generate_all_visualizations(
        self, training_result: Dict, metrics: TrainingMetrics, output_dir: Path
    ) -> Dict[str, str]:
        """Generate all training visualizations."""
        try:
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)

            visualization_paths = {}

            # 1. Loss curves
            loss_plot_path = self._create_loss_curves_plot(training_result, output_dir)
            visualization_paths["loss_curves"] = str(loss_plot_path)

            # 2. Metrics dashboard
            metrics_plot_path = self._create_metrics_dashboard(metrics, output_dir)
            visualization_paths["metrics_dashboard"] = str(metrics_plot_path)

            # 3. Reconstruction error distribution
            error_dist_path = self._create_error_distribution_plot(
                training_result, output_dir
            )
            visualization_paths["error_distribution"] = str(error_dist_path)

            # 4. Training progress timeline
            progress_path = self._create_training_progress_plot(
                training_result, output_dir
            )
            visualization_paths["training_progress"] = str(progress_path)

            # 5. Model architecture diagram
            arch_path = self._create_architecture_diagram(training_result, output_dir)
            visualization_paths["architecture_diagram"] = str(arch_path)

            # 6. Performance summary
            summary_path = self._create_performance_summary_plot(metrics, output_dir)
            visualization_paths["performance_summary"] = str(summary_path)

            return visualization_paths

        except Exception as e:
            logger.error(f"Failed to generate visualizations: {str(e)}")
            raise VisualizationError(f"Visualization generation failed: {str(e)}")

    def _create_loss_curves_plot(self, training_result: Dict, output_dir: Path) -> Path:
        """Create loss curves visualization."""
        history = training_result.get("training_history", [])
        if not history:
            raise VisualizationError("No training history available for loss curves")

        epochs = [h["epoch"] for h in history]
        train_losses = [h["train_loss"] for h in history]
        val_losses = [h["val_loss"] for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Loss curves
        ax1.plot(epochs, train_losses, label="Training Loss", color="blue", linewidth=2)
        ax1.plot(epochs, val_losses, label="Validation Loss", color="red", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss Curves")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Log scale loss curves
        ax2.semilogy(
            epochs, train_losses, label="Training Loss", color="blue", linewidth=2
        )
        ax2.semilogy(
            epochs, val_losses, label="Validation Loss", color="red", linewidth=2
        )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss (log scale)")
        ax2.set_title("Loss Curves (Log Scale)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = output_dir / "loss_curves.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def _create_metrics_dashboard(
        self, metrics: TrainingMetrics, output_dir: Path
    ) -> Path:
        """Create comprehensive metrics dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Loss comparison
        losses = [
            metrics.final_train_loss,
            metrics.final_val_loss,
            metrics.best_val_loss,
        ]
        loss_labels = ["Final Train", "Final Val", "Best Val"]
        axes[0, 0].bar(loss_labels, losses, color=["blue", "red", "green"])
        axes[0, 0].set_title("Loss Comparison")
        axes[0, 0].set_ylabel("Loss")

        # 2. Training statistics
        stats_data = {
            "Train Loss": [metrics.train_loss_mean, metrics.train_loss_std],
            "Val Loss": [metrics.val_loss_mean, metrics.val_loss_std],
        }
        x = np.arange(len(stats_data))
        width = 0.35
        axes[0, 1].bar(
            x - width / 2,
            [stats_data["Train Loss"][0], stats_data["Val Loss"][0]],
            width,
            label="Mean",
            color="lightblue",
        )
        axes[0, 1].bar(
            x + width / 2,
            [stats_data["Train Loss"][1], stats_data["Val Loss"][1]],
            width,
            label="Std",
            color="lightcoral",
        )
        axes[0, 1].set_title("Loss Statistics")
        axes[0, 1].set_ylabel("Value")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(["Train Loss", "Val Loss"])
        axes[0, 1].legend()

        # 3. Reconstruction error percentiles
        error_stats = metrics.reconstruction_error_stats
        percentiles = ["mean", "percentile_95", "percentile_99", "max"]
        values = [error_stats.get(p, 0) for p in percentiles]
        axes[0, 2].bar(percentiles, values, color="orange")
        axes[0, 2].set_title("Reconstruction Error Statistics")
        axes[0, 2].set_ylabel("Error")
        axes[0, 2].tick_params(axis="x", rotation=45)

        # 4. Training progress indicators
        progress_metrics = {
            "Epochs": metrics.total_epochs,
            "Best Epoch": metrics.best_epoch,
            "Convergence": metrics.convergence_epoch or 0,
        }
        axes[1, 0].bar(
            progress_metrics.keys(), progress_metrics.values(), color="purple"
        )
        axes[1, 0].set_title("Training Progress")
        axes[1, 0].set_ylabel("Epoch")

        # 5. Performance indicators
        performance_data = {
            "Val Accuracy": metrics.validation_accuracy_estimate,
            "Generalization": max(0, 1 - abs(metrics.generalization_gap)),
            "Stability": max(0, 1 - metrics.loss_variance),
        }
        axes[1, 1].bar(
            performance_data.keys(), performance_data.values(), color="green"
        )
        axes[1, 1].set_title("Performance Indicators")
        axes[1, 1].set_ylabel("Score (0-1)")
        axes[1, 1].set_ylim(0, 1)

        # 6. Training summary text
        summary_text = f"""Training Summary:
        
Total Epochs: {metrics.total_epochs}
Best Epoch: {metrics.best_epoch}
Training Time: {metrics.training_time_seconds:.1f}s

Final Train Loss: {metrics.final_train_loss:.6f}
Final Val Loss: {metrics.final_val_loss:.6f}
Best Val Loss: {metrics.best_val_loss:.6f}

Early Stopping: {'Yes' if metrics.early_stopping_triggered else 'No'}
Overfitting: {'Detected' if metrics.overfitting_detected else 'Not Detected'}
Convergence: {'Epoch ' + str(metrics.convergence_epoch) if metrics.convergence_epoch else 'Not Detected'}

Anomaly Threshold: {metrics.anomaly_detection_threshold:.6f}
"""
        axes[1, 2].text(
            0.05,
            0.95,
            summary_text,
            transform=axes[1, 2].transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
        )
        axes[1, 2].set_title("Training Summary")
        axes[1, 2].axis("off")

        plt.tight_layout()

        output_path = output_dir / "metrics_dashboard.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def _create_error_distribution_plot(
        self, training_result: Dict, output_dir: Path
    ) -> Path:
        """Create reconstruction error distribution plot."""
        metrics = training_result.get("metrics", {})

        # Create synthetic distribution based on statistics
        mean_error = metrics.get("validation_mean_error", 0.1)
        std_error = metrics.get("validation_std_error", 0.05)

        # Generate synthetic data for visualization
        np.random.seed(42)
        errors = np.random.normal(mean_error, std_error, 1000)
        errors = np.clip(errors, 0, None)  # Ensure non-negative

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Histogram
        ax1.hist(errors, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
        ax1.axvline(
            mean_error, color="red", linestyle="--", label=f"Mean: {mean_error:.4f}"
        )
        ax1.axvline(
            metrics.get("validation_percentile_95", mean_error + 2 * std_error),
            color="orange",
            linestyle="--",
            label="95th Percentile",
        )
        ax1.set_xlabel("Reconstruction Error")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Reconstruction Error Distribution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(errors, orientation="vertical")
        ax2.set_ylabel("Reconstruction Error")
        ax2.set_title("Error Distribution Box Plot")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = output_dir / "error_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def _create_training_progress_plot(
        self, training_result: Dict, output_dir: Path
    ) -> Path:
        """Create training progress timeline."""
        history = training_result.get("training_history", [])
        if not history:
            raise VisualizationError("No training history available")

        epochs = [h["epoch"] for h in history]
        train_losses = [h["train_loss"] for h in history]
        val_losses = [h["val_loss"] for h in history]

        fig, ax = plt.subplots(figsize=(12, 8))

        # Create timeline plot
        ax.plot(
            epochs, train_losses, "b-", linewidth=2, label="Training Loss", alpha=0.8
        )
        ax.plot(
            epochs, val_losses, "r-", linewidth=2, label="Validation Loss", alpha=0.8
        )

        # Mark best epoch
        best_epoch = val_losses.index(min(val_losses)) + 1
        best_loss = min(val_losses)
        ax.scatter(
            [best_epoch],
            [best_loss],
            color="gold",
            s=100,
            zorder=5,
            label=f"Best Epoch ({best_epoch})",
        )

        # Add annotations for key events
        if len(epochs) > 10:
            # Mark early phase
            ax.axvspan(
                1, min(10, len(epochs)), alpha=0.1, color="green", label="Early Phase"
            )

            # Mark final phase
            final_start = max(1, len(epochs) - 10)
            ax.axvspan(
                final_start, len(epochs), alpha=0.1, color="red", label="Final Phase"
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Progress Timeline")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = output_dir / "training_progress.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def _create_architecture_diagram(
        self, training_result: Dict, output_dir: Path
    ) -> Path:
        """Create model architecture diagram."""
        arch_info = training_result.get("model_architecture", {})

        fig, ax = plt.subplots(figsize=(12, 8))

        # Extract architecture details
        input_dim = arch_info.get("input_dim", 512)
        hidden_dims = arch_info.get("hidden_dims", [256, 128, 64])
        latent_dim = arch_info.get("latent_dim", 32)

        # Create architecture visualization
        layers = (
            [input_dim] + hidden_dims + [latent_dim] + hidden_dims[::-1] + [input_dim]
        )
        layer_names = (
            ["Input"]
            + [f"Hidden {i+1}" for i in range(len(hidden_dims))]
            + ["Latent"]
            + [f"Hidden {len(hidden_dims)-i}" for i in range(len(hidden_dims))]
            + ["Output"]
        )

        # Plot architecture
        x_positions = np.linspace(0, 10, len(layers))
        y_center = 0

        for i, (x, dim, name) in enumerate(zip(x_positions, layers, layer_names)):
            # Draw layer
            height = min(dim / max(layers) * 3, 3)  # Scale height
            rect = plt.Rectangle(
                (x - 0.3, y_center - height / 2),
                0.6,
                height,
                facecolor="lightblue",
                edgecolor="black",
                linewidth=2,
            )
            ax.add_patch(rect)

            # Add layer label
            ax.text(
                x,
                y_center - height / 2 - 0.3,
                f"{name}\n({dim})",
                ha="center",
                va="top",
                fontsize=9,
                weight="bold",
            )

            # Draw connections
            if i < len(layers) - 1:
                ax.arrow(
                    x + 0.3,
                    y_center,
                    x_positions[i + 1] - x - 0.6,
                    0,
                    head_width=0.1,
                    head_length=0.1,
                    fc="gray",
                    ec="gray",
                )

        # Add title and labels
        ax.set_title("Autoencoder Architecture", fontsize=14, weight="bold")
        ax.set_xlim(-1, 11)
        ax.set_ylim(-3, 3)
        ax.axis("off")

        # Add architecture info text
        info_text = f"""Architecture Details:
Input Dimension: {input_dim}
Hidden Layers: {hidden_dims}
Latent Dimension: {latent_dim}
Total Parameters: {arch_info.get('total_parameters', 'N/A')}
Model Type: Autoencoder"""

        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()

        output_path = output_dir / "architecture_diagram.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def _create_performance_summary_plot(
        self, metrics: TrainingMetrics, output_dir: Path
    ) -> Path:
        """Create performance summary visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Performance radar chart
        categories = [
            "Accuracy",
            "Stability",
            "Efficiency",
            "Convergence",
            "Generalization",
        ]
        values = [
            metrics.validation_accuracy_estimate,
            max(0, 1 - metrics.loss_variance),
            min(1, 100 / max(metrics.training_time_seconds, 1)),  # Efficiency score
            1.0 if metrics.convergence_epoch else 0.5,
            max(0, 1 - abs(metrics.generalization_gap)),
        ]

        # Normalize values to 0-1 range
        values = [min(max(v, 0), 1) for v in values]

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]

        ax1.plot(angles, values, "o-", linewidth=2, color="blue")
        ax1.fill(angles, values, alpha=0.25, color="blue")
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories)
        ax1.set_ylim(0, 1)
        ax1.set_title("Performance Radar Chart")
        ax1.grid(True)

        # 2. Training efficiency
        efficiency_data = {
            "Time per Epoch": metrics.training_time_seconds
            / max(metrics.total_epochs, 1),
            "Loss Reduction": (metrics.train_loss_mean - metrics.best_val_loss)
            / metrics.train_loss_mean,
            "Convergence Speed": (metrics.convergence_epoch or metrics.total_epochs)
            / metrics.total_epochs,
        }

        ax2.bar(efficiency_data.keys(), efficiency_data.values(), color="green")
        ax2.set_title("Training Efficiency Metrics")
        ax2.set_ylabel("Value")
        ax2.tick_params(axis="x", rotation=45)

        # 3. Quality indicators
        quality_scores = {
            "Final Performance": 1 - metrics.final_val_loss,
            "Best Performance": 1 - metrics.best_val_loss,
            "Consistency": 1 - metrics.val_loss_std,
            "Robustness": metrics.validation_accuracy_estimate,
        }

        colors = [
            "red" if v < 0.5 else "yellow" if v < 0.8 else "green"
            for v in quality_scores.values()
        ]
        ax3.barh(
            list(quality_scores.keys()), list(quality_scores.values()), color=colors
        )
        ax3.set_title("Quality Indicators")
        ax3.set_xlabel("Score")
        ax3.set_xlim(0, 1)

        # 4. Training summary metrics
        summary_metrics = {
            "Total Epochs": metrics.total_epochs,
            "Best Epoch": metrics.best_epoch,
            "Early Stop": 1 if metrics.early_stopping_triggered else 0,
            "Overfitting": 1 if metrics.overfitting_detected else 0,
        }

        ax4.bar(summary_metrics.keys(), summary_metrics.values(), color="purple")
        ax4.set_title("Training Summary")
        ax4.set_ylabel("Count/Flag")
        ax4.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        output_path = output_dir / "performance_summary.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def _generate_summary_statistics(
        self, metrics: TrainingMetrics, architecture: ModelArchitectureInfo
    ) -> Dict[str, Any]:
        """Generate summary statistics for the training report."""
        return {
            "overall_performance_score": self._calculate_overall_score(metrics),
            "training_efficiency": {
                "time_per_epoch": metrics.training_time_seconds
                / max(metrics.total_epochs, 1),
                "epochs_to_convergence": metrics.convergence_epoch
                or metrics.total_epochs,
                "early_stopping_benefit": metrics.early_stopping_triggered,
            },
            "model_quality": {
                "final_loss": metrics.final_val_loss,
                "best_loss": metrics.best_val_loss,
                "loss_improvement": (metrics.val_loss_mean - metrics.best_val_loss)
                / metrics.val_loss_mean,
                "stability_score": max(0, 1 - metrics.loss_variance),
            },
            "anomaly_detection_readiness": {
                "threshold": metrics.anomaly_detection_threshold,
                "estimated_accuracy": metrics.validation_accuracy_estimate,
                "reconstruction_quality": 1
                - metrics.reconstruction_error_stats["mean"],
            },
            "architecture_efficiency": {
                "parameters_per_dimension": architecture.total_parameters
                / max(architecture.input_dimension, 1),
                "compression_ratio": architecture.input_dimension
                / max(architecture.latent_dimension, 1),
                "model_complexity": len(architecture.hidden_dimensions),
            },
        }

    def _calculate_overall_score(self, metrics: TrainingMetrics) -> float:
        """Calculate overall performance score (0-1)."""
        # Weighted combination of different performance aspects
        accuracy_score = metrics.validation_accuracy_estimate
        stability_score = max(0, 1 - metrics.loss_variance)
        convergence_score = 1.0 if metrics.convergence_epoch else 0.5
        generalization_score = max(0, 1 - abs(metrics.generalization_gap))

        # Weighted average
        overall_score = (
            0.4 * accuracy_score
            + 0.2 * stability_score
            + 0.2 * convergence_score
            + 0.2 * generalization_score
        )

        return min(max(overall_score, 0), 1)

    def _analyze_model_performance(
        self, metrics: TrainingMetrics, training_result: Dict
    ) -> Dict[str, Any]:
        """Analyze model performance and provide insights."""
        analysis = {
            "overall_score": self._calculate_overall_score(metrics),
            "strengths": [],
            "weaknesses": [],
            "performance_category": "",
            "key_insights": [],
        }

        # Analyze strengths
        if metrics.validation_accuracy_estimate > 0.8:
            analysis["strengths"].append("High validation accuracy")

        if (
            metrics.convergence_epoch
            and metrics.convergence_epoch < metrics.total_epochs * 0.7
        ):
            analysis["strengths"].append("Fast convergence")

        if not metrics.overfitting_detected:
            analysis["strengths"].append("No overfitting detected")

        if metrics.loss_variance < 0.01:
            analysis["strengths"].append("Stable training")

        # Analyze weaknesses
        if metrics.generalization_gap > 0.1:
            analysis["weaknesses"].append("Large generalization gap")

        if metrics.overfitting_detected:
            analysis["weaknesses"].append("Overfitting detected")

        if not metrics.convergence_epoch:
            analysis["weaknesses"].append("No clear convergence")

        if metrics.validation_accuracy_estimate < 0.6:
            analysis["weaknesses"].append("Low validation accuracy")

        # Determine performance category
        overall_score = analysis["overall_score"]
        if overall_score >= 0.8:
            analysis["performance_category"] = "Excellent"
        elif overall_score >= 0.6:
            analysis["performance_category"] = "Good"
        elif overall_score >= 0.4:
            analysis["performance_category"] = "Fair"
        else:
            analysis["performance_category"] = "Poor"

        # Generate key insights
        analysis["key_insights"] = [
            f"Model achieved {metrics.validation_accuracy_estimate:.1%} estimated accuracy",
            f"Training completed in {metrics.training_time_seconds:.1f} seconds",
            f"Best performance at epoch {metrics.best_epoch} of {metrics.total_epochs}",
            f"Anomaly detection threshold set at {metrics.anomaly_detection_threshold:.6f}",
        ]

        return analysis

    def _generate_training_recommendations(
        self, metrics: TrainingMetrics, training_result: Dict
    ) -> List[str]:
        """Generate recommendations for improving training."""
        recommendations = []

        # Performance-based recommendations
        if metrics.validation_accuracy_estimate < 0.7:
            recommendations.append(
                "Consider increasing model capacity or training time"
            )

        if metrics.overfitting_detected:
            recommendations.append("Add regularization or reduce model complexity")

        if not metrics.convergence_epoch:
            recommendations.append("Increase training epochs or adjust learning rate")

        if metrics.generalization_gap > 0.1:
            recommendations.append("Improve data quality or add data augmentation")

        if metrics.loss_variance > 0.05:
            recommendations.append(
                "Use learning rate scheduling or reduce learning rate"
            )

        # Training efficiency recommendations
        if metrics.training_time_seconds > 3600:  # More than 1 hour
            recommendations.append(
                "Consider using GPU acceleration or reducing batch size"
            )

        if (
            metrics.early_stopping_triggered
            and metrics.best_epoch < metrics.total_epochs * 0.3
        ):
            recommendations.append(
                "Reduce early stopping patience or adjust learning rate"
            )

        # Data-specific recommendations
        reconstruction_mean = metrics.reconstruction_error_stats.get("mean", 0)
        if reconstruction_mean > 0.1:
            recommendations.append("Review data preprocessing and normalization")

        if not recommendations:
            recommendations.append("Training performance is satisfactory")

        return recommendations

    def _save_report_files(
        self, report_content: Dict, output_dir: Path
    ) -> Dict[str, str]:
        """Save report content to various file formats."""
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        report_paths = {}

        # Save JSON report
        json_path = output_dir / "training_report.json"
        with open(json_path, "w") as f:
            json.dump(report_content, f, indent=2, default=str)
        report_paths["json"] = str(json_path)

        # Save summary text report
        text_path = output_dir / "training_summary.txt"
        with open(text_path, "w") as f:
            f.write(self._generate_text_summary(report_content))
        report_paths["text"] = str(text_path)

        # Save metrics CSV
        csv_path = output_dir / "training_metrics.csv"
        self._save_metrics_csv(report_content, csv_path)
        report_paths["csv"] = str(csv_path)

        return report_paths

    def _generate_text_summary(self, report_content: Dict) -> str:
        """Generate human-readable text summary."""
        metadata = report_content["report_metadata"]
        metrics = report_content["training_metrics"]
        analysis = report_content["performance_analysis"]

        summary = f"""
TRAINING REPORT SUMMARY
=======================

Job ID: {metadata['training_job_id']}
Generated: {metadata['generation_timestamp']}
Performance Category: {analysis['performance_category']}
Overall Score: {analysis['overall_score']:.3f}

TRAINING RESULTS
----------------
Total Epochs: {metrics['total_epochs']}
Best Epoch: {metrics['best_epoch']}
Training Time: {metrics['training_time_seconds']:.1f} seconds

Final Training Loss: {metrics['final_train_loss']:.6f}
Final Validation Loss: {metrics['final_val_loss']:.6f}
Best Validation Loss: {metrics['best_val_loss']:.6f}

PERFORMANCE INDICATORS
----------------------
Validation Accuracy: {metrics['validation_accuracy_estimate']:.1%}
Anomaly Threshold: {metrics['anomaly_detection_threshold']:.6f}
Generalization Gap: {metrics['generalization_gap']:.6f}
Early Stopping: {'Yes' if metrics['early_stopping_triggered'] else 'No'}
Overfitting: {'Detected' if metrics['overfitting_detected'] else 'Not Detected'}

STRENGTHS
---------
"""

        for strength in analysis["strengths"]:
            summary += f"• {strength}\n"

        summary += "\nWEAKNESSES\n----------\n"
        for weakness in analysis["weaknesses"]:
            summary += f"• {weakness}\n"

        summary += "\nRECOMMENDATIONS\n---------------\n"
        for rec in report_content["recommendations"]:
            summary += f"• {rec}\n"

        return summary

    def _save_metrics_csv(self, report_content: Dict, csv_path: Path):
        """Save metrics as CSV file."""
        metrics = report_content["training_metrics"]

        # Flatten metrics dictionary
        flattened_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flattened_metrics[f"{key}_{subkey}"] = subvalue
            else:
                flattened_metrics[key] = value

        # Create DataFrame and save
        df = pd.DataFrame([flattened_metrics])
        df.to_csv(csv_path, index=False)

    def _update_database_record(
        self,
        training_job_id: int,
        report_content: Dict,
        report_paths: Dict,
        db: Session,
    ):
        """Update database with training report information."""
        try:
            # Check if report already exists
            existing_report = (
                db.query(TrainingReport)
                .filter(TrainingReport.training_job_id == training_job_id)
                .first()
            )

            metrics = report_content["training_metrics"]

            if existing_report:
                # Update existing report
                existing_report.final_loss = metrics["final_val_loss"]
                existing_report.validation_accuracy = metrics[
                    "validation_accuracy_estimate"
                ]
                existing_report.best_epoch = metrics["best_epoch"]
                existing_report.training_time_seconds = metrics["training_time_seconds"]
                existing_report.metrics_summary = json.dumps(metrics)
                existing_report.report_file_path = report_paths.get("json", "")
            else:
                # Create new report
                training_report = TrainingReport(
                    training_job_id=training_job_id,
                    final_loss=metrics["final_val_loss"],
                    validation_accuracy=metrics["validation_accuracy_estimate"],
                    best_epoch=metrics["best_epoch"],
                    training_time_seconds=metrics["training_time_seconds"],
                    model_parameters_path="",  # Would be set by training service
                    metrics_summary=json.dumps(metrics),
                    report_file_path=report_paths.get("json", ""),
                )
                db.add(training_report)

            db.commit()
            logger.info(f"Updated database record for training job {training_job_id}")

        except Exception as e:
            logger.error(f"Failed to update database record: {str(e)}")
            db.rollback()


# Global service instance
_training_report_generator = None


def get_training_report_generator() -> TrainingReportGenerator:
    """Get the global training report generator instance."""
    global _training_report_generator
    if _training_report_generator is None:
        _training_report_generator = TrainingReportGenerator()
    return _training_report_generator
