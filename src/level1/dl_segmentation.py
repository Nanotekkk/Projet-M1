"""Deep Learning Segmentation for Ground Detection - Level 1"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from typing import Optional
from src.core.segmentation_strategy import SegmentationStrategy, SegmentationResult


class DLSegmentation(SegmentationStrategy):
    """
    Deep Learning-based ground detection using scikit-learn's MLPClassifier.

    This module provides an alternative to RANSAC using a simple feed-forward
    neural network trained to classify ground vs. non-ground points.
    """

    def __init__(
        self,
        model: Optional[MLPClassifier] = None,
        confidence_threshold: float = 0.5,
        batch_size: int = 1000,
    ):
        """
        Initialize DL segmentation.

        Args:
            model: Pre-trained sklearn MLPClassifier. If None, creates a new one.
            confidence_threshold: Classification confidence threshold
            batch_size: Batch size for inference
        """
        if model is None:
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 64),
                activation='relu',
                random_state=42,
                max_iter=200,
                early_stopping=False
            )
        else:
            self.model = model
        
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.is_trained = False

    def segment(self, points: np.ndarray) -> SegmentationResult:
        """
        Segment ground points using deep learning.

        Args:
            points: Input point cloud (N, 3)

        Returns:
            SegmentationResult with ground point indices
        """
        if not self.is_trained:
            # If not trained, use a simple heuristic based on height
            return self._heuristic_segment(points)
        
        # Normalize points
        points_normalized = self.scaler.transform(points)

        # Predict probabilities
        predictions = self.model.predict_proba(points_normalized)
        confidence = np.max(predictions, axis=1)
        labels = self.model.predict(points_normalized)
        
        # Filter by confidence threshold
        ground_indices = np.where((labels == 1) & (confidence > self.confidence_threshold))[0]

        result_labels = np.zeros(len(points), dtype=int)
        result_labels[ground_indices] = 1

        return SegmentationResult(
            labels=result_labels,
            ground_indices=ground_indices,
            metadata={
                "method": "Deep Learning (MLP)",
                "num_inliers": len(ground_indices),
                "num_outliers": len(points) - len(ground_indices),
                "confidence_threshold": self.confidence_threshold,
                "model_type": "MLPClassifier"
            },
        )

    def _heuristic_segment(self, points: np.ndarray) -> SegmentationResult:
        """Use height-based heuristic when model is not trained"""
        # Assume ground points are those with lower z-values
        z_threshold = np.percentile(points[:, 2], 30)
        ground_indices = np.where(points[:, 2] <= z_threshold)[0]
        
        labels = np.zeros(len(points), dtype=int)
        labels[ground_indices] = 1
        
        return SegmentationResult(
            labels=labels,
            ground_indices=ground_indices,
            metadata={
                "method": "Height Heuristic",
                "num_inliers": len(ground_indices),
                "num_outliers": len(points) - len(ground_indices),
                "z_threshold": float(z_threshold),
            },
        )

    def train_model(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        epochs: int = 10,
    ) -> None:
        """
        Train the neural network on labeled data.

        Args:
            points: Training point cloud (N, 3)
            labels: Training labels (N,) binary (0=non-ground, 1=ground)
            epochs: Number of training epochs (ignored for sklearn)
        """
        # Fit scaler
        self.scaler.fit(points)
        points_normalized = self.scaler.transform(points)
        
        # Train model
        self.model.fit(points_normalized, labels)
        self.is_trained = True

    def get_name(self) -> str:
        return "Deep Learning Ground Detection (MLP)"
