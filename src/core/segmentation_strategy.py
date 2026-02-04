"""Segmentation Strategy - Strategy Pattern"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any


class SegmentationResult:
    """Result of segmentation operation"""

    def __init__(
        self,
        labels: np.ndarray,
        ground_indices: np.ndarray,
        metadata: Dict[str, Any] = None,
    ):
        self.labels = labels
        self.ground_indices = ground_indices
        self.metadata = metadata or {}

    def get_ground_points(self, points: np.ndarray) -> np.ndarray:
        """Get ground points from original point cloud"""
        return points[self.ground_indices]

    def get_non_ground_points(self, points: np.ndarray) -> np.ndarray:
        """Get non-ground points from original point cloud"""
        non_ground_indices = np.setdiff1d(
            np.arange(len(points)), self.ground_indices
        )
        return points[non_ground_indices]


class SegmentationStrategy(ABC):
    """Abstract base class for segmentation strategies"""

    @abstractmethod
    def segment(self, points: np.ndarray) -> SegmentationResult:
        """Segment point cloud into ground and non-ground points"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name"""
        pass
