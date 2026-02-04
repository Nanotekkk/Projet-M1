"""Teleportation Validator - Level 1"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class TeleportationPoint:
    """Represents a valid teleportation location"""

    position: np.ndarray
    is_valid: bool
    reason: str = ""
    distance_to_ground: float = 0.0


class TeleportationValidator:
    """
    Validates teleportation positions in VR environments.

    Ensures that teleportation points are on valid ground surfaces
    and respects safety constraints.
    """

    def __init__(
        self,
        min_ground_points: int = 10,
        max_height_above_ground: float = 0.5,
        min_surface_area: float = 1.0,
    ):
        """
        Initialize teleportation validator.

        Args:
            min_ground_points: Minimum ground points in validation radius
            max_height_above_ground: Max height above ground for valid teleport
            min_surface_area: Minimum surface area for valid landing zone
        """
        self.min_ground_points = min_ground_points
        self.max_height_above_ground = max_height_above_ground
        self.min_surface_area = min_surface_area

    def validate_position(
        self, position: np.ndarray, ground_points: np.ndarray, radius: float = 2.0
    ) -> TeleportationPoint:
        """
        Validate a teleportation position.

        Args:
            position: Proposed teleportation position (3,)
            ground_points: Ground point cloud (N, 3)
            radius: Search radius around position

        Returns:
            TeleportationPoint with validation result
        """
        # Check if position is above ground
        distances = np.linalg.norm(ground_points - position, axis=1)
        nearby_points = ground_points[distances < radius]

        if len(nearby_points) < self.min_ground_points:
            return TeleportationPoint(
                position=position,
                is_valid=False,
                reason="Insufficient ground points nearby",
            )

        # Find height above ground
        min_distance = distances.min()
        if min_distance > self.max_height_above_ground:
            return TeleportationPoint(
                position=position,
                is_valid=False,
                reason=f"Height above ground ({min_distance:.2f}m) exceeds limit",
                distance_to_ground=min_distance,
            )

        # Check surface flatness
        if len(nearby_points) > 3:
            flatness = self._check_surface_flatness(nearby_points)
            if not flatness:
                return TeleportationPoint(
                    position=position,
                    is_valid=False,
                    reason="Ground surface is not flat enough",
                )

        return TeleportationPoint(
            position=position,
            is_valid=True,
            reason="Valid teleportation point",
            distance_to_ground=min_distance,
        )

    def validate_multiple_positions(
        self, positions: np.ndarray, ground_points: np.ndarray, radius: float = 2.0
    ) -> list:
        """
        Validate multiple teleportation positions.

        Args:
            positions: Multiple proposed positions (M, 3)
            ground_points: Ground point cloud (N, 3)
            radius: Search radius around position

        Returns:
            List of TeleportationPoint results
        """
        return [
            self.validate_position(pos, ground_points, radius) for pos in positions
        ]

    @staticmethod
    def _check_surface_flatness(points: np.ndarray) -> bool:
        """
        Check if a surface is reasonably flat.

        Args:
            points: Points on surface (N, 3)

        Returns:
            True if surface is flat enough
        """
        # Fit plane and check standard deviation of distances
        centroid = points.mean(axis=0)
        centered = points - centroid

        _, _, Vt = np.linalg.svd(centered)
        normal = Vt[-1, :]

        distances = np.abs(np.dot(centered, normal))
        flatness_metric = distances.std()

        # Surface is considered flat if std of distances is small
        return flatness_metric < 0.5

    def get_valid_landing_zone(
        self, ground_points: np.ndarray, grid_step: float = 1.0
    ) -> np.ndarray:
        """
        Get all valid landing zones in ground point cloud.

        Args:
            ground_points: Ground point cloud (N, 3)
            grid_step: Step size for grid sampling

        Returns:
            Array of valid teleportation positions
        """
        min_bounds = ground_points.min(axis=0)
        max_bounds = ground_points.max(axis=0)

        # Create grid
        x = np.arange(min_bounds[0], max_bounds[0], grid_step)
        y = np.arange(min_bounds[1], max_bounds[1], grid_step)
        z = np.arange(min_bounds[2], max_bounds[2], grid_step)

        valid_positions = []
        for xi in x:
            for yi in y:
                for zi in z:
                    position = np.array([xi, yi, zi])
                    result = self.validate_position(position, ground_points)
                    if result.is_valid:
                        valid_positions.append(position)

        return np.array(valid_positions) if valid_positions else np.array([]).reshape(0, 3)
