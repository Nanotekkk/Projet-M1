"""3D Occupancy Grid for Navigation - Level 2

Author: Matheo LANCEA
Description: Spatial grid representation for obstacle avoidance
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class GridCell:
    """Represents a cell in the occupancy grid"""

    occupied: bool
    confidence: float  # Confidence that cell is occupied (0-1)
    cost: float  # Navigation cost for path planning


class OccupancyGrid3D:
    """
    3D occupancy grid representing navigable and obstacle space.

    Divides the environment into a regular 3D grid of cells,
    marking each as navigable or occupied by obstacles.
    """

    def __init__(
        self,
        min_bounds: np.ndarray,
        max_bounds: np.ndarray,
        cell_size: float = 0.5,
    ):
        """
        Initialize 3D occupancy grid.

        Args:
            min_bounds: Minimum coordinates of grid (3,)
            max_bounds: Maximum coordinates of grid (3,)
            cell_size: Size of each grid cell
        """
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
        self.cell_size = cell_size

        # Calculate grid dimensions
        self.grid_shape = np.ceil(
            (max_bounds - min_bounds) / cell_size
        ).astype(int)

        # Initialize grid: False = free, True = occupied
        self.grid = np.zeros(self.grid_shape, dtype=bool)
        self.confidence_grid = np.zeros(self.grid_shape, dtype=float)

    def world_to_grid_index(self, world_coords: np.ndarray) -> np.ndarray:
        """
        Convert world coordinates to grid indices.

        Args:
            world_coords: World coordinates (3,) or (N, 3)

        Returns:
            Grid indices as integers
        """
        grid_indices = (world_coords - self.min_bounds) / self.cell_size
        grid_indices = np.floor(grid_indices).astype(int)

        # Clip to grid bounds
        grid_indices = np.clip(grid_indices, 0, np.array(self.grid_shape) - 1)

        return grid_indices

    def grid_to_world_coords(self, grid_indices: np.ndarray) -> np.ndarray:
        """
        Convert grid indices to world coordinates.

        Args:
            grid_indices: Grid indices (3,) or (N, 3)

        Returns:
            World coordinates
        """
        return self.min_bounds + grid_indices * self.cell_size + self.cell_size / 2

    def mark_occupied(self, world_coords: np.ndarray, radius: float = 0.0) -> None:
        """
        Mark cells as occupied.

        Args:
            world_coords: World coordinates to mark (N, 3)
            radius: Expansion radius for occupation
        """
        grid_indices = self.world_to_grid_index(world_coords)

        if len(grid_indices.shape) == 1:
            grid_indices = grid_indices.reshape(1, -1)

        for idx in grid_indices:
            if self._in_bounds(idx):
                self.grid[tuple(idx)] = True
                self.confidence_grid[tuple(idx)] = 1.0

                # Expand occupation around point
                if radius > 0:
                    expansion_cells = int(radius / self.cell_size)
                    self._expand_occupation(idx, expansion_cells)

    def mark_free(self, world_coords: np.ndarray) -> None:
        """
        Mark cells as free space.

        Args:
            world_coords: World coordinates to mark (N, 3)
        """
        grid_indices = self.world_to_grid_index(world_coords)

        if len(grid_indices.shape) == 1:
            grid_indices = grid_indices.reshape(1, -1)

        for idx in grid_indices:
            if self._in_bounds(idx):
                self.grid[tuple(idx)] = False
                self.confidence_grid[tuple(idx)] = 0.0

    def is_navigable(self, world_coords: np.ndarray) -> bool:
        """
        Check if a location is navigable.

        Args:
            world_coords: World coordinate to check (3,)

        Returns:
            True if location is free of obstacles
        """
        grid_idx = self.world_to_grid_index(world_coords)
        if not self._in_bounds(grid_idx):
            return False
        return not self.grid[tuple(grid_idx)]

    def get_neighbors(
        self, grid_index: np.ndarray, include_diagonals: bool = False
    ) -> list:
        """
        Get neighboring grid cells.

        Args:
            grid_index: Current grid index (3,)
            include_diagonals: Include diagonal neighbors

        Returns:
            List of neighboring grid indices
        """
        neighbors = []
        deltas = [
            [-1, 0, 0],
            [1, 0, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, 1],
        ]

        if include_diagonals:
            # Add diagonal neighbors (26-connected)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx != 0 or dy != 0 or dz != 0:
                            deltas.append([dx, dy, dz])

        for delta in deltas:
            neighbor = grid_index + np.array(delta)
            if self._in_bounds(neighbor):
                neighbors.append(tuple(neighbor))

        return neighbors

    def get_free_cells(self) -> np.ndarray:
        """
        Get all free cells in the grid.

        Returns:
            Coordinates of free cells (N, 3)
        """
        free_cells = np.where(~self.grid)
        return np.column_stack(free_cells)

    def get_occupied_cells(self) -> np.ndarray:
        """
        Get all occupied cells in the grid.

        Returns:
            Coordinates of occupied cells (N, 3)
        """
        occupied_cells = np.where(self.grid)
        return np.column_stack(occupied_cells)

    def _in_bounds(self, grid_index: np.ndarray) -> bool:
        """Check if grid index is within bounds"""
        return np.all(grid_index >= 0) and np.all(grid_index < self.grid_shape)

    def _expand_occupation(self, center_idx: np.ndarray, radius: int) -> None:
        """Expand occupation around a center point"""
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    idx = center_idx + np.array([dx, dy, dz])
                    if self._in_bounds(idx):
                        self.grid[tuple(idx)] = True
                        self.confidence_grid[tuple(idx)] = max(
                            self.confidence_grid[tuple(idx)],
                            1.0 - (abs(dx) + abs(dy) + abs(dz)) / (radius + 1),
                        )
