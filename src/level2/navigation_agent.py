"""Navigation Agent - Level 2

Author: Matheo LANCEA
Description: Intelligent agent managing VR navigation with observer pattern
"""

import numpy as np
from typing import List, Optional
from src.core.observer import Observable, Observer
from src.level2.pathfinding_a_star import AStarPathfinder
from src.level2.occupancy_grid_3d import OccupancyGrid3D


class NavigationEvent:
    """Event data for navigation updates"""

    def __init__(self, event_type: str, data=None):
        self.event_type = event_type
        self.data = data


class NavigationAgent(Observable):
    """
    Intelligent navigation agent for VR environments.

    Manages pathfinding, path following, and dynamic recalculation
    for immersive navigation experiences.

    Design Pattern: Observer for event notifications, Singleton for agent state.
    """

    def __init__(self, occupancy_grid: OccupancyGrid3D):
        """
        Initialize navigation agent.

        Args:
            occupancy_grid: OccupancyGrid3D instance
        """
        super().__init__()
        self.occupancy_grid = occupancy_grid
        self.pathfinder = AStarPathfinder()

        self.current_position = None
        self.goal_position = None
        self.current_path = None
        self.current_waypoint_idx = 0

        self.path_deviation_threshold = 1.0  # Meters
        self.recalculation_enabled = True

    def set_goal(self, goal_position: np.ndarray) -> bool:
        """
        Set navigation goal and compute path.

        Args:
            goal_position: Target position (3,)

        Returns:
            True if path found, False otherwise
        """
        if not self.occupancy_grid.is_navigable(goal_position):
            self.notify("goal_not_navigable", goal_position)
            return False

        self.goal_position = goal_position

        if self.current_position is not None:
            path = self.pathfinder.find_path(
                self.current_position, goal_position, self.occupancy_grid
            )

            if path is not None:
                self.current_path = path
                self.current_waypoint_idx = 0
                self.notify("path_computed", {"path": path, "length": len(path)})
                return True
            else:
                self.notify("no_path_found", goal_position)
                return False

        return True

    def update_position(self, new_position: np.ndarray) -> None:
        """
        Update agent position and check for path deviation.

        Args:
            new_position: New position (3,)
        """
        self.current_position = new_position

        if self.current_path is None:
            return

        # Check deviation from path
        if self._is_off_path(new_position):
            self.notify("path_deviation", new_position)

            if self.recalculation_enabled and self.goal_position is not None:
                self.set_goal(self.goal_position)

        # Update next waypoint
        self._update_current_waypoint(new_position)

    def get_next_waypoint(self) -> Optional[np.ndarray]:
        """Get next waypoint in path"""
        if self.current_path is None:
            return None

        if self.current_waypoint_idx < len(self.current_path):
            return self.current_path[self.current_waypoint_idx]

        return None

    def get_path_following_direction(self) -> Optional[np.ndarray]:
        """
        Get direction to follow current path.

        Returns:
            Direction vector or None
        """
        if self.current_path is None or self.current_position is None:
            return None

        waypoint = self.get_next_waypoint()
        if waypoint is None:
            return None

        direction = waypoint - self.current_position
        norm = np.linalg.norm(direction)

        if norm > 0:
            return direction / norm

        return None

    def get_remaining_distance(self) -> float:
        """Get remaining distance to goal"""
        if self.goal_position is None or self.current_position is None:
            return float("inf")

        return np.linalg.norm(self.goal_position - self.current_position)

    def is_at_goal(self, tolerance: float = 0.5) -> bool:
        """Check if agent reached goal"""
        if self.goal_position is None or self.current_position is None:
            return False

        return np.linalg.norm(self.goal_position - self.current_position) < tolerance

    def _is_off_path(self, position: np.ndarray) -> bool:
        """Check if position deviates from path"""
        if self.current_path is None:
            return False

        # Find closest point on path
        path = np.array(self.current_path)
        distances = np.linalg.norm(path - position, axis=1)
        min_distance = distances.min()

        return min_distance > self.path_deviation_threshold

    def _update_current_waypoint(self, position: np.ndarray) -> None:
        """Update current waypoint based on position"""
        if self.current_path is None:
            return

        waypoint = self.get_next_waypoint()
        if waypoint is None:
            return

        distance = np.linalg.norm(waypoint - position)

        # If close to waypoint, move to next
        if distance < 0.5:  # 50cm threshold
            self.current_waypoint_idx += 1
            self.notify("waypoint_reached", {"waypoint": waypoint})

            if self.current_waypoint_idx >= len(self.current_path):
                self.notify("goal_reached", self.goal_position)
