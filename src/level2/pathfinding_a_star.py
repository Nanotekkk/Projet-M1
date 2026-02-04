"""A* Pathfinding Algorithm - Level 2"""

import numpy as np
import heapq
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass(order=True)
class PathNode:
    """Node for A* pathfinding"""

    f_cost: float
    position: Tuple = field(compare=False)
    g_cost: float = field(compare=False)
    parent: Optional["PathNode"] = field(default=None, compare=False)

    def __hash__(self):
        return hash(self.position)


class AStarPathfinder:
    """
    A* pathfinding algorithm for navigation in 3D space.

    Finds optimal paths considering obstacles and navigation costs.
    """

    def __init__(self, diagonal_movement: bool = True):
        """
        Initialize A* pathfinder.

        Args:
            diagonal_movement: Allow diagonal movement between cells
        """
        self.diagonal_movement = diagonal_movement

    def find_path(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        occupancy_grid,
        include_diagonals: bool = True,
    ) -> Optional[List[np.ndarray]]:
        """
        Find path from start to goal using A*.

        Args:
            start: Start position (3,)
            goal: Goal position (3,)
            occupancy_grid: OccupancyGrid3D instance
            include_diagonals: Allow diagonal movement

        Returns:
            List of waypoints or None if no path found
        """
        start_idx = tuple(occupancy_grid.world_to_grid_index(start))
        goal_idx = tuple(occupancy_grid.world_to_grid_index(goal))

        if occupancy_grid.grid[goal_idx]:
            return None  # Goal is occupied

        # Open and closed sets
        open_set = []
        closed_set = set()
        g_costs = {start_idx: 0.0}

        start_h = self._heuristic(start_idx, goal_idx)
        start_node = PathNode(
            f_cost=start_h,
            position=start_idx,
            g_cost=0.0,
        )

        heapq.heappush(open_set, start_node)

        while open_set:
            current = heapq.heappop(open_set)

            if current.position == goal_idx:
                return self._reconstruct_path(current, occupancy_grid)

            if current.position in closed_set:
                continue

            closed_set.add(current.position)

            # Get neighbors
            neighbors = occupancy_grid.get_neighbors(
                np.array(current.position), include_diagonals
            )

            for neighbor_pos in neighbors:
                if neighbor_pos in closed_set:
                    continue

                if occupancy_grid.grid[neighbor_pos]:
                    continue  # Obstacle

                # Calculate costs
                g_cost = g_costs[current.position] + self._movement_cost(
                    current.position, neighbor_pos
                )

                if neighbor_pos not in g_costs or g_cost < g_costs[neighbor_pos]:
                    g_costs[neighbor_pos] = g_cost
                    h_cost = self._heuristic(neighbor_pos, goal_idx)
                    f_cost = g_cost + h_cost

                    neighbor_node = PathNode(
                        f_cost=f_cost,
                        position=neighbor_pos,
                        g_cost=g_cost,
                        parent=current,
                    )

                    heapq.heappush(open_set, neighbor_node)

        return None  # No path found

    def find_path_with_constraints(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        occupancy_grid,
        forbidden_zones: List[np.ndarray] = None,
        prefer_zones: List[np.ndarray] = None,
    ) -> Optional[List[np.ndarray]]:
        """
        Find path with custom constraints and preferences.

        Args:
            start: Start position (3,)
            goal: Goal position (3,)
            occupancy_grid: OccupancyGrid3D instance
            forbidden_zones: Zones to avoid (list of min/max bounds)
            prefer_zones: Zones to prefer (affects cost)

        Returns:
            List of waypoints or None if no path found
        """
        # Similar to find_path but with custom cost function
        start_idx = tuple(occupancy_grid.world_to_grid_index(start))
        goal_idx = tuple(occupancy_grid.world_to_grid_index(goal))

        if occupancy_grid.grid[goal_idx]:
            return None

        open_set = []
        closed_set = set()
        g_costs = {start_idx: 0.0}

        start_h = self._heuristic(start_idx, goal_idx)
        start_node = PathNode(
            f_cost=start_h,
            position=start_idx,
            g_cost=0.0,
        )

        heapq.heappush(open_set, start_node)

        while open_set:
            current = heapq.heappop(open_set)

            if current.position == goal_idx:
                return self._reconstruct_path(current, occupancy_grid)

            if current.position in closed_set:
                continue

            closed_set.add(current.position)

            neighbors = occupancy_grid.get_neighbors(
                np.array(current.position), self.diagonal_movement
            )

            for neighbor_pos in neighbors:
                if neighbor_pos in closed_set:
                    continue

                if occupancy_grid.grid[neighbor_pos]:
                    continue

                # Check forbidden zones
                world_pos = occupancy_grid.grid_to_world_coords(np.array(neighbor_pos))
                if forbidden_zones:
                    if any(self._in_zone(world_pos, zone) for zone in forbidden_zones):
                        continue

                # Calculate cost with preferences
                movement_cost = self._movement_cost(current.position, neighbor_pos)

                if prefer_zones:
                    for zone in prefer_zones:
                        if self._in_zone(world_pos, zone):
                            movement_cost *= 0.5  # Reduce cost in preferred zones

                g_cost = g_costs[current.position] + movement_cost

                if neighbor_pos not in g_costs or g_cost < g_costs[neighbor_pos]:
                    g_costs[neighbor_pos] = g_cost
                    h_cost = self._heuristic(neighbor_pos, goal_idx)
                    f_cost = g_cost + h_cost

                    neighbor_node = PathNode(
                        f_cost=f_cost,
                        position=neighbor_pos,
                        g_cost=g_cost,
                        parent=current,
                    )

                    heapq.heappush(open_set, neighbor_node)

        return None

    @staticmethod
    def _heuristic(pos1: Tuple, pos2: Tuple) -> float:
        """Euclidean heuristic"""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    @staticmethod
    def _movement_cost(pos1: Tuple, pos2: Tuple) -> float:
        """Calculate movement cost between two cells"""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    @staticmethod
    def _reconstruct_path(node: PathNode, occupancy_grid) -> List[np.ndarray]:
        """Reconstruct path from goal to start"""
        path_indices = []
        current = node

        while current is not None:
            path_indices.append(current.position)
            current = current.parent

        path_indices.reverse()

        # Convert to world coordinates
        path = [occupancy_grid.grid_to_world_coords(np.array(idx)) for idx in path_indices]

        return path

    @staticmethod
    def _in_zone(point: np.ndarray, zone: Tuple[np.ndarray, np.ndarray]) -> bool:
        """Check if point is in zone"""
        min_bound, max_bound = zone
        return np.all(point >= min_bound) and np.all(point <= max_bound)
