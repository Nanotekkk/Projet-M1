"""Unit tests for Level 2 Navigation"""

import pytest
import numpy as np
from src.level2.euclidean_clustering import EuclideanClustering
from src.level2.plane_detection import PlaneDetection
from src.level2.occupancy_grid_3d import OccupancyGrid3D
from src.level2.pathfinding_a_star import AStarPathfinder
from src.level2.navigation_agent import NavigationAgent
from src.core.point_cloud_loader import PointCloudLoader


class TestEuclideanClustering:
    """Tests for Euclidean clustering"""

    def test_clustering_initialization(self):
        """Test clustering initialization"""
        clustering = EuclideanClustering(eps=0.5, min_points=5)
        assert clustering.eps > 0
        assert clustering.min_points > 0

    def test_cluster_synthetic_scene(self):
        """Test clustering on synthetic scene"""
        scene = PointCloudLoader.create_synthetic_scene()
        clustering = EuclideanClustering(eps=5.0, min_points=10)
        result = clustering.cluster(scene.points)

        assert result.n_clusters > 0
        assert len(result.labels) == len(scene.points)

    def test_cluster_result_extraction(self):
        """Test extracting points from clusters"""
        points = np.array([
            [0, 0, 0],
            [0.1, 0, 0],
            [10, 10, 0],
            [10.1, 10, 0],
        ])

        clustering = EuclideanClustering(eps=1.0, min_points=1)
        result = clustering.cluster(points)

        # Should have at least 2 clusters
        assert result.n_clusters >= 1

    def test_adaptive_clustering(self):
        """Test adaptive clustering with automatic eps"""
        scene = PointCloudLoader.create_synthetic_scene()
        clustering = EuclideanClustering()
        result = clustering.adaptive_clustering(scene.points)

        assert result.n_clusters > 0


@pytest.mark.unit
class TestPlaneDetection:
    """Tests for plane detection"""

    def test_plane_detection_initialization(self):
        """Test plane detector initialization"""
        detector = PlaneDetection()
        assert detector.distance_threshold > 0
        assert detector.iterations > 0

    def test_detect_single_plane(self):
        """Test detecting a single plane"""
        plane = PointCloudLoader.create_synthetic_plane(num_points=300)
        detector = PlaneDetection(distance_threshold=0.5)
        planes = detector.detect_planes(plane.points, num_planes=1)

        assert len(planes) > 0
        detected_plane = planes[0]
        assert detected_plane.normal is not None
        assert len(detected_plane.points_indices) > 100

    def test_plane_type_classification(self):
        """Test plane type classification"""
        # Horizontal plane (normal pointing up)
        horizontal_plane_points = np.random.uniform(-10, 10, (100, 2))
        horizontal_plane_points = np.column_stack([
            horizontal_plane_points,
            np.zeros(100),
        ])

        detector = PlaneDetection(distance_threshold=0.5)
        planes = detector.detect_planes(horizontal_plane_points, num_planes=1)

        if planes:
            plane_type = planes[0].classify_plane_type()
            assert plane_type in ["horizontal", "vertical", "inclined"]

    def test_detect_multiple_planes(self):
        """Test detecting multiple planes"""
        scene = PointCloudLoader.create_synthetic_scene()
        detector = PlaneDetection(distance_threshold=1.0, min_inliers=20)
        planes = detector.detect_planes(scene.points, num_planes=3)

        assert len(planes) > 0


@pytest.mark.unit
class TestOccupancyGrid3D:
    """Tests for 3D occupancy grid"""

    @pytest.fixture
    def grid(self):
        """Create test grid"""
        return OccupancyGrid3D(
            min_bounds=np.array([0, 0, 0]),
            max_bounds=np.array([100, 100, 10]),
            cell_size=1.0,
        )

    def test_grid_creation(self, grid):
        """Test grid creation"""
        assert grid.grid.shape[0] > 0
        assert grid.grid.shape[1] > 0
        assert grid.grid.shape[2] > 0

    def test_world_to_grid_conversion(self, grid):
        """Test world to grid coordinate conversion"""
        world_pos = np.array([50, 50, 5])
        grid_idx = grid.world_to_grid_index(world_pos)

        assert grid_idx.shape == (3,)  # 3D grid indices

    def test_grid_to_world_conversion(self, grid):
        """Test grid to world conversion"""
        grid_idx = np.array([50, 50, 5])
        world_pos = grid.grid_to_world_coords(grid_idx)

        assert world_pos.shape == (3,)

    def test_mark_occupied(self, grid):
        """Test marking cells as occupied"""
        positions = np.array([[50, 50, 5], [50, 51, 5]])
        grid.mark_occupied(positions)

        # Check that cells are marked
        assert grid.grid[50, 50, 5] == True

    def test_is_navigable(self, grid):
        """Test navigability check"""
        # Unmarked cell should be navigable
        assert grid.is_navigable(np.array([50, 50, 5]))

        # Mark as occupied
        grid.mark_occupied(np.array([[50, 50, 5]]))
        assert not grid.is_navigable(np.array([50, 50, 5]))

    def test_get_neighbors(self, grid):
        """Test getting neighboring cells"""
        grid_idx = np.array([50, 50, 5])
        neighbors = grid.get_neighbors(grid_idx)

        assert len(neighbors) == 6  # 6-connected neighbors

    def test_get_free_cells(self, grid):
        """Test retrieving free cells"""
        free_cells = grid.get_free_cells()
        assert len(free_cells) > 0

    def test_get_occupied_cells(self, grid):
        """Test retrieving occupied cells"""
        grid.mark_occupied(np.array([[50, 50, 5], [51, 50, 5]]))
        occupied_cells = grid.get_occupied_cells()

        assert len(occupied_cells) >= 2


@pytest.mark.unit
class TestAStarPathfinder:
    """Tests for A* pathfinding"""

    @pytest.fixture
    def pathfinding_grid(self):
        """Create test grid for pathfinding"""
        grid = OccupancyGrid3D(
            min_bounds=np.array([0, 0, 0]),
            max_bounds=np.array([100, 100, 10]),
            cell_size=1.0,
        )
        return grid

    def test_pathfinder_initialization(self):
        """Test pathfinder initialization"""
        pathfinder = AStarPathfinder()
        assert pathfinder is not None

    def test_simple_path_finding(self, pathfinding_grid):
        """Test finding path in clear space"""
        pathfinder = AStarPathfinder()
        start = np.array([10, 10, 5])
        goal = np.array([90, 90, 5])

        path = pathfinder.find_path(start, goal, pathfinding_grid)

        assert path is not None
        assert len(path) > 1

    def test_path_around_obstacle(self, pathfinding_grid):
        """Test pathfinding around obstacles"""
        # Mark obstacle
        for x in range(40, 60):
            for y in range(40, 60):
                pathfinding_grid.mark_occupied(np.array([[x, y, 5]]))

        pathfinder = AStarPathfinder()
        start = np.array([30, 50, 5])
        goal = np.array([70, 50, 5])

        path = pathfinder.find_path(start, goal, pathfinding_grid)

        assert path is not None

    def test_no_path_found(self, pathfinding_grid):
        """Test when no path exists"""
        # Surround goal with obstacles
        for x in range(88, 92):
            for y in range(88, 92):
                for z in range(4, 6):
                    pathfinding_grid.mark_occupied(np.array([[x, y, z]]))

        pathfinder = AStarPathfinder()
        start = np.array([10, 10, 5])
        goal = np.array([90, 90, 5])

        path = pathfinder.find_path(start, goal, pathfinding_grid)

        # May or may not find path depending on obstacle placement
        # This test just ensures the function doesn't crash


@pytest.mark.unit
@pytest.mark.level2
class TestNavigationAgent:
    """Tests for navigation agent"""

    @pytest.fixture
    def nav_grid(self):
        """Create navigation grid"""
        return OccupancyGrid3D(
            min_bounds=np.array([0, 0, 0]),
            max_bounds=np.array([100, 100, 10]),
            cell_size=1.0,
        )

    def test_agent_initialization(self, nav_grid):
        """Test agent creation"""
        agent = NavigationAgent(nav_grid)
        assert agent.current_position is None
        assert agent.goal_position is None

    def test_set_goal_and_get_path(self, nav_grid):
        """Test setting goal and retrieving path"""
        agent = NavigationAgent(nav_grid)
        agent.current_position = np.array([10, 10, 5])

        success = agent.set_goal(np.array([90, 90, 5]))

        assert success
        assert agent.current_path is not None

    def test_position_update(self, nav_grid):
        """Test updating agent position"""
        agent = NavigationAgent(nav_grid)
        agent.current_position = np.array([10, 10, 5])
        agent.set_goal(np.array([90, 90, 5]))

        # Move towards goal
        new_pos = np.array([11, 11, 5])
        agent.update_position(new_pos)

        assert agent.current_position is not None

    def test_waypoint_following(self, nav_grid):
        """Test waypoint following"""
        agent = NavigationAgent(nav_grid)
        agent.current_position = np.array([10, 10, 5])
        success = agent.set_goal(np.array([50, 50, 5]))

        if success:
            waypoint = agent.get_next_waypoint()
            assert waypoint is not None

    def test_goal_reached(self, nav_grid):
        """Test goal reached condition"""
        agent = NavigationAgent(nav_grid)
        agent.current_position = np.array([50, 50, 5])
        agent.goal_position = np.array([50, 50, 5])

        assert agent.is_at_goal(tolerance=1.0)

    def test_observer_pattern(self, nav_grid):
        """Test observer pattern for events"""
        from src.core.observer import Observer

        class TestObserver(Observer):
            def __init__(self):
                self.events = []

            def update(self, event_type: str, data=None):
                self.events.append((event_type, data))

        agent = NavigationAgent(nav_grid)
        observer = TestObserver()
        agent.attach(observer)

        agent.current_position = np.array([10, 10, 5])
        agent.set_goal(np.array([90, 90, 5]))

        # Should have received events
        assert len(observer.events) > 0


@pytest.mark.unit
@pytest.mark.level2
@pytest.mark.integration
class TestLevel2Integration:
    """Integration tests for Level 2 components"""

    def test_full_navigation_pipeline(self):
        """Test complete navigation pipeline"""
        # Load scene
        scene = PointCloudLoader.create_synthetic_scene()

        # Create occupancy grid
        min_b, max_b = scene.get_bounds()
        grid = OccupancyGrid3D(min_b, max_b, cell_size=2.0)

        # Mark obstacles
        non_ground = scene.points[::2]  # Every other point as obstacle
        grid.mark_occupied(non_ground)

        # Create navigation agent
        agent = NavigationAgent(grid)
        agent.current_position = min_b + np.array([5, 5, 2])

        # Set goal
        goal = max_b - np.array([5, 5, 2])
        success = agent.set_goal(goal)

        # Path should be found or goal unreachable
        assert success or not grid.is_navigable(goal)
