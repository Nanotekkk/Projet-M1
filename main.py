"""Main demonstration script for LiDAR VR Navigation System"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.point_cloud_loader import PointCloudLoader
from src.level1.ransac_segmentation import RANSACSegmentation
from src.level1.dl_segmentation import DLSegmentation
from src.level1.teleportation_validator import TeleportationValidator
from src.level2.euclidean_clustering import EuclideanClustering
from src.level2.plane_detection import PlaneDetection
from src.level2.occupancy_grid_3d import OccupancyGrid3D
from src.level2.pathfinding_a_star import AStarPathfinder
from src.level2.navigation_agent import NavigationAgent
from src.core.observer import Observer


class ConsoleObserver(Observer):
    """Observer for printing navigation events"""

    def update(self, event_type: str, data=None):
        print(f"[EVENT] {event_type}: {data}")


def demo_level1_ransac():
    """Demonstrate Level 1 RANSAC ground detection"""
    print("\n" + "=" * 60)
    print("LEVEL 1 - RANSAC Ground Detection")
    print("=" * 60)

    # Load synthetic point cloud
    print("\n1. Loading synthetic scene...")
    scene = PointCloudLoader.create_synthetic_scene(num_ground=5000, num_obstacles=2000)
    print(f"   Loaded {len(scene)} points")

    # Perform RANSAC segmentation
    print("\n2. Performing RANSAC ground detection...")
    ransac = RANSACSegmentation(distance_threshold=0.5, iterations=200)
    result = ransac.segment(scene.points)

    print(f"   Ground points detected: {len(result.ground_indices)}")
    print(f"   Non-ground points: {len(scene.points) - len(result.ground_indices)}")
    print(f"   Detection rate: {len(result.ground_indices) / len(scene.points) * 100:.1f}%")

    if result.metadata["plane_normal"] is not None:
        print(f"   Plane normal: {result.metadata['plane_normal']}")
        print(f"   Plane distance: {result.metadata['plane_distance']:.3f}")

    return scene, result


def demo_level1_dl():
    """Demonstrate Level 1 Deep Learning segmentation"""
    print("\n" + "=" * 60)
    print("LEVEL 1 - Deep Learning Ground Detection")
    print("=" * 60)

    # Load synthetic point cloud
    print("\n1. Loading synthetic scene...")
    scene = PointCloudLoader.create_synthetic_scene()
    print(f"   Loaded {len(scene)} points")

    # Create and train DL model
    print("\n2. Training Deep Learning model...")
    dl = DLSegmentation(confidence_threshold=0.5)

    # Create synthetic labels (ground points are first 60%)
    labels = np.zeros(len(scene.points), dtype=int)
    labels[: int(0.6 * len(scene.points))] = 1

    dl.train_model(scene.points, labels, epochs=5)
    print("   Training complete")

    # Perform segmentation
    print("\n3. Performing segmentation...")
    result = dl.segment(scene.points)

    print(f"   Ground points detected: {len(result.ground_indices)}")
    print(f"   Detection rate: {len(result.ground_indices) / len(scene.points) * 100:.1f}%")

    return scene, result


def demo_level1_teleportation(scene, segmentation_result):
    """Demonstrate Level 1 VR teleportation validation"""
    print("\n" + "=" * 60)
    print("LEVEL 1 - VR Teleportation Validation")
    print("=" * 60)

    # Get ground points
    ground_points = segmentation_result.get_ground_points(scene.points)
    print(f"\n1. Using {len(ground_points)} ground points")

    # Create validator
    print("\n2. Creating teleportation validator...")
    validator = TeleportationValidator(
        min_ground_points=5, max_height_above_ground=0.5
    )

    # Test multiple positions
    print("\n3. Validating teleportation positions...")
    test_positions = np.array([
        ground_points.mean(axis=0),
        ground_points.mean(axis=0) + np.array([5, 0, 0]),
        ground_points.mean(axis=0) + np.array([0, 5, 0]),
        ground_points.mean(axis=0) + np.array([0, 0, 5]),  # Too high
    ])

    results = validator.validate_multiple_positions(test_positions, ground_points)

    for i, result in enumerate(results):
        print(
            f"   Position {i}: {'VALID' if result.is_valid else 'INVALID'} - {result.reason}"
        )


def demo_level2_clustering():
    """Demonstrate Level 2 Euclidean clustering"""
    print("\n" + "=" * 60)
    print("LEVEL 2 - Euclidean Clustering")
    print("=" * 60)

    # Load scene
    print("\n1. Loading synthetic scene...")
    scene = PointCloudLoader.create_synthetic_scene()
    print(f"   Loaded {len(scene)} points")

    # Perform clustering
    print("\n2. Performing euclidean clustering...")
    clustering = EuclideanClustering(eps=5.0, min_points=10)
    result = clustering.cluster(scene.points)

    print(f"   Number of clusters: {result.n_clusters}")
    print(f"   Cluster centers: {result.cluster_centers.shape}")

    return scene, result


def demo_level2_plane_detection(scene):
    """Demonstrate Level 2 plane detection"""
    print("\n" + "=" * 60)
    print("LEVEL 2 - Plane Detection")
    print("=" * 60)

    print("\n1. Detecting planes in scene...")
    detector = PlaneDetection(distance_threshold=1.0, min_inliers=50)
    planes = detector.detect_planes(scene.points, num_planes=3)

    print(f"   Planes detected: {len(planes)}")

    for i, plane in enumerate(planes):
        print(f"\n   Plane {i + 1}:")
        print(f"      Type: {plane.plane_type}")
        print(f"      Normal: {plane.normal}")
        print(f"      Points: {len(plane.points_indices)}")
        print(f"      Area: {plane.area:.2f}")


def demo_level2_occupancy_grid():
    """Demonstrate Level 2 occupancy grid"""
    print("\n" + "=" * 60)
    print("LEVEL 2 - 3D Occupancy Grid")
    print("=" * 60)

    # Load scene
    print("\n1. Loading scene...")
    scene = PointCloudLoader.create_synthetic_scene()
    min_b, max_b = scene.get_bounds()

    # Create grid
    print("\n2. Creating occupancy grid...")
    grid = OccupancyGrid3D(min_b, max_b, cell_size=2.0)
    print(f"   Grid shape: {grid.grid.shape}")
    print(f"   Cell size: {grid.cell_size}")

    # Mark obstacles
    print("\n3. Marking obstacles...")
    non_ground = scene.points[::3]  # Every 3rd point
    grid.mark_occupied(non_ground)

    # Check navigability
    print("\n4. Checking navigability...")
    free_cells = grid.get_free_cells()
    occupied_cells = grid.get_occupied_cells()

    print(f"   Free cells: {len(free_cells)}")
    print(f"   Occupied cells: {len(occupied_cells)}")


def demo_level2_pathfinding():
    """Demonstrate Level 2 A* pathfinding"""
    print("\n" + "=" * 60)
    print("LEVEL 2 - A* Pathfinding")
    print("=" * 60)

    # Setup
    print("\n1. Setting up environment...")
    scene = PointCloudLoader.create_synthetic_scene()
    min_b, max_b = scene.get_bounds()

    grid = OccupancyGrid3D(min_b, max_b, cell_size=2.0)

    # Mark some obstacles
    print("\n2. Creating obstacles...")
    obstacle_center = (min_b + max_b) / 2
    for i in range(-10, 10):
        for j in range(-10, 10):
            pos = obstacle_center + np.array([i * 2, j * 2, 0])
            if grid.is_navigable(pos):
                grid.mark_occupied(np.array([pos]))

    # Find path
    print("\n3. Finding path...")
    pathfinder = AStarPathfinder()
    start = min_b + np.array([5, 5, 2])
    goal = max_b - np.array([5, 5, 2])

    path = pathfinder.find_path(start, goal, grid)

    if path:
        print(f"   Path found with {len(path)} waypoints")
        print(f"   Start: {start}")
        print(f"   Goal: {goal}")
        print(f"   Path length: {sum(np.linalg.norm(np.diff(path, axis=0), axis=1)):.2f}m")
    else:
        print("   No path found")


def demo_level2_navigation_agent():
    """Demonstrate Level 2 navigation agent"""
    print("\n" + "=" * 60)
    print("LEVEL 2 - Navigation Agent with Observer Pattern")
    print("=" * 60)

    # Setup
    print("\n1. Creating navigation environment...")
    scene = PointCloudLoader.create_synthetic_scene()
    min_b, max_b = scene.get_bounds()

    grid = OccupancyGrid3D(min_b, max_b, cell_size=2.0)
    grid.mark_occupied(scene.points[::5])

    # Create agent and observer
    print("\n2. Creating navigation agent...")
    agent = NavigationAgent(grid)

    observer = ConsoleObserver()
    agent.attach(observer)

    # Set position and goal
    print("\n3. Setting navigation goal...")
    start_pos = min_b + np.array([10, 10, 2])
    goal_pos = max_b - np.array([10, 10, 2])

    agent.current_position = start_pos
    success = agent.set_goal(goal_pos)

    if success:
        print(f"\n4. Following path...")
        # Simulate following path
        for i in range(min(5, len(agent.current_path))):
            waypoint = agent.get_next_waypoint()
            if waypoint is not None:
                agent.update_position(waypoint)

        print(f"\n   Agent status:")
        print(f"   Current waypoint: {agent.current_waypoint_idx}/{len(agent.current_path)}")
        print(f"   Remaining distance: {agent.get_remaining_distance():.2f}m")


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 60)
    print("LiDAR VR Navigation System Demonstration")
    print("=" * 60)

    try:
        # Level 1 demonstrations
        print("\n\nLEVEL 1 - FUNDAMENTALS")
        print("=" * 60)

        scene_l1, result_ransac = demo_level1_ransac()
        demo_level1_teleportation(scene_l1, result_ransac)

        print("\n")
        scene_l1_dl, result_dl = demo_level1_dl()

        # Level 2 demonstrations
        print("\n\nLEVEL 2 - ADVANCED NAVIGATION")
        print("=" * 60)

        scene_l2, result_clustering = demo_level2_clustering()
        demo_level2_plane_detection(scene_l2)
        demo_level2_occupancy_grid()
        demo_level2_pathfinding()
        demo_level2_navigation_agent()

        print("\n" + "=" * 60)
        print("[✓] All demonstrations completed successfully")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n[✗] Error during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
