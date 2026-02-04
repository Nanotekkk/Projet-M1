"""Benchmarking and Comparison Script for Navigation Algorithms"""

import numpy as np
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.core.point_cloud_loader import PointCloudLoader
from src.level1.ransac_segmentation import RANSACSegmentation
from src.level1.dl_segmentation import DLSegmentation
from src.level2.euclidean_clustering import EuclideanClustering
from src.level2.plane_detection import PlaneDetection
from src.level2.occupancy_grid_3d import OccupancyGrid3D
from src.level2.pathfinding_a_star import AStarPathfinder
from src.level2.navigation_agent import NavigationAgent


class PerformanceBenchmark:
    """Benchmark class for algorithm performance comparison"""

    def __init__(self):
        self.results = {}

    def benchmark_function(self, name: str, func, *args, **kwargs) -> float:
        """Benchmark a function and record execution time"""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        self.results[name] = {
            "time": elapsed,
            "result": result,
        }

        return elapsed

    def print_results(self):
        """Print benchmark results in a formatted table"""
        print("\n" + "=" * 70)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("=" * 70)
        print(f"{'Algorithm':<40} {'Time (ms)':<15} {'Memory (MB)':<15}")
        print("-" * 70)

        total_time = 0
        for name, data in sorted(
            self.results.items(), key=lambda x: x[1]["time"], reverse=True
        ):
            time_ms = data["time"] * 1000
            print(f"{name:<40} {time_ms:<15.2f}")
            total_time += data["time"]

        print("-" * 70)
        print(f"{'Total Time':<40} {total_time * 1000:<15.2f}")
        print("=" * 70)


def benchmark_level1_segmentation():
    """Benchmark Level 1 segmentation methods"""
    print("\n" + "=" * 70)
    print("LEVEL 1 - SEGMENTATION BENCHMARKS")
    print("=" * 70)

    benchmark = PerformanceBenchmark()

    # Create test data
    print("\nPreparing test data...")
    scene_sizes = [1000, 5000, 10000]

    for size in scene_sizes:
        print(f"\n--- Point Cloud Size: {size} points ---")

        # Create synthetic scene
        scene = PointCloudLoader.create_synthetic_scene(
            num_ground=int(0.6 * size), num_obstacles=int(0.4 * size)
        )

        # RANSAC segmentation
        print("  RANSAC...", end=" ", flush=True)
        ransac = RANSACSegmentation(iterations=200)
        t_ransac = benchmark.benchmark_function(
            f"RANSAC ({size} points)", ransac.segment, scene.points
        )
        print(f"✓ {t_ransac * 1000:.2f}ms")

        # DL segmentation
        print("  Deep Learning...", end=" ", flush=True)
        dl = DLSegmentation()
        t_dl = benchmark.benchmark_function(
            f"Deep Learning ({size} points)", dl.segment, scene.points
        )
        print(f"✓ {t_dl * 1000:.2f}ms")

    benchmark.print_results()


def benchmark_level2_algorithms():
    """Benchmark Level 2 navigation algorithms"""
    print("\n" + "=" * 70)
    print("LEVEL 2 - ADVANCED NAVIGATION BENCHMARKS")
    print("=" * 70)

    benchmark = PerformanceBenchmark()

    # Create test scene
    print("\nPreparing test environment...")
    scene = PointCloudLoader.create_synthetic_scene(num_ground=5000, num_obstacles=3000)

    # Euclidean Clustering
    print("\n1. Euclidean Clustering...")
    clustering = EuclideanClustering(eps=5.0, min_points=10)
    t_cluster = benchmark.benchmark_function(
        "Euclidean Clustering", clustering.cluster, scene.points
    )
    print(f"   ✓ {t_cluster * 1000:.2f}ms")

    # Plane Detection
    print("\n2. Plane Detection...")
    detector = PlaneDetection(distance_threshold=1.0, min_inliers=50)
    t_planes = benchmark.benchmark_function(
        "Plane Detection", detector.detect_planes, scene.points, 3
    )
    print(f"   ✓ {t_planes * 1000:.2f}ms")

    # Occupancy Grid Creation
    print("\n3. 3D Occupancy Grid...")
    min_b, max_b = scene.get_bounds()

    def create_and_fill_grid():
        grid = OccupancyGrid3D(min_b, max_b, cell_size=2.0)
        grid.mark_occupied(scene.points[::5])
        return grid

    t_grid = benchmark.benchmark_function("Occupancy Grid Creation", create_and_fill_grid)
    print(f"   ✓ {t_grid * 1000:.2f}ms")

    grid = benchmark.results["Occupancy Grid Creation"]["result"]

    # A* Pathfinding
    print("\n4. A* Pathfinding...")
    pathfinder = AStarPathfinder()
    start = min_b + np.array([5, 5, 2])
    goal = max_b - np.array([5, 5, 2])

    def find_path():
        return pathfinder.find_path(start, goal, grid)

    t_astar = benchmark.benchmark_function("A* Pathfinding", find_path)
    print(f"   ✓ {t_astar * 1000:.2f}ms")

    # Navigation Agent
    print("\n5. Navigation Agent Setup...")
    agent = NavigationAgent(grid)
    agent.current_position = start

    def nav_setup():
        return agent.set_goal(goal)

    t_nav = benchmark.benchmark_function("Navigation Agent Setup", nav_setup)
    print(f"   ✓ {t_nav * 1000:.2f}ms")

    benchmark.print_results()


def benchmark_scalability():
    """Test algorithm scalability with increasing point cloud sizes"""
    print("\n" + "=" * 70)
    print("SCALABILITY ANALYSIS")
    print("=" * 70)

    sizes = [1000, 5000, 10000, 50000, 100000]
    results = {"RANSAC": [], "DL": [], "Clustering": []}

    print(f"\n{'Size':<10} {'RANSAC (ms)':<15} {'DL (ms)':<15} {'Clustering (ms)':<15}")
    print("-" * 55)

    for size in sizes:
        scene = PointCloudLoader.create_synthetic_scene(
            num_ground=int(0.6 * size), num_obstacles=int(0.4 * size)
        )

        # RANSAC
        ransac = RANSACSegmentation(iterations=100)
        start = time.perf_counter()
        ransac.segment(scene.points)
        t_ransac = (time.perf_counter() - start) * 1000

        # DL
        dl = DLSegmentation()
        start = time.perf_counter()
        dl.segment(scene.points)
        t_dl = (time.perf_counter() - start) * 1000

        # Clustering
        clustering = EuclideanClustering(eps=5.0, min_points=10)
        start = time.perf_counter()
        clustering.cluster(scene.points)
        t_cluster = (time.perf_counter() - start) * 1000

        results["RANSAC"].append(t_ransac)
        results["DL"].append(t_dl)
        results["Clustering"].append(t_cluster)

        print(
            f"{size:<10} {t_ransac:<15.2f} {t_dl:<15.2f} {t_cluster:<15.2f}"
        )

    print("-" * 55)

    # Print complexity analysis
    print("\nComplexity Analysis:")
    for alg, times in results.items():
        # Calculate growth factor (time for largest / time for smallest)
        if times[0] > 0:
            growth = times[-1] / times[0]
            print(f"  {alg}: {growth:.2f}x growth (100x point increase)")


def benchmark_memory_usage():
    """Analyze memory usage patterns"""
    print("\n" + "=" * 70)
    print("MEMORY USAGE ANALYSIS")
    print("=" * 70)

    import tracemalloc

    sizes = [1000, 10000, 50000]

    print(f"\n{'Size':<10} {'Peak Memory (MB)':<20}")
    print("-" * 30)

    for size in sizes:
        scene = PointCloudLoader.create_synthetic_scene(
            num_ground=int(0.6 * size), num_obstacles=int(0.4 * size)
        )

        tracemalloc.start()

        # RANSAC
        ransac = RANSACSegmentation()
        ransac.segment(scene.points)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        print(f"{size:<10} {peak_mb:<20.2f}")

    print("-" * 30)


def compare_segmentation_methods():
    """Compare RANSAC vs Deep Learning segmentation"""
    print("\n" + "=" * 70)
    print("SEGMENTATION METHODS COMPARISON")
    print("=" * 70)

    scene = PointCloudLoader.create_synthetic_scene(num_ground=5000, num_obstacles=3000)

    # RANSAC
    print("\nRANSAC Ground Detection:")
    ransac = RANSACSegmentation(distance_threshold=0.5, iterations=200)
    result_ransac = ransac.segment(scene.points)
    ground_count_ransac = len(result_ransac.ground_indices)
    detection_rate_ransac = ground_count_ransac / len(scene.points) * 100

    print(f"  Ground points detected: {ground_count_ransac}")
    print(f"  Detection rate: {detection_rate_ransac:.1f}%")
    print(f"  Non-ground points: {len(scene.points) - ground_count_ransac}")

    # Deep Learning
    print("\nDeep Learning Ground Detection:")
    dl = DLSegmentation(confidence_threshold=0.5)

    # Create training data
    train_labels = np.zeros(len(scene.points), dtype=int)
    train_labels[: int(0.6 * len(scene.points))] = 1

    dl.train_model(scene.points, train_labels, epochs=3)
    result_dl = dl.segment(scene.points)
    ground_count_dl = len(result_dl.ground_indices)
    detection_rate_dl = ground_count_dl / len(scene.points) * 100

    print(f"  Ground points detected: {ground_count_dl}")
    print(f"  Detection rate: {detection_rate_dl:.1f}%")
    print(f"  Non-ground points: {len(scene.points) - ground_count_dl}")

    # Comparison
    print("\nComparison:")
    print(f"  Difference in detection: {abs(detection_rate_ransac - detection_rate_dl):.1f}%")

    # Agreement between methods
    agreement = np.sum(
        result_ransac.labels == result_dl.labels
    ) / len(scene.points) * 100
    print(f"  Method agreement: {agreement:.1f}%")


def main():
    """Run all benchmarks"""
    print("\n" + "=" * 70)
    print("LiDAR VR Navigation System - Performance Benchmarks")
    print("=" * 70)

    try:
        # Level 1 benchmarks
        benchmark_level1_segmentation()

        # Level 2 benchmarks
        benchmark_level2_algorithms()

        # Scalability analysis
        benchmark_scalability()

        # Memory analysis
        benchmark_memory_usage()

        # Method comparison
        compare_segmentation_methods()

        print("\n" + "=" * 70)
        print("✓ All benchmarks completed successfully!")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n✗ Benchmark error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
