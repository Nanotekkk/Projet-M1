"""Main Demonstration Script - RANSAC Plane Detection

Demonstrates:
1. Multi-plane detection with RANSAC
2. Comparison with other methods (Linear Regression, K-Means)
3. Visualization in Open3D with colored planes
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.plane_detection.ransac_detector import RANSACPlaneDetector
from src.plane_detection.model_comparison import ModelComparison
from src.visualization.open3d_visualizer import Open3DVisualizer, create_synthetic_scene
from src.core.point_cloud_loader import PointCloudLoader


def load_ply_file():
    """Load a PLY file from disk"""
    print("\n" + "="*80)
    print("LOAD PLY FILE")
    print("="*80)
    
    file_path = input("\nEnter the path to the PLY file: ").strip()
    
    if not file_path:
        print("   No file specified. Using synthetic data instead.")
        return None
    
    try:
        filepath = Path(file_path)
        if not filepath.exists():
            print(f"   Error: File not found: {file_path}")
            return None
        
        print(f"\n   Loading PLY file: {filepath.name}...")
        point_cloud = PointCloudLoader.load_from_file(filepath)
        points = point_cloud.points
        print(f"   Successfully loaded {len(points)} points")
        if point_cloud.colors is not None:
            print(f"   Point cloud includes color information")
        return points
    except Exception as e:
        print(f"   Error loading file: {e}")
        return None


def get_input_source():
    """Ask user to choose between synthetic data or loading a PLY file"""
    print("\n" + "="*80)
    print("POINT CLOUD SOURCE SELECTION")
    print("="*80)
    print("\n1. Generate synthetic point cloud (with multiple planes)")
    print("2. Load PLY file from disk")
    
    choice = input("\nSelect option (1 or 2): ").strip()
    
    if choice == "2":
        points = load_ply_file()
        if points is not None:
            return points
        else:
            print("   Falling back to synthetic data...")
            return create_synthetic_scene(num_points=10000, num_planes=3, noise_level=0.1)
    else:
        print("\n1. Creating synthetic point cloud with multiple planes...")
        return create_synthetic_scene(num_points=10000, num_planes=3, noise_level=0.1)


def demo_ransac_multi_plane(points=None):
    "Demonstrate RANSAC multi-plane detection"
    print("\n" + "=" * 80)
    print("DEMO 1: RANSAC MULTI-PLANE DETECTION")
    print("=" * 80)

    # Use provided points or create synthetic scene
    if points is None:
        print("\n1. Creating synthetic point cloud with multiple planes...")
        points = create_synthetic_scene(num_points=10000, num_planes=3, noise_level=0.1)
    else:
        print("\n1. Using loaded point cloud...")
    print(f"   Processing {len(points)} points")

    # Detect planes
    print("\n2. Detecting planes with RANSAC...")
    detector = RANSACPlaneDetector(
        distance_threshold=0.15,
        iterations=1000,
        min_points_per_plane=50,
        max_planes=5,
    )

    planes = detector.detect_planes(points)
    print(f"   Found {len(planes)} planes")

    # Print plane information
    print("\n3. Plane Information:")
    print("-" * 80)
    for plane in planes:
        print(f"\n   Plane {plane.plane_id}:")
        print(f"     Normal: {plane.normal}")
        print(f"     Distance: {plane.distance:.4f}")
        print(f"     Inlier count: {plane.inlier_count} ({100*plane.inlier_count/len(points):.1f}%)")
        print(f"     Color: RGB{plane.color}")

    # Visualize
    print("\n4. Launching Open3D visualization...")
    visualizer = Open3DVisualizer("RANSAC Multi-Plane Detection")
    visualizer.visualize_point_cloud_with_planes(points, planes)

    return points, planes


def demo_model_comparison(points: np.ndarray):
    """Compare different plane detection methods"""
    print("\n" + "=" * 80)
    print("DEMO 2: MODEL COMPARISON")
    print("=" * 80)

    print("\n1. Comparing plane detection methods...")
    print("   Methods: RANSAC, Linear Regression, K-Means")

    comparator = ModelComparison()
    results = comparator.compare_all_methods(points)

    # Print comparison
    print("\n2. Comparison Results:")
    comparator.print_comparison(results)

    # Print detailed metrics
    print("\n3. Detailed Metrics:")
    print("-" * 80)
    for method_name, result in results.items():
        print(f"\n   {method_name}:")
        print(f"     Inlier Count: {result.inlier_count}")
        print(f"     Inlier Ratio: {result.inlier_ratio:.3f}")
        print(f"     Computation Time: {result.computation_time*1000:.3f} ms")
        print(f"     Plane Normal: {result.plane_normal}")

    return results


def demo_visualization_comparison(points: np.ndarray, results: dict):
    """Visualize comparison results"""
    print("\n" + "=" * 80)
    print("DEMO 3: VISUALIZATION COMPARISON")
    print("=" * 80)

    print("\nVisualizing results from each method...")
    print("(Close each visualization window to proceed to the next)")

    visualizer = Open3DVisualizer()
    visualizer.visualize_comparison_results(points, results)


def demo_real_scene():
    """Demonstrate with more realistic scene"""
    print("\n" + "=" * 80)
    print("DEMO 4: REALISTIC INDOOR SCENE")
    print("=" * 80)

    print("\n1. Creating realistic indoor scene...")
    # Floor
    floor_x = np.random.uniform(-10, 10, 5000)
    floor_y = np.random.uniform(-10, 10, 5000)
    floor_z = np.random.normal(0, 0.05, 5000)
    floor = np.column_stack([floor_x, floor_y, floor_z])

    # Wall 1 (vertical)
    wall1_x = np.ones(2000) * 10
    wall1_y = np.random.uniform(-10, 10, 2000)
    wall1_z = np.random.uniform(0, 3, 2000)
    wall1 = np.column_stack([wall1_x, wall1_y, wall1_z])

    # Wall 2 (vertical)
    wall2_x = np.random.uniform(-10, 10, 2000)
    wall2_y = np.ones(2000) * 10
    wall2_z = np.random.uniform(0, 3, 2000)
    wall2 = np.column_stack([wall2_x, wall2_y, wall2_z])

    # Ceiling
    ceiling_x = np.random.uniform(-10, 10, 2000)
    ceiling_y = np.random.uniform(-10, 10, 2000)
    ceiling_z = np.random.normal(3, 0.05, 2000)
    ceiling = np.column_stack([ceiling_x, ceiling_y, ceiling_z])

    scene = np.vstack([floor, wall1, wall2, ceiling])
    print(f"   Created scene with {len(scene)} points")

    # Detect planes
    print("\n2. Detecting planes in scene...")
    detector = RANSACPlaneDetector(
        distance_threshold=0.2,
        iterations=500,
        min_points_per_plane=100,
        max_planes=6,
    )

    planes = detector.detect_planes(scene)
    print(f"   Found {len(planes)} planes")

    # Print plane information
    print("\n3. Detected Planes:")
    for plane in planes:
        print(f"   Plane {plane.plane_id}: {plane.inlier_count} inliers "
              f"({100*plane.inlier_count/len(scene):.1f}%)")

    # Visualize
    print("\n4. Launching visualization...")
    visualizer = Open3DVisualizer("Indoor Scene - Multi-Plane Detection")
    visualizer.visualize_point_cloud_with_planes(scene, planes)


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 80)
    print("LIDAR RANSAC PLANE DETECTION - COMPLETE DEMONSTRATION")
    print("=" * 80)

    try:
        # Get input source (synthetic or PLY file)
        points = get_input_source()
        print(f"   Using {len(points)} points for analysis")

        # Ask which demos to run
        demo_choice = input("\nRun all demos (1) or basic RANSAC only (2)? Default is all (1): ").strip()
        
        if demo_choice == "2":
            # Run only basic RANSAC
            _, planes = demo_ransac_multi_plane(points)
        else:
            # Run all demos
            _, planes = demo_ransac_multi_plane(points)

            # Demo 2: Model comparison
            results = demo_model_comparison(points)

            # Demo 3: Visualization comparison
            demo_visualization_comparison(points, results)

            # Demo 4: Realistic scene
            demo_real_scene()

        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED")
        print("=" * 80)
        print("\nKey Results:")
        if planes:
            print(f"  - Found {len(planes)} planes in point cloud")
            for plane in planes[:3]:
                print(f"    Plane {plane.plane_id}: {plane.inlier_count} inliers "
                      f"({100*plane.inlier_count/len(points):.1f}%)")

    except ImportError as e:
        print(f"\nError: Missing dependency - {e}")
        print("\nPlease install required packages:")
        print("  pip install -r requirements.txt")
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
