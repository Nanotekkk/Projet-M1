"""Test PLY file loading functionality"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.point_cloud_loader import PointCloudLoader
from src.plane_detection.ransac_detector import RANSACPlaneDetector

def create_test_ply_file():
    """Create a simple test PLY file"""
    output_file = Path(__file__).parent / "test_plane.ply"
    
    # Create a simple plane with noise
    np.random.seed(42)
    x = np.random.uniform(-10, 10, 500)
    y = np.random.uniform(-10, 10, 500)
    z = np.random.normal(0, 0.1, 500)  # Mostly flat with small noise
    
    points = np.column_stack([x, y, z])
    
    # Write to PLY file
    with open(output_file, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        # Write vertex data
        for point in points:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    
    return output_file

def main():
    print("\n" + "="*80)
    print("PLY FILE LOADING TEST")
    print("="*80)
    
    # Create test PLY file
    print("\n1. Creating test PLY file...")
    test_file = create_test_ply_file()
    print(f"   ✓ Created: {test_file}")
    
    # Load the PLY file
    print("\n2. Loading PLY file with PointCloudLoader...")
    try:
        point_cloud = PointCloudLoader.load_from_file(test_file)
        print(f"   ✓ Successfully loaded {len(point_cloud.points)} points")
        print(f"   Points shape: {point_cloud.points.shape}")
        print(f"   Color information: {point_cloud.colors is not None}")
    except Exception as e:
        print(f"   ✗ Error loading file: {e}")
        return
    
    # Test RANSAC on loaded PLY data
    print("\n3. Testing RANSAC on loaded PLY data...")
    try:
        detector = RANSACPlaneDetector(
            distance_threshold=0.15,
            iterations=1000,
            min_points_per_plane=50,
            max_planes=3,
        )
        planes = detector.detect_planes(point_cloud.points)
        print(f"   ✓ Found {len(planes)} planes")
        
        for plane in planes:
            print(f"      Plane {plane.plane_id}: {plane.inlier_count} inliers "
                  f"({100*plane.inlier_count/len(point_cloud.points):.1f}%)")
    except Exception as e:
        print(f"   ✗ Error during plane detection: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Clean up
    print("\n4. Cleaning up test file...")
    test_file.unlink()
    print(f"   ✓ Removed: {test_file}")
    
    print("\n" + "="*80)
    print("✓ PLY LOADING TEST COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nYour project now supports:")
    print("  • Loading PLY files via PointCloudLoader")
    print("  • Running RANSAC on loaded point clouds")
    print("  • Interactive menu in main.py to choose PLY files")

if __name__ == "__main__":
    main()
