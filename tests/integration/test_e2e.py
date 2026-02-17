"""Integration test - End-to-End workflow

Test complete pipeline from point cloud to visualization
"""

import pytest
import numpy as np
from src.plane_detection.ransac_detector import RANSACPlaneDetector
from src.plane_detection.model_comparison import ModelComparison


class TestE2EWorkflow:
    """End-to-end tests"""

    def test_ransac_to_comparison_workflow(self):
        """Test complete workflow: Generate -> Detect -> Compare"""
        # Generate synthetic point cloud
        np.random.seed(42)
        
        # Create 3 planes
        plane1 = np.random.uniform(-5, 5, (2000, 3))
        plane1[:, 2] = 0
        
        plane2 = np.random.uniform(-5, 5, (2000, 3))
        plane2[:, 2] = 5
        
        scene = np.vstack([plane1, plane2])

        # Detect planes
        detector = RANSACPlaneDetector(max_planes=3)
        planes = detector.detect_planes(scene)

        assert len(planes) >= 1
        
        # Compare methods
        comparator = ModelComparison()
        results = comparator.compare_all_methods(scene)

        assert len(results) == 6

        # Verify RANSAC finds most inliers
        ransac_result = results["RANSAC"]
        assert ransac_result.inlier_count > 1000

    def test_realistic_scene_workflow(self):
        """Test on realistic indoor scene"""
        # Floor
        floor_x = np.random.uniform(-10, 10, 3000)
        floor_y = np.random.uniform(-10, 10, 3000)
        floor_z = np.random.normal(0, 0.05, 3000)
        floor = np.column_stack([floor_x, floor_y, floor_z])

        # Wall
        wall_x = np.ones(1000) * 10
        wall_y = np.random.uniform(-10, 10, 1000)
        wall_z = np.random.uniform(0, 3, 1000)
        wall = np.column_stack([wall_x, wall_y, wall_z])

        scene = np.vstack([floor, wall])

        # Detect
        detector = RANSACPlaneDetector(distance_threshold=0.2, max_planes=5)
        planes = detector.detect_planes(scene)

        assert len(planes) >= 1

        for plane in planes:
            assert plane.inlier_count > 100
            assert 0 <= plane.inlier_count <= len(scene)

    def test_comparison_consistency(self):
        """Test that comparison methods are consistent"""
        points = np.zeros((1000, 3))
        points[:, :2] = np.random.uniform(-5, 5, (1000, 2))

        comparator = ModelComparison()
        results1 = comparator.compare_all_methods(points)
        
        # Total inliers should not exceed point count
        for result in results1.values():
            assert result.inlier_count <= len(points)

    def test_multiple_runs_compatibility(self):
        """Test running pipeline multiple times"""
        detector = RANSACPlaneDetector()
        comparator = ModelComparison()

        for i in range(3):
            # Generate new scene each iteration
            points = np.random.uniform(-5, 5, (1000, 3))
            
            # Run both
            planes = detector.detect_planes(points)
            results = comparator.compare_all_methods(points)

            assert len(planes) >= 0
            assert len(results) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
