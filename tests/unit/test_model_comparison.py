"""Tests for Model Comparison

Author: Matheo LANCEA
Description: Unit tests for plane detection method comparison
"""

import pytest
import numpy as np
from src.plane_detection.model_comparison import ModelComparison, ComparisonResult


class TestModelComparison:
    """Test suite for model comparison"""

    @pytest.fixture
    def comparator(self):
        """Create a model comparison instance"""
        return ModelComparison()

    @pytest.fixture
    def synthetic_plane(self):
        """Create a synthetic plane point cloud"""
        np.random.seed(42)
        x = np.random.uniform(-5, 5, 1000)
        y = np.random.uniform(-5, 5, 1000)
        z = np.random.normal(0, 0.1, 1000)  # Ground plane with noise
        return np.column_stack([x, y, z])

    def test_comparator_initialization(self, comparator):
        """Test comparator initialization"""
        assert isinstance(comparator, ModelComparison)

    def test_compare_all_methods(self, comparator, synthetic_plane):
        """Test comparison of all methods"""
        results = comparator.compare_all_methods(synthetic_plane)

        # Should return results for all 6 methods
        assert len(results) == 6

        expected_methods = {
            "RANSAC",
            "Linear Regression",
            "K-Means",
        }

        assert set(results.keys()) == expected_methods

    def test_result_structure(self, comparator, synthetic_plane):
        """Test ComparisonResult structure"""
        results = comparator.compare_all_methods(synthetic_plane)

        for method_name, result in results.items():
            assert isinstance(result, ComparisonResult)

            # Check required attributes
            assert hasattr(result, "model_name")
            assert hasattr(result, "inlier_count")
            assert hasattr(result, "inlier_ratio")
            assert hasattr(result, "plane_normal")
            assert hasattr(result, "plane_distance")
            assert hasattr(result, "computation_time")
            assert hasattr(result, "inlier_indices")
            assert hasattr(result, "additional_metrics")

            # Check attribute types
            assert isinstance(result.model_name, str)
            assert isinstance(result.inlier_count, (int, np.integer))
            assert isinstance(result.inlier_ratio, (float, np.floating))
            assert isinstance(result.plane_normal, np.ndarray)
            assert isinstance(result.computation_time, float)
            assert isinstance(result.inlier_indices, np.ndarray)
            assert isinstance(result.additional_metrics, dict)

    def test_inlier_ratio_valid_range(self, comparator, synthetic_plane):
        """Test that inlier ratios are in valid range"""
        results = comparator.compare_all_methods(synthetic_plane)

        for result in results.values():
            assert 0 <= result.inlier_ratio <= 1

    def test_inlier_count_consistency(self, comparator, synthetic_plane):
        """Test that inlier count matches ratio"""
        results = comparator.compare_all_methods(synthetic_plane)

        n_points = len(synthetic_plane)

        for result in results.values():
            expected_count = result.inlier_ratio * n_points
            assert abs(result.inlier_count - expected_count) < 1

    def test_plane_normal_normalization(self, comparator, synthetic_plane):
        """Test that plane normals are unit vectors"""
        results = comparator.compare_all_methods(synthetic_plane)

        for result in results.values():
            norm = np.linalg.norm(result.plane_normal)
            assert abs(norm - 1.0) < 1e-9

    def test_computation_time_positive(self, comparator, synthetic_plane):
        """Test that computation times are positive"""
        results = comparator.compare_all_methods(synthetic_plane)

        for result in results.values():
            assert result.computation_time >= 0

    def test_ransac_method(self, comparator, synthetic_plane):
        """Test RANSAC method specifically"""
        result = comparator._method_ransac(synthetic_plane)

        assert result.model_name == "RANSAC"
        assert result.inlier_count > 500  # Should detect most points
        assert 0 <= result.inlier_ratio <= 1

    def test_linear_regression_method(self, comparator, synthetic_plane):
        """Test Linear Regression method"""
        result = comparator._method_linear_regression(synthetic_plane)

        assert result.model_name == "Linear Regression"
        assert result.inlier_count > 0
        assert "coefficients" in result.additional_metrics

    def test_kmeans_method(self, comparator, synthetic_plane):
        """Test K-Means method"""
        result = comparator._method_kmeans(synthetic_plane)

        assert result.model_name == "K-Means"
        assert result.inlier_count > 0
        assert "n_clusters" in result.additional_metrics

    def test_gmm_method(self, comparator, synthetic_plane):
        """Test Gaussian Mixture Model method"""
        result = comparator._method_gmm(synthetic_plane)

        assert result.model_name == "GMM"
        assert result.inlier_count > 0
        assert "bic" in result.additional_metrics

    def test_pca_method(self, comparator, synthetic_plane):
        """Test PCA method"""
        result = comparator._method_pca(synthetic_plane)

        assert result.model_name == "PCA"
        assert result.inlier_count > 0
        assert "variance_explained" in result.additional_metrics

    def test_height_based_method(self, comparator, synthetic_plane):
        """Test Height-Based method"""
        result = comparator._method_height_based(synthetic_plane)

        assert result.model_name == "Height-Based"
        assert result.inlier_count > 0
        assert "z_threshold" in result.additional_metrics

    def test_print_comparison(self, comparator, synthetic_plane, capsys):
        """Test print comparison output"""
        results = comparator.compare_all_methods(synthetic_plane)

        # Should not raise
        comparator.print_comparison(results)

        # Check output
        captured = capsys.readouterr()
        assert "PLANE DETECTION METHOD COMPARISON" in captured.out
        assert "Method" in captured.out
        assert "Inliers" in captured.out

    def test_method_results_comparable(self, comparator, synthetic_plane):
        """Test that results from different methods can be compared"""
        results = comparator.compare_all_methods(synthetic_plane)

        # Create simple ranking
        by_inliers = sorted(results.items(), key=lambda x: x[1].inlier_count, reverse=True)
        by_time = sorted(results.items(), key=lambda x: x[1].computation_time)

        # Should have valid orderings
        assert len(by_inliers) == 6
        assert len(by_time) == 6

    def test_reproducibility(self, synthetic_plane):
        """Test that results are reproducible with fixed seed"""
        np.random.seed(42)
        comp1 = ModelComparison()
        results1 = comp1.compare_all_methods(synthetic_plane)

        np.random.seed(42)
        comp2 = ModelComparison()
        results2 = comp2.compare_all_methods(synthetic_plane)

        # Results should be similar
        for method in results1.keys():
            r1 = results1[method]
            r2 = results2[method]
            assert r1.inlier_count == r2.inlier_count

    def test_comparison_on_structured_data(self):
        """Test comparison methods on structured test data"""
        # Create perfectly flat plane
        points = np.zeros((1000, 3))
        points[:, :2] = np.random.uniform(-5, 5, (1000, 2))

        comparator = ModelComparison()
        results = comparator.compare_all_methods(points)

        # All methods should find high inlier ratios
        for result in results.values():
            assert result.inlier_ratio > 0.8

    def test_handles_noisy_data(self):
        """Test comparison methods handle noisy data"""
        # Create noisy plane
        np.random.seed(42)
        points = np.zeros((1000, 3))
        points[:, :2] = np.random.uniform(-5, 5, (1000, 2))
        points[:, 2] = np.random.normal(0, 2, 1000)  # Significant noise

        comparator = ModelComparison()
        results = comparator.compare_all_methods(points)

        # Methods should still return valid results
        for result in results.values():
            assert 0 <= result.inlier_ratio <= 1
            assert result.computation_time >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
