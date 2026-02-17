"""Open3D Visualization Module

Visualizes point clouds with colored planes for easy identification.
"""

import numpy as np
import open3d as o3d
from typing import List, Optional, Tuple
from src.plane_detection.ransac_detector import PlaneResult


class Open3DVisualizer:
    """Open3D-based point cloud visualization with plane coloring"""

    def __init__(self, window_title: str = "Point Cloud Viewer"):
        """
        Initialize visualizer.

        Args:
            window_title: Title for the visualization window
        """
        self.window_title = window_title
        self.vis = o3d.visualization.Visualizer()
        self.geometries = []

    def visualize_point_cloud_with_planes(
        self,
        points: np.ndarray,
        planes: List[PlaneResult],
        background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """
        Visualize point cloud with different colored planes.

        Args:
            points: Full point cloud (N, 3)
            planes: List of detected planes
            background_color: RGB background color
        """
        self.vis.create_window(self.window_title, width=1200, height=800)

        # Create colors array
        colors = np.ones((len(points), 3))  # Start with white

        # Color points according to their plane membership
        unassigned_color = (0.8, 0.8, 0.8)  # Light gray for unassigned points
        colors[:] = unassigned_color

        for plane in planes:
            colors[plane.inlier_indices] = plane.color

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        self.vis.add_geometry(pcd)

        # Draw plane normals as arrows
        self._draw_plane_normals(planes)

        # Set background color
        opt = self.vis.get_render_option()
        opt.background_color = np.array(background_color)
        opt.point_size = 3.0

        # Run visualization
        self.vis.run()
        self.vis.destroy_window()

    def visualize_multiple_methods(
        self,
        points: np.ndarray,
        method_planes: dict,
    ) -> None:
        """
        Create separate visualizations for each detection method.

        Args:
            points: Full point cloud (N, 3)
            method_planes: Dict mapping method names to lists of PlaneResult
        """
        for method_name, planes in method_planes.items():
            print(f"\nVisualizing results from {method_name}...")
            self.visualize_point_cloud_with_planes(
                points,
                planes,
                background_color=(1.0, 1.0, 1.0)
            )

    def visualize_comparison_results(
        self,
        points: np.ndarray,
        comparison_results: dict,
    ) -> None:
        """
        Visualize plane detection results from model comparison.

        Args:
            points: Full point cloud (N, 3)
            comparison_results: Dict from ModelComparison.compare_all_methods()
        """
        # Convert comparison results to planes for visualization
        method_planes = {}

        for method_name, result in comparison_results.items():
            # Create a PlaneResult for visualization
            colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
            color = colors[hash(method_name) % len(colors)]

            plane = PlaneResult(
                plane_id=0,
                normal=result.plane_normal,
                distance=result.plane_distance,
                inlier_indices=result.inlier_indices,
                inlier_count=result.inlier_count,
                color=color,
            )
            method_planes[method_name] = [plane]

        self.visualize_multiple_methods(points, method_planes)

    def visualize_simple(
        self,
        points: np.ndarray,
        point_colors: Optional[np.ndarray] = None,
        title: str = "Point Cloud",
    ) -> None:
        """
        Simple point cloud visualization without planes.

        Args:
            points: Point cloud (N, 3)
            point_colors: Optional custom colors (N, 3)
            title: Window title
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window(title, width=1200, height=800)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if point_colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(point_colors)

        vis.add_geometry(pcd)

        opt = vis.get_render_option()
        opt.background_color = np.array([1.0, 1.0, 1.0])
        opt.point_size = 3.0

        vis.run()
        vis.destroy_window()

    def _draw_plane_normals(self, planes: List[PlaneResult]) -> None:
        """
        Draw plane normals as arrows for better visualization.

        Args:
            planes: List of detected planes
        """
        for plane in planes:
            # Get a point on the plane (centroid of inliers)
            # This is approximate - in practice you'd need point data
            origin = np.array([0, 0, 0])

            # Draw arrow from origin along normal
            end = origin + 0.5 * plane.normal
            line_points = o3d.utility.Vector3dVector([origin, end])
            line_colors = o3d.utility.Vector3dVector(
                [plane.color, plane.color]
            )

            line_set = o3d.geometry.LineSet()
            line_set.points = line_points
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.colors = line_colors

            self.vis.add_geometry(line_set)

    @staticmethod
    def create_mesh_from_plane(
        center: np.ndarray,
        normal: np.ndarray,
        size: float = 1.0,
    ) -> o3d.geometry.TriangleMesh:
        """
        Create a simple mesh representation of a plane.

        Args:
            center: Center point of the plane
            normal: Normal vector of the plane
            size: Size of the mesh

        Returns:
            Open3D TriangleMesh
        """
        # Create a square mesh perpendicular to the normal
        mesh = o3d.geometry.TriangleMesh.create_box(
            width=size, height=size, depth=0.01
        )

        # Rotate mesh to align with normal
        # This is simplified - full rotation requires additional computation
        mesh.translate(center)

        return mesh


def create_synthetic_scene(
    num_points: int = 5000,
    num_planes: int = 2,
    noise_level: float = 0.05,
) -> np.ndarray:
    """
    Create synthetic point cloud scene with multiple planes.

    Args:
        num_points: Total number of points
        num_planes: Number of planes to generate
        noise_level: Gaussian noise standard deviation

    Returns:
        Point cloud array (N, 3)
    """
    points = []

    for plane_id in range(num_planes):
        plane_points = num_points // num_planes

        # Generate points on plane
        # Plane at height z = plane_id
        x = np.random.uniform(-5, 5, plane_points)
        y = np.random.uniform(-5, 5, plane_points)
        z = np.ones(plane_points) * (plane_id * 2)

        # Add noise
        x += np.random.normal(0, noise_level, plane_points)
        y += np.random.normal(0, noise_level, plane_points)
        z += np.random.normal(0, noise_level, plane_points)

        plane_cloud = np.column_stack([x, y, z])
        points.append(plane_cloud)

    return np.vstack(points)
