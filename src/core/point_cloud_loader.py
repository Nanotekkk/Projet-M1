"""Point Cloud Loader - Factory Pattern

Author: Matheo LANCEA
Description: Factory pattern implementation for creating and loading 3D point clouds
"""

import numpy as np
from typing import Union, Tuple, Optional
from abc import ABC, abstractmethod
from pathlib import Path


class PointCloud:
    """Wrapper class for point cloud data"""

    def __init__(self, points: np.ndarray, colors: Optional[np.ndarray] = None):
        self.points = points
        self.colors = colors
        self.normals = None

    def __len__(self) -> int:
        return len(self.points)

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get min and max bounds of the point cloud"""
        return self.points.min(axis=0), self.points.max(axis=0)

    def get_center(self) -> np.ndarray:
        """Get the center of the point cloud"""
        return self.points.mean(axis=0)

    def downsample(self, voxel_size: float) -> "PointCloud":
        """Downsample point cloud using voxel grid"""
        # Voxel grid downsampling using numpy
        min_bounds = self.points.min(axis=0)
        max_bounds = self.points.max(axis=0)
        
        # Create grid indices
        grid_indices = ((self.points - min_bounds) / voxel_size).astype(int)
        
        # Find unique voxels and keep one point per voxel
        unique_voxels, inverse_indices = np.unique(
            grid_indices, axis=0, return_inverse=True
        )
        
        # Average points in each voxel
        downsampled_points = []
        downsampled_colors = []
        
        for i in range(len(unique_voxels)):
            mask = inverse_indices == i
            downsampled_points.append(self.points[mask].mean(axis=0))
            if self.colors is not None:
                downsampled_colors.append(self.colors[mask].mean(axis=0))
        
        points = np.array(downsampled_points)
        colors = np.array(downsampled_colors) if downsampled_colors else None
        return PointCloud(points, colors)


class PointCloudLoader:
    """Factory for loading point clouds from various formats"""

    @staticmethod
    def load_from_file(filepath: Union[str, Path]) -> PointCloud:
        """Load point cloud from file (PLY, PCD, XYZ, etc.)"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Point cloud file not found: {filepath}")

        suffix = filepath.suffix.lower()
        
        if suffix == ".xyz":
            points = np.loadtxt(filepath)
            return PointCloud(points)
        elif suffix == ".ply":
            points, colors = PointCloudLoader._load_ply(filepath)
            return PointCloud(points, colors)
        elif suffix == ".pcd":
            points, colors = PointCloudLoader._load_pcd(filepath)
            return PointCloud(points, colors)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    @staticmethod
    def _load_ply(filepath: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load PLY file without open3d"""
        points = []
        colors = []
        has_color = False
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            in_header = True
            num_vertices = 0
            
            for line in lines:
                line = line.strip()
                if line == "end_header":
                    in_header = False
                    break
                if line.startswith("element vertex"):
                    num_vertices = int(line.split()[-1])
                if "red" in line or "green" in line:
                    has_color = True
            
            # Read vertex data
            vertex_idx = 0
            for line in lines:
                if vertex_idx >= num_vertices:
                    break
                line = line.strip()
                if not in_header and line and not line.startswith("element"):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                            if has_color and len(parts) >= 6:
                                colors.append([float(parts[3])/255, float(parts[4])/255, float(parts[5])/255])
                            vertex_idx += 1
                        except ValueError:
                            pass
        
        return np.array(points), np.array(colors) if colors else None

    @staticmethod
    def _load_pcd(filepath: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load PCD file without open3d"""
        points = []
        colors = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            data_start = 0
            
            for i, line in enumerate(lines):
                if line.startswith("POINTS"):
                    num_points = int(line.split()[-1])
                if line.startswith("DATA"):
                    data_start = i + 1
                    break
            
            # Read point data
            for i in range(data_start, len(lines)):
                parts = lines[i].strip().split()
                if len(parts) >= 3:
                    try:
                        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        if len(parts) >= 6:
                            colors.append([float(parts[3])/255, float(parts[4])/255, float(parts[5])/255])
                    except ValueError:
                        pass
        
        return np.array(points), np.array(colors) if colors else None

    @staticmethod
    def create_synthetic_plane(
        width: float = 100,
        depth: float = 100,
        num_points: int = 10000,
        noise: float = 0.1,
    ) -> PointCloud:
        """Create a synthetic plane point cloud for testing"""
        x = np.random.uniform(-width / 2, width / 2, num_points)
        y = np.random.uniform(-depth / 2, depth / 2, num_points)
        z = np.random.normal(0, noise, num_points)

        points = np.column_stack([x, y, z])
        return PointCloud(points)

    @staticmethod
    def create_synthetic_scene(
        num_ground: int = 5000,
        num_obstacles: int = 2000,
        ground_width: float = 100,
    ) -> PointCloud:
        """Create a synthetic scene with ground and obstacles"""
        # Ground plane
        x_ground = np.random.uniform(-ground_width / 2, ground_width / 2, num_ground)
        y_ground = np.random.uniform(-ground_width / 2, ground_width / 2, num_ground)
        z_ground = np.random.normal(0, 0.1, num_ground)
        ground_points = np.column_stack([x_ground, y_ground, z_ground])

        # Obstacles (cubes)
        cube_centers = np.array([[20, 20, 2], [-20, 30, 2], [0, -25, 2]])
        obstacle_points = []
        for center in cube_centers:
            size = 3
            n_points = num_obstacles // len(cube_centers)
            cube_x = np.random.uniform(center[0] - size / 2, center[0] + size / 2, n_points)
            cube_y = np.random.uniform(center[1] - size / 2, center[1] + size / 2, n_points)
            cube_z = np.random.uniform(center[2] - size / 2, center[2] + size / 2, n_points)
            cube = np.column_stack([cube_x, cube_y, cube_z])
            obstacle_points.append(cube)

        all_points = np.vstack([ground_points] + obstacle_points)
        return PointCloud(all_points)
