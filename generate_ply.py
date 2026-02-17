"""Generate random PLY files for testing

This script generates PLY point cloud files with various geometric patterns
useful for testing RANSAC plane detection.
"""

import numpy as np
from pathlib import Path
import argparse
from typing import Tuple


def generate_random_ply(
    output_file: str,
    num_points: int = 5000,
    pattern: str = "multi_plane",
    noise_level: float = 0.1,
    add_colors: bool = False,
) -> None:
    """
    Generate a random PLY file.
    
    Args:
        output_file: Path to save the PLY file
        num_points: Total number of points to generate
        pattern: Type of pattern ('random', 'single_plane', 'multi_plane', 'sphere', 'cube')
        noise_level: Amount of noise to add (0.0 - 1.0)
        add_colors: Whether to add random RGB colors
    """
    
    if pattern == "random":
        points = np.random.uniform(-10, 10, (num_points, 3))
        
    elif pattern == "single_plane":
        x = np.random.uniform(-10, 10, num_points)
        y = np.random.uniform(-10, 10, num_points)
        z = np.random.normal(0, noise_level, num_points)
        points = np.column_stack([x, y, z])
        
    elif pattern == "multi_plane":
        # Plane 1 (floor)
        plane1_pts = int(num_points * 0.4)
        x1 = np.random.uniform(-10, 10, plane1_pts)
        y1 = np.random.uniform(-10, 10, plane1_pts)
        z1 = np.random.normal(0, noise_level, plane1_pts)
        plane1 = np.column_stack([x1, y1, z1])
        
        # Plane 2 (wall 1)
        plane2_pts = int(num_points * 0.3)
        x2 = np.ones(plane2_pts) * 10
        y2 = np.random.uniform(-10, 10, plane2_pts)
        z2 = np.random.uniform(0, 5, plane2_pts)
        plane2 = np.column_stack([x2, y2, z2])
        
        # Plane 3 (wall 2)
        plane3_pts = int(num_points * 0.3)
        x3 = np.random.uniform(-10, 10, plane3_pts)
        y3 = np.ones(plane3_pts) * 10
        z3 = np.random.uniform(0, 5, plane3_pts)
        plane3 = np.column_stack([x3, y3, z3])
        
        points = np.vstack([plane1, plane2, plane3])
        
    elif pattern == "sphere":
        # Generate points on a sphere surface
        theta = np.random.uniform(0, 2*np.pi, num_points)
        phi = np.random.uniform(0, np.pi, num_points)
        r = 10 + np.random.normal(0, noise_level, num_points)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        points = np.column_stack([x, y, z])
        
    elif pattern == "cube":
        # Random points inside a cube
        x = np.random.uniform(-5, 5, num_points)
        y = np.random.uniform(-5, 5, num_points)
        z = np.random.uniform(-5, 5, num_points)
        points = np.column_stack([x, y, z])
        
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    # Ensure we have exactly num_points
    if len(points) > num_points:
        points = points[:num_points]
    elif len(points) < num_points:
        # Pad with random points if needed
        padding = num_points - len(points)
        extra_points = np.random.uniform(-10, 10, (padding, 3))
        points = np.vstack([points, extra_points])
    
    # Generate colors if requested
    if add_colors:
        colors = np.random.randint(0, 256, (len(points), 3), dtype=np.uint8)
    else:
        colors = None
    
    # Write PLY file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        
        f.write("end_header\n")
        
        # Write vertex data
        for i, point in enumerate(points):
            if colors is not None:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                       f"{colors[i][0]} {colors[i][1]} {colors[i][2]}\n")
            else:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    
    print(f"✓ Generated PLY file: {output_path}")
    print(f"  Points: {len(points)}")
    print(f"  Pattern: {pattern}")
    print(f"  Colors: {colors is not None}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Generate random PLY files for testing")
    parser.add_argument("output", nargs="?", default="random_test.ply",
                       help="Output PLY file path (default: random_test.ply)")
    parser.add_argument("-n", "--num-points", type=int, default=5000,
                       help="Number of points (default: 5000)")
    parser.add_argument("-p", "--pattern", 
                       choices=["random", "single_plane", "multi_plane", "sphere", "cube"],
                       default="multi_plane",
                       help="Point pattern (default: multi_plane)")
    parser.add_argument("--noise", type=float, default=0.1,
                       help="Noise level (default: 0.1)")
    parser.add_argument("--colors", action="store_true",
                       help="Add random RGB colors")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("PLY FILE GENERATOR")
    print("="*80)
    
    generate_random_ply(
        output_file=args.output,
        num_points=args.num_points,
        pattern=args.pattern,
        noise_level=args.noise,
        add_colors=args.colors,
    )
    
    print("\n" + "="*80)
    print("USAGE WITH MAIN.PY")
    print("="*80)
    print(f"\nTo use this file with RANSAC plane detection:")
    print(f"  python main.py")
    print(f"  > Select option: 2 (Load PLY file)")
    print(f"  > Enter path: {args.output}")
    print(f"  > Select: 1 (Run all demos) or 2 (RANSAC only)")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
