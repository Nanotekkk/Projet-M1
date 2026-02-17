# LiDAR RANSAC Plane Detection

**Author:** Matheo LANCEA  
**Project:** M1 University Project  
**Status:** ✅ Active

A comprehensive Python implementation focusing on **RANSAC-based multi-plane detection** with advanced visualization in Open3D and comparison with other plane fitting methods.

---

## 🎯 Project Objectives

### Core Features
- ✅ **Multi-plane detection** using RANSAC algorithm
- ✅ **Color-coded plane visualization** in Open3D
- ✅ **Method comparison**: RANSAC vs Linear Regression, K-Means
- ✅ **PLY file loading** support for real point cloud data
- ✅ **Synthetic scene generation** for testing
- ✅ **Performance metrics** and computation time analysis
- ✅ **Realistic indoor scene** detection (floor, walls, ceiling)

---

## 🚀 Quick Start

### Installation

```bash
# Navigate to project
cd Projet-M1

# Create virtual environment (optional but recommended)
python -m venv venv
# On Windows: venv\Scripts\activate
# On Linux/Mac: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Run Demonstrations

```bash
# Interactive mode - choose data source and demos
python main.py

# Then select:
# 1. Generate synthetic data or Load PLY file
# 2. Run all 4 demos or RANSAC only
```

**Demos include:**
1. **RANSAC Multi-Plane Detection** - Detects 3 planes with color coding
2. **Method Comparison** - Compares RANSAC vs Linear Regression vs K-Means
3. **Visualization Comparison** - View results from each method in Open3D
4. **Realistic Indoor Scene** - Floor + walls + ceiling detection

---

## 📊 Detection Methods

### Comparison of 3 Methods

| Method | Pros | Cons | Speed |
|--------|------|------|-------|
| **RANSAC** | ✓ Robust to outliers | Requires parameter tuning | Medium |
| | ✓ Detects all planes | Iterative approach | |
| **Linear Regression** | ✓ Very fast | ✗ Sensitive to outliers | Very Fast |
| | ✓ Simple to implement | ✗ Assumes z=f(x,y) | |
| **K-Means** | ✓ Unsupervised learning | ✗ Must specify k clusters | Fast |
| | ✓ Global segmentation | ✗ Different results per run | |

### Algorithm Details

#### RANSAC (RANdom SAmple Consensus)
- Iteratively samples 3 random points
- Fits plane using SVD
- Counts inliers within distance threshold
- Keeps plane with most inliers
- Removes inliers and repeats

#### Linear Regression
- Fits plane to equation: z = ax + by + c
- Uses scikit-learn LinearRegression
- Fast and simple
- Best for relatively flat surfaces

#### K-Means
- Clusters points into k groups
- Selects cluster with most points as plane
- Fits plane to selected cluster
- Good for multi-plane scenes

---

## 🎨 Color-Coded Planes

Each detected plane gets a unique color:

| Color | RGB | Plane ID |
|-------|-----|----------|
| 🔴 Red | (255, 0, 0) | 0 |
| 🟢 Green | (0, 255, 0) | 1 |
| 🔵 Blue | (0, 0, 255) | 2 |
| 🟡 Yellow | (255, 255, 0) | 3 |
| 🟣 Magenta | (255, 0, 255) | 4 |
| 🔷 Cyan | (0, 255, 255) | 5 |
| 🟠 Orange | (255, 165, 0) | 6+ |

Unassigned points appear in light gray.

---

## 📁 Project Structure

```
Projet-M1/
├── src/
│   ├── plane_detection/
│   │   ├── __init__.py
│   │   ├── ransac_detector.py      # RANSAC multi-plane detection
│   │   └── model_comparison.py     # 3-method comparison framework
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── open3d_visualizer.py    # Open3D visualization
│   │
│   └── core/
│       ├── __init__.py
│       └── point_cloud_loader.py   # Load PLY/PCD/XYZ files
│
├── tests/
│   ├── unit/
│   │   ├── test_ransac.py
│   │   └── test_model_comparison.py
│   └── integration/
│       └── test_e2e.py
│
├── main.py                          # Interactive demonstration
├── generate_ply.py                  # Generate random PLY files
├── test_ply_loading.py              # Test PLY loading
├── requirements.txt
└── README.md
```

---

## 💻 Usage Examples

### Example 1: Basic RANSAC Detection

```python
import numpy as np
from src.plane_detection.ransac_detector import RANSACPlaneDetector
from src.visualization.open3d_visualizer import Open3DVisualizer

# Create sample point cloud
points = np.random.randn(10000, 3)

# Detect planes
detector = RANSACPlaneDetector(
    distance_threshold=0.15,
    iterations=1000,
    max_planes=5
)
planes = detector.detect_planes(points)

# Visualize
visualizer = Open3DVisualizer("My Point Cloud")
visualizer.visualize_point_cloud_with_planes(points, planes)

# Print results
for plane in planes:
    print(f"Plane {plane.plane_id}:")
    print(f"  Normal: {plane.normal}")
    print(f"  Inliers: {plane.inlier_count}")
    print(f"  Color: {plane.color}")
```

### Example 2: Compare Methods

```python
from src.plane_detection.model_comparison import ModelComparison
import numpy as np

# Create point cloud
points = np.random.randn(5000, 3)

# Compare methods
comparator = ModelComparison()
results = comparator.compare_all_methods(points)

# Print comparison table
comparator.print_comparison(results)

# Access individual results
for method_name, result in results.items():
    print(f"\n{method_name}:")
    print(f"  Inliers: {result.inlier_count}")
    print(f"  Ratio: {result.inlier_ratio:.2%}")
    print(f"  Time: {result.computation_time*1000:.2f} ms")
    print(f"  Normal: {result.plane_normal}")
```

### Example 3: Load PLY File

```python
from src.core.point_cloud_loader import PointCloudLoader
from src.plane_detection.ransac_detector import RANSACPlaneDetector

# Load PLY file
point_cloud = PointCloudLoader.load_from_file("my_data.ply")
points = point_cloud.points

# Detect planes
detector = RANSACPlaneDetector()
planes = detector.detect_planes(points)

print(f"Loaded {len(points)} points")
print(f"Detected {len(planes)} planes")
```

### Example 4: Generate Random PLY Files

```bash
# Generate multi-plane PLY with 10000 points
python generate_ply.py my_scene.ply -n 10000 -p multi_plane --colors

# Generate sphere
python generate_ply.py sphere.ply -n 5000 -p sphere

# Generate cube
python generate_ply.py cube.ply -n 5000 -p cube
```

---

## 🔧 Configuration

### RANSAC Parameters

```python
detector = RANSACPlaneDetector(
    distance_threshold=0.15,       # Max distance to plane (inlier threshold)
    iterations=1000,               # Number of RANSAC iterations
    min_points_per_plane=50,       # Minimum points to define a plane
    max_planes=5,                  # Maximum planes to detect
    inlier_ratio_threshold=0.05    # Minimum inlier ratio (5%)
)
```

**Tips:**
- Lower `distance_threshold` for cleaner results (more strict)
- Increase `iterations` for more robust detection
- Adjust `max_planes` based on expected scene complexity
- Use `inlier_ratio_threshold` to filter out noise

---

## � PLY Generation Guide

### Generate Various Shapes

Use `generate_ply.py` to create test data with different geometric patterns:

#### 1. Single Plane (Flat Surface)
```bash
python generate_ply.py single_plane.ply -p single_plane -n 5000
```
**Use case:** Test basic plane detection on flat ground  
**Features:** 5000 points on a flat surface with optional noise

#### 2. Multi-Plane (Indoor Scene)
```bash
python generate_ply.py multi_plane.ply -p multi_plane -n 10000 --colors
```
**Use case:** Test RANSAC with multiple planes (floor + 2 walls)  
**Features:**
- Floor: 4000 points at z ≈ 0
- Wall 1: 3000 points at x ≈ 10 (vertical)
- Wall 2: 3000 points at y ≈ 10 (vertical)
- Optional RGB colors for better visualization

#### 3. Sphere (3D Surface)
```bash
python generate_ply.py sphere.ply -p sphere -n 5000
```
**Use case:** Test robustness to curved surfaces (no flat planes)  
**Features:**
- Points distributed on sphere surface
- Radius: 10 units
- Good for testing false positive rejection

#### 4. Cube (Random Volume)
```bash
python generate_ply.py cube.ply -p cube -n 5000
```
**Use case:** Random points in 3D space  
**Features:**
- Points randomly distributed inside cubic volume
- Box dimensions: -5 to +5 in all axes
- Ideal for general clustering tests

#### 5. Fully Random Points
```bash
python generate_ply.py random.ply -p random -n 5000
```
**Use case:** Test outlier robustness  
**Features:**
- Completely random point distribution
- Range: -10 to +10 in all axes
- No structure for stress testing

### Command Options Reference

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| Output file | Filename to save | `random_test.ply` | `scene.ply` |
| `-n, --num-points` | Number of points | 5000 | `-n 10000` |
| `-p, --pattern` | Shape type | multi_plane | `-p sphere` |
| `--noise` | Noise level 0.0-1.0 | 0.1 | `--noise 0.05` |
| `--colors` | Add RGB colors | No | `--colors` |

### Advanced Examples

**Large realistic scene:**
```bash
python generate_ply.py office.ply -p multi_plane -n 50000 --colors --noise 0.05
```
Creates: 50,000 points, multiple planes, with colors, minimal noise

**Noisy sphere for robust testing:**
```bash
python generate_ply.py noisy_sphere.ply -p sphere -n 10000 --noise 0.3
```
Creates: Sphere with heavy noise (30%)

**Clean simple plane:**
```bash
python generate_ply.py clean_floor.ply -p single_plane -n 3000 --noise 0.01
```
Creates: Very clean single plane (1% noise)

**Generate all test types:**
```bash
python generate_ply.py floor.ply -p single_plane -n 5000
python generate_ply.py building.ply -p multi_plane -n 10000 --colors
python generate_ply.py sphere.ply -p sphere -n 8000
python generate_ply.py box.ply -p cube -n 6000
python generate_ply.py cloud.ply -p random -n 7000
```

**Batch generation (Bash):**
```bash
for pattern in single_plane multi_plane sphere cube random; do
    python generate_ply.py test_${pattern}.ply -p $pattern -n 5000
done
```

### Using Generated Files

1. **Interactive mode:**
```bash
python main.py
# Select: 2 (Load PLY file)
# Enter: multi_plane.ply
# Start detection!
```

2. **Python script:**
```python
from src.core.point_cloud_loader import PointCloudLoader
from src.plane_detection.ransac_detector import RANSACPlaneDetector

# Load PLY
pc = PointCloudLoader.load_from_file("sphere.ply")
detector = RANSACPlaneDetector(max_planes=3)
planes = detector.detect_planes(pc.points)
print(f"Found {len(planes)} planes in {len(pc.points)} points")
```

3. **Programmatic generation:**
```python
from generate_ply import generate_random_ply

# Generate multiple types
generate_random_ply("test_plane.ply", num_points=5000, pattern="single_plane")
generate_random_ply("test_room.ply", num_points=15000, pattern="multi_plane", add_colors=True)
generate_random_ply("test_sphere.ply", num_points=8000, pattern="sphere", noise_level=0.2)
generate_random_ply("test_cube.ply", num_points=10000, pattern="cube")
```

### PLY File Format

Generated PLY files support:
- ASCII format (text-based, human-readable)
- Optional RGB color data (0-255 per channel)
- XYZ coordinates (float precision)
- Compatible with Open3D, CloudCompare, Meshlab

Example PLY header:
```
ply
format ascii 1.0
element vertex 5000
property float x
property float y
property float z
property uchar red      (optional if --colors)
property uchar green    (optional if --colors)
property uchar blue     (optional if --colors)
end_header
```

---

## �📈 Performance Metrics

Each method returns:
- **Inlier Count**: Number of points on detected plane
- **Inlier Ratio**: Percentage of points (0.0 - 1.0)
- **Plane Normal**: Unit normal vector (nx, ny, nz)
- **Plane Distance**: Distance from origin (d in equation)
- **Computation Time**: Execution time in seconds
- **Additional Metrics**: Method-specific data

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_ransac.py -v

# Run model comparison tests
pytest tests/unit/test_model_comparison.py -v

# Run with coverage report
pytest --cov=src tests/
```

---

## 📝 Real-World Example

```python
# Indoor scene with floor, walls, ceiling

from src.visualization.open3d_visualizer import create_synthetic_scene
from src.plane_detection.ransac_detector import RANSACPlaneDetector
from src.visualization.open3d_visualizer import Open3DVisualizer

# Create realistic scene
points = create_synthetic_scene(
    num_points=15000,
    num_planes=3,
    noise_level=0.1
)

# Detect with RANSAC
detector = RANSACPlaneDetector(
    distance_threshold=0.2,
    iterations=500,
    max_planes=6
)
planes = detector.detect_planes(points)

# Visualize
viz = Open3DVisualizer("Indoor Scene")
viz.visualize_point_cloud_with_planes(points, planes)

# Print summary
print(f"\n{'='*50}")
print(f"Scene Analysis")
print(f"{'='*50}")
print(f"Total points: {len(points)}")
print(f"Planes detected: {len(planes)}")
for plane in planes:
    coverage = 100 * plane.inlier_count / len(points)
    print(f"  Plane {plane.plane_id}: {plane.inlier_count} inliers ({coverage:.1f}%)")
```

---

## ⚡ Performance Tips

1. **Improve Speed:**
   - Use Linear Regression for quick results
   - Downsample large point clouds
   - Reduce RANSAC iterations

2. **Improve Quality:**
   - Increase RANSAC iterations
   - Fine-tune distance_threshold
   - Remove outliers first

3. **Better Visualization:**
   - Adjust point size in visualizer
   - Use background color for contrast
   - Normalize plane normals

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Open3D window doesn't appear | Install Open3D: `pip install open3d` |
| Planes not detected | Increase iterations or lower threshold |
| Slow performance | Downsample points or use Linear Regression |
| PLY file not loading | Check file format (must be ASCII or binary PLY) |
| Wrong plane normals | Check if normals are normalized |

---

## 📚 Dependencies

```
numpy>=1.24.0          # Numerical computing
scikit-learn>=1.3.0    # Machine learning
scipy>=1.11.0          # Scientific computing
open3d>=0.17.0         # 3D visualization
pandas>=2.0.0          # Data analysis
```

---

## 📖 References

- **RANSAC**: Fischler & Bolles (1981) - "Random Sample Consensus: A Paradigm for Model Fitting"
- **Open3D**: Zhou et al. - "Open3D: A Modern Library for 3D Data Processing"
- **SVD**: Linear Algebra - Singular Value Decomposition for plane fitting
- **Linear Regression**: Least squares plane fitting

---

## 📄 License

**Educational Project** - Free to use and modify

---

## 👨‍💻 Author

**Matheo LANCEA**  
M1 University Project  
February 2026

---

## 🔗 Quick Links

- **Start Here**: See [START_HERE.txt](START_HERE.txt)
- **Generate PLY**: `python generate_ply.py --help`
- **Run Tests**: `pytest -v`
- **Quick Start**: `python main.py`

---

**Last Updated:** February 17, 2026  
**Version:** 2.0  
**Status:** Active Development ✅


