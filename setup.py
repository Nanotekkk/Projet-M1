from setuptools import setup, find_packages

setup(
    name="lidar-vr-navigation",
    version="1.0.0",
    description="Intelligent navigation system for 3D point clouds in VR environments",
    author="Matheo LANCEA",
    author_email="matheo.lancea@lacatholille.fr",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "open3d>=0.14.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "tensorflow>=2.8.0",
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "matplotlib>=3.4.0",
        "pydantic>=1.8.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "pytest-cov>=3.0.0"],
    },
)
