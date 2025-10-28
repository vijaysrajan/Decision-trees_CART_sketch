"""
Setup script for theta-sketch-tree package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
with open(readme_file, "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="theta-sketch-tree",
    version="0.1.0-dev",
    author="Vijay Sankar Rajan",
    author_email="vijay.sankar.rajan@gmail.com",
    description="CART Decision Tree trained on theta sketches with binary feature inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vijaysrajan/Decision-trees_CART_sketch",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "pyyaml>=5.4.0",
        "datasketches>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "viz": [
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
