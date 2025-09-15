#!/usr/bin/env python3
"""Setup script for nanoVLA"""

import os
import re
from pathlib import Path
from setuptools import setup, find_packages

# Read version from package
def get_version():
    init_file = Path(__file__).parent / "nanovla" / "src" / "nanovla" / "__init__.py"
    with open(init_file, "r", encoding="utf-8") as f:
        content = f.read()
        version_match = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", content)
        if version_match:
            return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

# Read long description from README
def get_long_description():
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        with open(readme_file, "r", encoding="utf-8") as f:
            return f.read()
    return "nanoVLA: A minimal Vision-Language-Action framework"

# Read requirements
def get_requirements(req_file="requirements.txt"):
    req_path = Path(__file__).parent / req_file
    if req_path.exists():
        with open(req_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

setup(
    name="nanovla",
    version=get_version(),
    author="nanoVLA Team",
    author_email="team@nanovla.ai",
    description="A minimal Vision-Language-Action framework",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/nanovla/nanovla",
    
    # Package discovery
    package_dir={"": "nanovla/src"},
    packages=find_packages(where="nanovla/src"),
    include_package_data=True,
    
    # Python version requirement
    python_requires=">=3.9",
    
    # Dependencies
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "Pillow>=8.0.0",
        "numpy>=1.20.0",
        "typing-extensions>=4.0.0",
    ],
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
            "mypy",
        ],
        "vision": [
            "torchvision>=0.15.0",
            "opencv-python",
            "albumentations",
        ],
        "robotics": [
            "gym>=0.21.0",
            "pybullet",
            "mujoco",
        ],
        "training": [
            "wandb",
            "tensorboard",
            "accelerate",
            "datasets",
        ],
        "rl": [
            "verl",
            "ray[default]",
        ],
        "all": [
            "pytest>=6.0", "pytest-cov", "black", "isort", "flake8", "mypy",
            "torchvision>=0.15.0", "opencv-python", "albumentations",
            "gym>=0.21.0", "pybullet", "mujoco",
            "wandb", "tensorboard", "accelerate", "datasets",
        ],
    },
    
    # Entry points
    entry_points={
        "console_scripts": [
            "nanovla-train=nanovla.training.cli:main",
            "nanovla-eval=nanovla.evaluation.cli:main",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Keywords
    keywords="vision language action robotics vlm vla machine-learning",
    
    # Project URLs
    project_urls={
        "Homepage": "https://github.com/nanovla/nanovla",
        "Documentation": "https://nanovla.readthedocs.io/",
        "Repository": "https://github.com/nanovla/nanovla",
        "Bug Tracker": "https://github.com/nanovla/nanovla/issues",
    },
    
    # Package data
    package_data={
        "nanovla": [
            "py.typed",
            "**/*.json",
            "**/*.yaml",
            "**/*.yml",
        ],
    },
    
    # Zip safe
    zip_safe=False,
)
