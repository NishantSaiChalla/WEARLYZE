"""Setup script for Fashion Detection System."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fashion-detection",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive fashion detection system using YOLOv8 and CLIP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fashion-detection",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "fashion-detect=inference.cli:main",
            "fashion-train=training.train:main",
            "fashion-evaluate=evaluation.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "fashion_detection": ["config/*.yaml", "config/*.yml"],
    },
)