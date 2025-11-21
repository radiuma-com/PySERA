#!/usr/bin/env python3

from setuptools import setup, find_packages

# Read the README file for long description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "pysera - A Python library for radiomics feature extraction with multiprocessing support"


# Read requirements from requirements.txt
def read_requirements(filename="requirements-library.txt"):
    """Read requirements from requirements.txt, ignoring comments and blank lines."""
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]


setup(
    name="pysera",
    version="2.1.5",
    author="Dr. Mohammad R. Salmanpour, A.H. Pouria",
    author_email="M.salmanpoor66@gmail.com",
    description="pysera (Python-based Standardized Extraction for Radiomics Analysis) is a comprehensive Python library for radiomics feature extraction from medical imaging data. It provides a simple, single-function API with built-in multiprocessing support and comprehensive report capabilities.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/radiuma-com/PySERA",
    packages=find_packages(),
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "pysera=pysera.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "pysera": ["config/*.yaml", "config/*.json"],
    },
    keywords="medical-imaging Standardized-Radiomics-Feature-Extraction Quantitative-Analysis IBSI_Evaluation-Standardization "
             "healthcare",
    project_urls={
        "Bug Reports": "https://github.com/radiuma-com/PySERA/issues",
        "Source": "https://github.com/radiuma-com/PySERA",
        "Documentation": "https://pysera.readthedocs.io/",
    },
)
