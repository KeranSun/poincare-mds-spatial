from setuptools import setup, find_packages

setup(
    name="poincare-mds",
    version="0.2.0",
    description="Hyperbolic embedding for spatial transcriptomics using Poincaré MDS",
    long_description=open("poincare_mds/README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Keran Sun",
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19",
        "torch>=1.9",
        "geoopt>=0.5.0",
        "scipy>=1.5",
        "scikit-learn>=0.24",
    ],
    extras_require={
        "scanpy": ["scanpy>=1.8"],
        "dev": ["pytest>=6.0"],
        "all": ["scanpy>=1.8", "matplotlib>=3.3", "pandas>=1.0"],
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
