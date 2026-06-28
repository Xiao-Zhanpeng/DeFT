# setup.py — DeFT package
from setuptools import setup, find_packages

setup(
    name="deft",
    version="0.1.0",
    description="DeFT: Descriptor-Forked Test-Time Adaptation for Medical Image Denoising",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.23,<2.0",
        "scipy>=1.10.0",
        "scikit-image>=0.20.0",
        "monai>=1.3.0",
        "tqdm>=4.64.0",
        "matplotlib>=3.5.0",
        "pandas>=1.4.0",
        "imageio>=2.25.0",
    ],
)
