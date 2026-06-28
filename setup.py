from setuptools import setup, find_packages

setup(
    name="nfsw_detector",
    version="1.0.0",
    description="Multimodal Harmful Content Detection and Alert System for Short Video Platforms",
    author="NFSW Detector Team",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "torchvision",
        "torchaudio",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "pyyaml",
        "decord",
        "gradio",
        "plotly",
        "imageio-ffmpeg",
        "opencv-python",
        "tensorboard",
    ],
    entry_points={
        "console_scripts": [
            "nfsw-detect=main:main",
        ],
    },
)
