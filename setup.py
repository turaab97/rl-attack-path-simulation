"""
setup.py – MMAI 845 RL Attack-Path Simulation
"""

from setuptools import find_packages, setup

setup(
    name="rl-attack-path-simulation",
    version="0.1.0",
    description=(
        "RL-based attack path simulation against enterprise AI infrastructure "
        "using NASim, PPO, and DQN (MMAI 845 Final Project)"
    ),
    author="Syed Ali Turab",
    author_email="info@turab.sh",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    install_requires=[
        "nasim>=0.3.0",
        "stable-baselines3[extra]>=2.3.0",
        "gymnasium>=0.29.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "tensorboard>=2.14.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=24.0.0",
            "isort>=5.13.0",
            "flake8>=7.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "train-agent=training.train:main",
            "eval-agent=training.evaluate:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
