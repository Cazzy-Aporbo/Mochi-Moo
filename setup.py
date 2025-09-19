"""
setup.py - Installation configuration for Mochi-Moo
Author: Cazandra Aporbo MS
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="mochi-moo",
    version="1.0.0",
    author="Cazandra Aporbo MS",
    author_email="becaziam@gmail.com",
    description="A superintelligent assistant who dreams in matte rainbow and thinks in ten-dimensional pastel origami",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Cazzy-Aporbo/Mochi-Moo",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "pillow>=10.0.0",
        "colormath>=3.0.0",
        "webcolors>=1.13",
        "aiohttp>=3.8.0",
        "asyncio>=3.4.3",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "python-multipart>=0.0.6",
        "httpx>=0.24.0",
        "websockets>=11.0.0",
        "redis>=4.6.0",
        "pymongo>=4.4.0",
        "sqlalchemy>=2.0.0",
        "alembic>=1.11.0",
        "cryptography>=41.0.0",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.0",
        "marshmallow>=3.20.0",
        "click>=8.1.0",
        "rich>=13.5.0",
        "tqdm>=4.65.0",
        "loguru>=0.7.0",
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
        "black>=23.7.0",
        "isort>=5.12.0",
        "mypy>=1.4.0",
        "flake8>=6.0.0",
        "pre-commit>=3.3.0"
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipython>=8.14.0",
            "notebook>=7.0.0",
            "jupyterlab>=4.0.0"
        ],
        "ml": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "sentence-transformers>=2.2.0",
            "langchain>=0.0.200",
            "openai>=0.27.0",
            "tiktoken>=0.4.0"
        ],
        "audio": [
            "whisper>=1.0.0",
            "soundfile>=0.12.0",
            "librosa>=0.10.0",
            "pyaudio>=0.2.13"
        ],
        "vision": [
            "opencv-python>=4.8.0",
            "scikit-image>=0.21.0",
            "imageio>=2.31.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "mochi=mochi_moo.cli:main",
            "mochi-server=mochi_moo.server:run_server",
        ],
    },
    include_package_data=True,
    package_data={
        "mochi_moo": [
            "data/*.json",
            "assets/*.png",
            "templates/*.html",
            "static/css/*.css",
            "static/js/*.js"
        ],
    },
    zip_safe=False,
)

# ===== requirements.txt =====
# Core dependencies for Mochi-Moo
# Author: Cazandra Aporbo MS
# Generated: 2025

# Core numerical and scientific computing
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Visualization and color management
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
pillow>=10.0.0
colormath>=3.0.0
webcolors>=1.13

# Async and web framework
aiohttp>=3.8.0
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
python-multipart>=0.0.6
httpx>=0.24.0
websockets>=11.0.0

# Database and caching
redis>=4.6.0
pymongo>=4.4.0
sqlalchemy>=2.0.0
alembic>=1.11.0

# Security and authentication
cryptography>=41.0.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-dotenv>=1.0.0

# Configuration and serialization
pyyaml>=6.0.0
marshmallow>=3.20.0

# CLI and utilities
click>=8.1.0
rich>=13.5.0
tqdm>=4.65.0
loguru>=0.7.0

# Testing and development
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
black>=23.7.0
isort>=5.12.0
mypy>=1.4.0
flake8>=6.0.0
pre-commit>=3.3.0
