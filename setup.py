#!/usr/bin/env python
# For installing the Peach package
import torch
from os import path
from setuptools import find_packages, setup

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 7], "Requires PyTorch >= 1.7"

# Get version from Peach/__init__.py
def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "AutoVQE", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")
    return version

# Install Peach package
setup(
    name="AutoVQE",
    version=get_version(),
    author="MilkTea",
    packages=find_packages(exclude=("configs", "tests*")),
    # package_dir=PROJECTS,
    python_requires=">=3.6",
    install_requires=[
        # These dependencies are not pure-python.
        # In general, avoid adding more dependencies like them because they are not
        # guaranteed to be installable by `pip install` on all platforms.
        # To tell if a package is pure-python, go to https://pypi.org/project/{name}/#files
        "timm",
        "einops",
    ],
)