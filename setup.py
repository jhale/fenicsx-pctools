import re

from setuptools import setup, find_packages

VERSION = re.findall('__version__ = "(.*)"', open("fenics_pctools/__init__.py", "r").read())[0]

REQUIREMENTS = [
    "matplotlib",
    "pandas",
    "dolfiny@git+https://github.com/jhale/dolfiny.git@jhale/fix-gmsh-tools#egg=dolfiny"
]

REQUIREMENTS_dev = [
    "black",
]

setup(
    name="fenics_pctools",
    description="FEniCS Preconditioning Tools",
    version=VERSION,
    python_requires=">=3.7",
    authors="Martin Řehoř, Jack S. Hale",
    author_email="martin.rehor@rafinex.com",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    extras_require={"dev": REQUIREMENTS_dev},
    # include_package_data=True,  # no MANIFEST.in at the moment
    zip_safe=False,
)
