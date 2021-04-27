import re

from setuptools import setup, find_packages

VERSION = re.findall('__version__ = "(.*)"', open("fenics_pctools/__init__.py", "r").read())[0]

REQUIREMENTS = [
    "matplotlib>=3.3.4",
    "pandas>=1.1.2",
    "dolfiny @ git+https://github.com/michalhabera/dolfiny.git@bba2a8b#egg=dolfiny",
]

REQUIREMENTS_dev = [
    "black>=19.10b0",
]

setup(
    name="fenics_pctools",
    description="FEniCS Preconditioning Tools",
    version=VERSION,
    python_requires=">=3.8.5",
    author="Martin Řehoř",
    author_email="martin.rehor@rafinex.com",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    extras_require={"dev": REQUIREMENTS_dev},
    # include_package_data=True,  # no MANIFEST.in at the moment
    zip_safe=False,
)
