import pathlib
from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="yaml-kit",
    version="0.0.3",
    description="A YAML-based tool for managing config files for ML experiments.",  # noqa
    long_description=long_description,
    packages=find_packages(),
)
