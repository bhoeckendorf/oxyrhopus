from setuptools import setup, find_packages

setup(
    name="oxyrhopus",
    version="0.0.1",
    description="",
    long_description="",
    author="Burkhard Hoeckendorf",
    author_email="burkhard.hoeckendorf@pm.me",
    url="https://github.com/bhoeckendorf/oxyrhopus",
    license="Apache License Version 2.0",
    packages=find_packages(exclude=("tests*", "docs*")),
    install_requires=["hydra-core", "pytorch", "torchvision"]
)
