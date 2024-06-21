# setup.py
from setuptools import setup, find_packages

setup(
    name="mcising",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "imageio"
    ],
    tests_require=[
        "unittest",
        "pytest"
    ],
    entry_points={
        'console_scripts': [
            'generate_ising_data=mcising.ising_data_generate:main',
        ],
    },
    author="Burak Ç.",
    author_email="bcivitcioglu@gmail.com",
    description="A package for generating Ising model data using Metropolis algorithm on a square lattice for nearest neighbor and next nearest neighbor interactions.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/bcivitcioglu/mcising",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
