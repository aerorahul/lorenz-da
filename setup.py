#!/usr/bin/env python
"""
Setup script for Lorenz Data Assimilation Framework
"""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='lorenzda',
    version='1.0.0',
    author='Rahul Mahajan',
    author_email='rahul.mahajan@nasa.gov',
    description='Data Assimilation Framework for Lorenz Models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/username/lorenz-da',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.16.0',
        'scipy>=1.2.0',
        'matplotlib>=3.0.0',
        'netCDF4>=1.4.0',
    ],
    entry_points={
        'console_scripts': [
            'lorenzda=lorenzda.cli:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
