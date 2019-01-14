#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from setuptools import find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='nevergrad',
    version='0.1.1',
    license='MIT',
    description='A Python toolbox for performing gradient-free optimization',
    long_description=long_description,
    author='Facebook AI Research',
    url="https://github.com/facebookresearch/nevergrad",
    packages=find_packages(),
    classifiers=['License :: OSI Approved :: MIT License',
                 'Intended Audience :: Science/Research',
                 'Topic :: Scientific/Engineering',
                 'Programming Language :: Python'],
    install_requires=requirements,
)
