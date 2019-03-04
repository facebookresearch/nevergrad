#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from setuptools import find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
# assuming there is either "  # extra: " if it is an extra or nothing
requirements_with_extra = [x.split("  # extra: ") for x in requirements]


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name='nevergrad',
    version='0.1.4',
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
    data_files=[('', ['LICENSE', 'requirements.txt']),
                ('nevergrad', ["nevergrad/benchmark/additional/example.py",
                               "nevergrad/instrumentation/examples/script.py"])],
    install_requires=[x[0] for x in requirements_with_extra if len(x) == 1],
    extras_require={"all": [x[0] for x in requirements_with_extra if len(x) == 2],
                    "benchmark": [x[0] for x in requirements_with_extra if len(x) == 2 and x[1] == "benchmark"]}
)
