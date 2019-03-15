#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List
from setuptools import setup
from setuptools import find_packages

requirements: Dict[str, List[str]] = {}
for name in ["dev", "bench", "main"]:
    with open(f'requirements/{name}.txt') as f:
        requirements[name] = f.read().splitlines()


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name='nevergrad',
    version='0.1.6',
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
    data_files=[('', ['LICENSE', 'requirements/main.txt', 'requirements/dev.txt', 'requirements/bench.txt']),
                ('nevergrad', ["nevergrad/benchmark/additional/example.py",
                               "nevergrad/instrumentation/examples/script.py"])],
    install_requires=requirements["main"],
    extras_require={"all": [x for reqs in requirements.values() for x in reqs],
                    "dev": requirements["main"] + requirements["dev"],
                    "benchmark": requirements["main"] + requirements["bench"]}
)
