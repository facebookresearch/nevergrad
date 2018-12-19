#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# from distutils.core import setup
from setuptools import setup


with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(name='nevergrad',
      version='0.1.0',
      description='Gradient-free optimization toolbox',
      author='Facebook AI Research',
      packages=['nevergrad'],
      install_requires=requirements,)
