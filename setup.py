#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
import os
import sys
import typing as tp
from pathlib import Path
from setuptools import setup
from setuptools import find_packages
from setuptools.command.install import install


# read requirements

requirements: tp.Dict[str, tp.List[str]] = {}
for extra in ["dev", "bench", "main"]:
    requirements[extra] = Path(f"requirements/{extra}.txt").read_text().splitlines()


# build long description

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def _replace_relative_links(regex: tp.Match[str]) -> str:
    """Converts relative links into links to master"""
    string = regex.group()
    link = regex.group("link")
    name = regex.group("name")
    if not link.startswith("http") and Path(link).exists():
        githuburl = (
            "github.com/facebookresearch/nevergrad/blob/master"
            if not link.endswith((".png", ".gif"))
            else "raw.githubusercontent.com/facebookresearch/nevergrad/master"
        )
        string = f"[{name}](https://{githuburl}/{link})"
    return string


pattern = re.compile(r"\[(?P<name>.+?)\]\((?P<link>\S+?)\)")
long_description = re.sub(pattern, _replace_relative_links, long_description)


# find version

init_str = Path("nevergrad/__init__.py").read_text()
match = re.search(r"^__version__ = \"(?P<version>[\w\.]+?)\"$", init_str, re.MULTILINE)
assert match is not None, "Could not find version in nevergrad/__init__.py"
version = match.group("version")


class VerifyCircleCiVersionCommand(install):  # type: ignore
    """Custom command to verify that the git tag matches CircleCI version"""

    description = "verify that the git tag matches CircleCI version"

    def run(self) -> None:
        tag = os.getenv("CIRCLE_TAG")
        if tag != version:
            info = f"Git tag: {tag} does not match the version of this app: {version}"
            sys.exit(info)


# setup
setup(
    name="nevergrad",
    version=version,
    license="MIT",
    description="A Python toolbox for performing gradient-free optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Facebook AI Research",
    url="https://github.com/facebookresearch/nevergrad",
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python",
    ],
    install_requires=requirements["main"],
    extras_require={
        "all": requirements["dev"] + requirements["bench"],
        "dev": requirements["dev"],
        "benchmark": requirements["bench"],
    },
    package_data={"nevergrad": ["py.typed", "*.csv", "*.py"]},
    python_requires=">=3.6",
    cmdclass={"verify_circleci_version": VerifyCircleCiVersionCommand},
)
