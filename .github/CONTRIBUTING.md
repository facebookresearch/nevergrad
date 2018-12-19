# Contributing to nevergrad
We want to make contributing to this project as easy and transparent as
possible.

## Our Development Process

We use pre-commit hooks to make sure the code follows the same coding style. We currently use autpep8 and pylint. To install them, just run `pre-commit install` once, and they will be activated for all your commits on this repository.

Most of the code is covered by tests, run `nosetests nevergrad` to run all the tests. Type hints can also be checked with `mypy --ignore-missing-imports --strict nevergrad`.


## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style  
We use pep8, but allow lines to be as long as 140 characters.
Please use the pre-commit hooks to ensure correctness (see section "Our Development Process").

## License
By contributing to `nevergrad`, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
