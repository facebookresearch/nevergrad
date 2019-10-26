# Contributing to nevergrad
We want to make contributing to this project as easy and transparent as possible.

## Our Development Process

To install `nevergrad` in development mode (if you wish to contribute to it), clone the repository and run `pip install -e '.[all]'` from inside the repository folder.

Most of the code is covered by unit tests. You can run them with:
```
pytest nevergrad --cov=nevergrad
```

For type checking, please install `numpy-stubs` (git clone it then `pip install .`). You can then run `mypy` on `nevergrad` with:
```
mypy --implicit-reexport --strict nevergrad
```
You can however omit the `--strict` mode and/or the installation of the `numpy` stubs for a simplified version of the checks.
Indeed, if you are not familiar with type checking, we do not want it to be an annoyance and you can can even ignore errors by adding `# type: ignore` at the end of lines flagged as incorrect.
If we consider it useful to have correct typing, we will update the code after your pull request is merged.

Unit tests and type checks (in non-strict mode) will be automatically run every time a pull request is submitted/updated.

Finally, we use pre-commit hooks to make sure the code follows the same coding style. We currently use  `autpep8` and `pylint`. To install them, just run `pre-commit install` once, and they will be activated for all your commits on this repository.

## Guidelines

We have added some specific guidelines on how to [add a new algorithm](../docs/adding_an_algorithm.md) which can help you find your way in the structure of the optimization subpackage.

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

## Documentation

Documentation can be build with `make html` from the `docs` folder.

## License
By contributing to `nevergrad`, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
