# Changelog

## master

- Removed `BaseFunction` in favor of `InstrumentedFunction` and use instrumentation instead of
  defining specific transforms (breaking change for benchmark function implementation).

## v0.1.6

- corrected a bug introduced by v0.1.5 for `PSO`.
- activated `tell_not_ask` for `PSO`, `TBPSA` and differential evolution algorithms.
- added a pruning mechanisms for optimizers archive in order to avoid using a huge amount of memory.
- corrected typing after activating `numpy-stubs`.

## v0.1.5

- provided different install procedures for optimization, benchmark and dev (requirements differ).
- added an experimental `tell_not_asked` method to optimizers.
- switched to `pytest` for testing, and removed dependency to `nosetests` and `genty`.
- made archive more memory efficient by using bytes as key instead of tuple of floats.
- started rewritting some optimizers as instance of a family of optimizers (experimental).
- added pseudotime in benchmarks for both steady mode and batch mode.
- made the whole chain from `Optimizer` to `BenchmarkChunk` stateful and able to restart from where it was stopped.
- started introducing `tell_not_asked` method (experimental).

## v0.1.4

- fixed `PSO` in asynchronous case
- started refactoring `instrumentation` in depth, and more specifically instantiation of external code (breaking change)
- Added Photonics and ARCoating test functions
- Added variants of algorithms

## v0.1.3

- multiple bug fixes
- multiple typo corrections (including modules changing names)
- added MLDA functions
- allowed steady state in experiments
- allowed custom file types for external code instantiation
- added dissymetric noise case to `ArtificialFunction`
- prepared an `Instrumentation` class to simplify instrumentation (breaking changes will come)
- added new algorithms and benchmarks
- improved plotting
- added a transform method to `BaseFunction` (more breaking changes will come)

Work on `instrumentation` will continue and breaking changes will be pushed in the following versions.

## v0.1.0

Initial version
