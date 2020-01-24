# Changelog

## master

- `Candidate` class is removed, and is completely replaced by `Parameter` [#459](https://github.com/facebookresearch/nevergrad/pull/459)
- New parametrization is now as efficient as in v0.3.0 (see CHANGELOG for v0.3.1 for contect)
- `CandidateMaker` (`optimizer.create_candidate`) raises `DeprecationWarning`s since it new candidates/parameters
  can be straightforwardly created (`parameter.spawn_child(new_value=new_value)`)
- Optimizers can now hold any parametrization, not just `Instrumentation`. This for instance mean that when you
  do `OptimizerClass(instrumentation=12, budget=100)`, the instrumentation (and therefore the candidates) will be of class
  `ng.p.Array` (and not `ng.p.Instrumentation`), and their attribute `value` will be the corresponding `np.ndarray` value.
  You can still use `args` and `kwargs` if you want, but it's no more needed!
- Old `instrumentation` classes now raise deprecation warnings, and will disappear in versions >0.3.2.
  Hence, prefere using parameters from `ng.p` than `ng.var`, and avoid using `ng.Instrumentation` altogether if
  you don't need it anymore (or import it through `ng.p.Instrumentation`)

## v0.3.1 (2019-01-23)

**Note**: this is the first step to propagate the instrumentation/parametrization framework.
 Learn more on the [Facebook user group](https://www.facebook.com/notes/nevergrad-users/moving-to-new-parametrization-upcoming-unstability-and-breaking-changes/639090766861215/).
 If you are looking for stability, await for version 0.4.0, but the intermediary releases will help by providing
 deprecation warnings.

### Breaking changes

- `FolderFunction` must now be accessed through `nevergrad.parametrization.FolderFunction`
- Instrumentation names are changed (possibly breaking for benchmarks records)

### Other changes

- Old instrumentation classes now all inherits from the new parametrization classes [#391](https://github.com/facebookresearch/nevergrad/pull/391). Both systems coexists, but optimizers
  use the old API at this point (it will use the new one in version 0.3.2).
- Temporary performance loss is expected in orded to keep compatibility between `Variable` and `Parameter` frameworks.
- `PSO` now uses initialization by sampling the parametrization, instead of sampling all the real space. A new `WidePSO`
 optimizer was created, using the previous initial sampling method [#467](https://github.com/facebookresearch/nevergrad/pull/467).

## v0.3.0 (2019-01-08)

**Note**: this version is stable, but the following versions will include breaking changes which may cause instability. The aim of this changes will be to update the instrumentation system for more flexibility. See PR #323 and [Fb user group](https://www.facebook.com/groups/nevergradusers/) for more information.

### Breaking changes

- `Instrumentation` is now a `Variable` for simplicity and flexibility. The `Variable` API has therefore heavily changed,
  and bigger changes are coming (`instrumentation` will become `parametrization` with a different API). This should only impact custom-made variables.
- `InstrumentedFunction` has been aggressively deprecated to solve bugs and simplify code, in favor of using the `Instrumentation` directly at the optimizer initialization,
  and of using `ExperimentFunction` to define functions to be used in benchmarks. Main differences are:
  * `instrumentation` attribute is renamed to `parametrization` for forward compatibility.
  *  `__init__` takes exactly two arguments (main function and parametrization/instrumentation) and
  * calls to `__call__` is directly forwarded to the main function (instead of converting from data space),
- `Candidates` have now a `uid` instead of a `uuid` for compatibility reasons.
- Update archive `keys/items_as_array` methods to `keys/items_as_arrays` for consistency.

### Other changes

- Benchmark plots now show confidence area (using partially transparent lines).
- `Chaining` optimizer family enables chaining of algorithms.
- Cleaner installation.
- New simplified `Log` variable for log-distributed scalars.
- Cheap constraints can now be provided through the `Instrumentation`
- Added preliminary multiobjective function support (may be buggy for the time being, and API will change)
- New callback for dumping parameters and loss, and loading them back easily for display (display yet to come).
- Added a new parametrization module which is expected to soon replace the instrumentation module.
- Added new test cases: games, power system, etc (experimental)
- Added new algorithms: quasi-opposite one shot optimizers

## v0.2.2

### Breaking changes

- instrumentations now hold a `random_state` attribute which can be seeded (`optimizer.instrumentation.random_state.seed(12)`).
  Seeding `numpy`'s global random state seed **before** using the instrumentation still works (but if not, this change can break reproducibility).
  The random state is used by the optimizers through the `optimizer._rng` property.

### Other changes

- added a `Scalar` variable as a shortcut to `Array(1).asscalar(dtype)` to simplify specifying instrumentation.
- added `suggest` method to optimizers in order to manually provide the next `Candidate` from the `ask` method (experimental feature, name and behavior may change).
- populated `nevergrad`'s namespace so that `import nevergrad as ng` gives access to `ng.Instrumentation`, `ng.var` and `ng.optimizers`. The
  `optimizers` namespace is quite messy, some non-optimizer objects will eventually be removed from there.
- renamed `optimize` to `minimize` to be more explicit. Using `optimize` will raise a `DeprecationWarning` for the time being.
- added first game-oriented testbed function in the `functions.rl` module. This is still experimental and will require refactoring before the API becomes stable.

## v0.2.1

### Breaking changes

- changed `tanh` to `arctan` as default for bounded variables (much wider range).
- changed cumulative Gaussian density to `arctan` for rescaling in `BO` (much wider range).
- renamed `Array.asfloat` method to `Array.asscalar` and allow casting to `int` as well through an argument.

### Other changes

- fixed `tell_not_asked` for `DE` family of optimizers.
- added `dump` and `load` method to `Optimizer`.
- Added warnings against inefficient settings: `BO` algorithms with dis-continuous or noisy instrumentations
  without appropriate parametrization, `PSO` and `DE` for low budget.
- improved benchmark plots legend.

## v0.2.0

### Breaking changes

- first parameter of optimizers is now `instrumentation` instead of `dimension`. This allows the optimizer
  to have information on the underlying structure. `int`s are still allowed as before and will set the instrumentation
  to the `Instrumentation(var.Array(n))` (which is basically the identity).
- removed `BaseFunction` in favor of `InstrumentedFunction` and use instrumentation instead of
  defining specific transforms (breaking change for benchmark function implementation).
- `ask()` and `provide_recommendation()` now return a `Candidate` with attributes `args`, `kwargs` (depending on the instrumentation)
  and `data` (the array which was formerly returned). `tell` must now receive this candidate as well instead of
  the array.
- removed `tell_not_asked` in favor of `tell`. A new `num_tell_not_asked` attribute is added to check the number of `tell` calls with non-asked points.

### Other changes

- updated `bayesion-optimization` version to 1.0.1.
- from now on, optimizers should preferably implement `_internal_ask_candidate` and `_internal_tell_candidate` instead of `_internal_ask`
  and `_internal_tell`. This should take at most one more line: `x = candidate.data`.
- added an `_asked` private attribute to register uuid of particuels that were asked for.
- solved `ArtificialFunction` delay bug.

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
