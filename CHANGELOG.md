# Changelog

## master

### Breaking changes

- `copy()` method of a `Parameter` does not change the parameters's random state anymore (it used to reset it to `None` [#1048](https://github.com/facebookresearch/nevergrad/pull/1048)
- `MultiobjectiveFunction` does not exist anymore  [#1034](https://github.com/facebookresearch/nevergrad/pull/1034).
- `Choice` and `TransitionChoice` have some of their API changed for uniformization. In particular, `indices` is now an
  `ng.p.Array` (and not an `np.ndarray`) which contains the selected indices (or index) of the `Choice`. The sampling is
  performed by specific "layers" that are applied to `Data` parameters [#1065](https://github.com/facebookresearch/nevergrad/pull/1065).
- `Parameter.set_standardized_space` does not take a `deterministic` parameter anymore
  [#1068](https://github.com/facebookresearch/nevergrad/pull/1068).  This is replaced by the more
  general `with ng.p.helpers.determistic_sampling(parameter)` context. One-shot algorithms are also updated to choose
  options of `Choice` parameters deterministically, since it is a simpler behavior to expect compared to sampling the
  standardized space than sampling the option stochastically from there
- `RandomSearch` now defaults to sample values using the `parameter.sample()` instead of a Gaussian
   [#1068](https://github.com/facebookresearch/nevergrad/pull/1068).  The only difference comes with bounded
  variables since in this case `parameter.sample()` samples uniformly (unless otherwise specified).
  The previous behavior can be obtained with `RandomSearchMaker(sampler="gaussian")`.
- `PSO` API has been slightly changed [#1073](https://github.com/facebookresearch/nevergrad/pull/1073)
- `Parameter` instances `descriptor` attribute is deprecated, in favor of a combinaison of an analysis function
  (`ng.p.helpers.analyze`) returning information about the parameter (eg: whether continuous, deterministic etc...)
  and a new `function` attribute which can be used to provide information about the function (eg: whether deterministic etc)
  [#1076](https://github.com/facebookresearch/nevergrad/pull/1076).
- Half the budget alloted to solve cheap constrained is now used by a sub-optimizer
  [#1047](https://github.com/facebookresearch/nevergrad/pull/1047). More changes of constraint management will land
  in the near future.
- Experimental methods `Array.set_recombination` and `Array.set_mutation(custom=.)` are removed in favor of
  layers changing `Array` behaviors [#1086](https://github.com/facebookresearch/nevergrad/pull/1086).
  Caution: this is still very experimental (and undocumented).

### Important changes

- `Parameter` classes are undergoing heavy changes, please open an issue if you encounter any problem.
  The midterm aim is to allow for simpler constraint management.
- `Parameter` have been updated  have undergone heavy changes to ease the handling of their tree structure (
  [#1029](https://github.com/facebookresearch/nevergrad/pull/1029)
  [#1036](https://github.com/facebookresearch/nevergrad/pull/1036)
  [#1038](https://github.com/facebookresearch/nevergrad/pull/1038)
  [#1043](https://github.com/facebookresearch/nevergrad/pull/1043)
  [#1044](https://github.com/facebookresearch/nevergrad/pull/1044)
  )
- `Parameter` classes have now a layer structure [#1045](https://github.com/facebookresearch/nevergrad/pull/1045)
  which simplifies changing their behavior. In future PRs this system will take charge of bounds, other constraints,
  sampling etc.
- The layer structures allows disentangling bounds and log-distribution. This goal has been reached with
  [#1053](https://github.com/facebookresearch/nevergrad/pull/1053) but may create some instabilities. In particular,
  the representation (`__repr__`) of `Array` has changed, and their `bounds` attribute is no longer reliable for now.
  This change will eventually lead to a new syntax for settings bounds and distribution, but it's not ready yet.
- `DE` initial sampling as been updated to take bounds into accounts [#1058](https://github.com/facebookresearch/nevergrad/pull/1058)
- `Array` can now take `lower` and `upper` bounds as initialization arguments. The array is initialized at its average
  if not `init` is provided and both bounds are provided. In this case, sampling will be uniformm between these bounds.


### Other changes

- the new `nevergrad.errors` module gathers errors and warnings used throughout the package (WIP) [#1031](https://github.com/facebookresearch/nevergrad/pull/1031).
- `EvolutionStrategy` now defaults to NSGA2 selection in the multiobjective case
- A new experimental callback adds an early stopping mechanism
  [#1054](https://github.com/facebookresearch/nevergrad/pull/1054).
- `Choice`-like parameters now accept integers are inputs instead of a list, as a shortcut for `range(num)`
  [#1106](https://github.com/facebookresearch/nevergrad/pull/1106).

## 0.4.3 (2021-01-28)

### Important changes

- `tell` method can now receive a list/array of losses for multi-objective optimization [#775](https://github.com/facebookresearch/nevergrad/pull/775). For now it is neither robust, nor scalable, nor stable, nor optimal so be careful when using it. More information in the [documentation](https://facebookresearch.github.io/nevergrad/optimization.html#multiobjective-minimization-with-nevergrad).
- The old way to perform multiobjective optimization, through the use of :code:`MultiobjectiveFunction`, is now deprecated and will be removed after version 0.4.3 [#1017](https://github.com/facebookresearch/nevergrad/pull/1017).
- By default, the optimizer now returns the best set of parameter as recommendation [#951](https://github.com/facebookresearch/nevergrad/pull/951), considering that the function is deterministic. The previous behavior would use an estimation of noise to provide the pessimistic best point, leading to unexpected behaviors [#947](https://github.com/facebookresearch/nevergrad/pull/947). You can can back to this behavior by specifying: :code:`parametrization.descriptors.deterministic_function = False`

### Other

- `DE` and its variants have been updated to make full use of the multi-objective losses [#789](https://github.com/facebookresearch/nevergrad/pull/789). Other optimizers convert multiobjective problems to a volume minimization, which is not always as efficient.
- as an **experimental** feature we have added some preliminary support for constraint management through penalties.
  From then on the prefered option for penalty is to register a function returning a positive float when the constraint is satisfied.
  While we will wait fore more testing before documenting it, this may already cause instabilities and errors when adding cheap constraints.
  Please open an issue if you encounter a problem.
- `tell` argument `value` is renamed to `loss` for clarification [#774](https://github.com/facebookresearch/nevergrad/pull/774). This can be breaking when using named arguments!
- `ExperimentFunction` now automatically records arguments used for their instantiation so that they can both be used to create a new copy, and as descriptors if there are of type  int/bool/float/str [#914](https://github.com/facebookresearch/nevergrad/pull/914 [#914](https://github.com/facebookresearch/nevergrad/pull/914)).
- from now on, code formatting needs to be [`black`](https://black.readthedocs.io/en/stable/) compliant. This is
  simply performed by running `black nevergrad`. A continuous integration checks that PRs are compliant, and the
  precommit hooks have been adapted. For PRs branching from an old master, you can run `black --line-length=110 nevergrad/<path_to_modified_file>` to make your code easier to merge.
- Pruning has been patched to make sure it is not activated too often upon convergence [#1014](https://github.com/facebookresearch/nevergrad/pull/1014). The bug used to lead to important slowdown when reaching near convergence.

## 0.4.2 (2020-08-04)

- `recommend` now provides an evaluated candidate when possible. For non-deterministic parametrization like `Choice`, this means we won't resample, and we will actually recommend the best past evaluated candidate [#668](https://github.com/facebookresearch/nevergrad/pull/668).  Still, some optimizers (like `TBPSA`) may recommend a non-evaluated point.
- `Choice` and `TransitionChoice` can now take a `repetitions` parameters for sampling several times, it is equivalent to :code:`Tuple(*[Choice(options) for _ in range(repetitions)])` but can be be up to 30x faster for large numbers of repetitions [#670](https://github.com/facebookresearch/nevergrad/pull/670) [#696](https://github.com/facebookresearch/nevergrad/pull/696).
- Defaults for bounds in `Array` is now `bouncing`, which is a variant of `clipping` avoiding over-sompling on the bounds [#684](https://github.com/facebookresearch/nevergrad/pull/684) and [#691](https://github.com/facebookresearch/nevergrad/pull/691).

This version should be robust. Following versions may become more unstable as we will add more native multiobjective optimization as an **experimental** feature. We also are in the process of simplifying the naming pattern for the "NGO/Shiwa" type optimizers which may cause some changes in the future.

## 0.4.1 (2020-05-07)

- `Archive` now stores the best corresponding candidate. This requires twice the memory compared to before the change. [#594](https://github.com/facebookresearch/nevergrad/pull/594)
- `Parameter` now holds a `loss: Optional[float]` attribute which is set and used by optimizers after the `tell` method.
- Quasi-random samplers (`LHSSearch`, `HammersleySearch`, `HaltonSearch` etc...) now sample in the full range of bounded variables when the `full_range_sampling` is `True` [#598](https://github.com/facebookresearch/nevergrad/pull/598). This required some ugly hacks, help is most welcome to find nices solutions.
- `full_range_sampling` is activated by default if both range are provided in `Array.set_bounds`.
- Propagate parametrization system features (generation tracking, ...) to `OnePlusOne` based algorithms [#599](https://github.com/facebookresearch/nevergrad/pull/599).
- Moved the `Selector` dataframe overlay so that basic requirements do not include `pandas` (only necessary for benchmarks) [#609](https://github.com/facebookresearch/nevergrad/pull/609)
- Changed the version name pattern (removed the `v`) to unify with `pypi` versions. Expect more frequent intermediary versions to be pushed (deployment has now been made pseudo-automatic).
- Started implementing more ML-oriented testbeds [#642](https://github.com/facebookresearch/nevergrad/pull/642)


## v0.4.0 (2020-03-09)

### Breaking and important changes

- Removed all deprecated code [#499](https://github.com/facebookresearch/nevergrad/pull/499). That includes:
  - `instrumentation` as init parameter of an `Optimizer` (replaced by `parametrization`)
  - `instrumentation` as attribute of an `Optimizer` (replaced by `parametrization`)
  - `candidate_maker` (not needed anymore)
  - `optimize` methods of `Optimizer` (renamed to `minimize`)
  - all the `instrumentation` subpackage (replaced by `parametrization`) and its legacy methods (`set_cheap_constraint_checker` etc)
- Removed `ParametrizedOptimizer` and `OptimizerFamily` in favor of `ConfiguredOptimizer` with simpler usage [#518](https://github.com/facebookresearch/nevergrad/pull/518) [#521](https://github.com/facebookresearch/nevergrad/pull/521).
- Some variants of algorithms have been removed from the `ng.optimizers` namespace to simplify it. All such variants can be easily created
  using the corresponding `ConfiguredOptimizer`. Also, adding `import nevergrad.optimization.experimentalvariants` will populate `ng.optimizers.registry`
  with all variants, and they are all available for benchmarks [#528](https://github.com/facebookresearch/nevergrad/pull/528).
- Renamed `a_min` and `a_max` in `Array`, `Scalar` and `Log` parameters for clarity.
  Using old names will raise a deprecation warning for the time being.
- `archive` is pruned much more often (eg.: for `num_workers=1`, usually pruned to 100 elements when reaching 1000),
  so you should not rely on it for storing all results, use a callback instead [#571](https://github.com/facebookresearch/nevergrad/pull/571).
  If this is a problem for you, let us know why and we'll find a solution!

### Other changes

- Propagate parametrization system features (generation tracking, ...) to `TBPSA`, `PSO` and `EDA` based algorithms.
- Rewrote multiobjective core system [#484](https://github.com/facebookresearch/nevergrad/pull/484).
- Activated Windows CI (still a bit flaky, with a few deactivated tests).
- Better callbacks in `np.callbacks`, including exporting to [`hiplot`](https://github.com/facebookresearch/hiplot).
- Activated [documentation](https://facebookresearch.github.io/nevergrad/) on github pages.
- Scalar now takes optional `lower` and `upper` bounds at initialization, and `sigma` (and optionnally `init`)
  if is automatically set to a sensible default [#536](https://github.com/facebookresearch/nevergrad/pull/536).


## v0.3.2 (2020-02-05)


### Breaking changes (possibly for next version)

- Fist argument of optimizers is renamed to `parametrization` instead of `instrumentation` for consistency [#497](https://github.com/facebookresearch/nevergrad/pull/497). There is currently a deprecation warning, but this will be breaking in v0.4.0.
- Old `instrumentation` classes now raise deprecation warnings, and will disappear in versions >0.3.2.
  Hence, prefere using parameters from `ng.p` than `ng.var`, and avoid using `ng.Instrumentation` altogether if
  you don't need it anymore (or import it through `ng.p.Instrumentation`).
- `CandidateMaker` (`optimizer.create_candidate`) raises `DeprecationWarning`s since it new candidates/parameters
  can be straightforwardly created (`parameter.spawn_child(new_value=new_value)`)
- `Candidate` class is completely removed, and is completely replaced by `Parameter` [#459](https://github.com/facebookresearch/nevergrad/pull/459).
  This should not break existing code since `Parameter` can be straightforwardly used as a `Candidate`.

### Other changes

- New parametrization is now as efficient as in v0.3.0 (see CHANGELOG for v0.3.1 for contect)
- Optimizers can now hold any parametrization, not just `Instrumentation`. This for instance mean that when you
  do `OptimizerClass(instrumentation=12, budget=100)`, the instrumentation (and therefore the candidates) will be of class
  `ng.p.Array` (and not `ng.p.Instrumentation`), and their attribute `value` will be the corresponding `np.ndarray` value.
  You can still use `args` and `kwargs` if you want, but it's no more needed!
- Added *experimental* evolution-strategy-like algorithms using new parametrization [#471](https://github.com/facebookresearch/nevergrad/pull/471)
  (the behavior and API of these optimizers will probably evolve in the near future).
- `DE` algorithms comply with the new parametrization system and can be set to use parameter's recombination.
- Fixed array as bounds in `Array` parameters

## v0.3.1 (2020-01-23)

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

## v0.3.0 (2020-01-08)

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
