# Adding an algorithm

These guidelines are for people who want to add an algorithm to `nevergrad`. Feel free to update them if you find them unclear or think they should evolve.

## Where to add the algorithm?

All optimizers are implemented in the `ng.optimization` subpackage, and all optimizer classes are available either in the `ng.optimization.optimizerlib` module (which is aliased to `ng.optimizers`, or through the optimizer registry: `ng.optimizers.registry`.

Implementations are however spread into several files:
- [optimizerlib.py](../nevergrad/optimization/optimizerlib.py): this is the default file, where most algorithms are implemented. It also imports optimizers from all other files.
- [oneshot.py](../nevergrad/optimization/oneshot.py): this is where one-shot optimizers are implemented
- [differentialevolution.py](../nevergrad/optimization/differentialevolution.py): this is where evolutionary algorithms are implemteted.
- [recastlib.py](../nevergrad/optimization/recastlib.py): this is where we implement ask & tell versions of existing Python implementations which do not follow this pattern. The underlying class which helps spawn a subprocess to run the existing implementation into is in [recaster.py](../nevergrad/optimization/recaster.py). Hopefully, you won't need this.

If you implement one new algorithm and if this algorithm is not one-shot/evolutionary/recast, you should implement it into [optimizerlib.py](../nevergrad/optimization/optimizerlib.py). If you implement a whole family of algorithms, you are welcome to create a new corresponding file.
Still, this structure is not final, it is bound to evolve and you are welcome to amend it.


## How to implement it?

### Base class features

All algorithms derive from a base class named `Optimizer` and are registered through a decorator. The implementation of the base class is [here](../nevergrad/optimization/base.py).
This base class implements the `ask` and `tell` interface.

It records all evaluated points through the `archive` attribute of class `Archive`. It can be seen be used as if it was of type `Dict[np.ndarray, Value]`, but since `np.ndarray` are not hashable, the underlying implementation converts arrays into bytes and register them into the `archive.bytesdict` dictionary. `Archive` however does not implement `keys` and `items` methods because converting from bytes to array is not very efficient, one should therefore interate on `bytesdict` and the keys can then be transformed back to arrays using `np.frombuffer(key)`. See [OnePlusOne implementation](../nevergrad/optimization/optimizerlib.py) for an example.


The key tuple if the point location, and `Value` is a class with attributes:
- `count`: number of evaluations at this point.
- `mean`: mean value of the evaluations at this point.
- `variance`: variance of the evaluations at this point.

For more details, see the implementation in [utils.py](../nevergrad/optimization/utils.py).

Through the archive, you can therefore access most useful information about past evaluations. A pruning mechanism makes sure this archive does
not grow too much. This pruning can be tuned through the `pruning` attribute of the optimizer (the default is very conservative).

The base `Optimizer` class also tracks the best optimistic and pessimistic points through the `current_bests` attribute which is of type:
```
Dict[str, Point]
```
The key string is either `optimistic` or `pessimistic`, and the `Point` value is a `Value` with an additional `x` attribute, recording the location of the point.

### Methods and attributes

4 methods are designed to be overriden:
- `__init__`: for the initialization of your algorithm
- `_internal_ask_candidate`: to fetch the next point to be evaluated. This function is the only one that is absolutely required to be overriden. The default `ask` method calls this method (please do not override the default `ask`).
- `_internal_tell_candidate`: to update your algorithm with the new point. The default `tell` method calls this internal method after updating the archive (see paragraph above), please do not override it.
- `_internal_provide_recommendation`: to provide the final recommendation. By default, the recommendation is the pessimistic best point.
- `_internal_tell_not_asked` (optional): if the optimizer must handle points differently if they were not asked for, this method must be implemented. If you do not want to support this, you can raise `base.TellNotAskedNotSupportedError`. A unit test will make sure that the optimizer either accepts the point or raises this error.

These functions work with `Candidate` instances, which hold the data point, and `args` and `kwargs` depending on the instrumentation provided at the initilization of the optimizer. Such instances can be conveniently created through the `create_candidate` instance of each optimizer. This object creates `Candidate` object in 3 ways: `opt.create_candidate(args, kwargs, data)`, `opt.create_candidate.from_arguments(*args, **kwargs)` and `opt.create_candidate.from_data(data)`. The last one is probably the one you will need to use inside the `_internal_ask_candidate` method.


These functions work with `Candidate` instances, which hold the data point, and `args` and `kwargs` depending on the instrumentation provided at the initilization of the optimizer. Such instances can be conveniently created through the `create_candidate` instance of each optimizer. This object creates `Candidate` object in 3 ways: `opt.create_candidate(args, kwargs, data)`, `opt.create_candidate.from_arguments(*args, **kwargs)` and `opt.create_candidate.from_data(data)`. The last one is probably the one you will need to use inside the `_internal_ask_candidate` method.



These functions work with `Candidate` instances, which hold the data point, and `args` and `kwargs` depending on the instrumentation provided at the initilization of the optimizer. Such instances can be conveniently created through the `create_candidate` instance of each optimizer. This object creates `Candidate` object in 3 ways: `opt.create_candidate(args, kwargs, data)`, `opt.create_candidate.from_arguments(*args, **kwargs)` and `opt.create_candidate.from_data(data)`. The last one is probably the one you will need to use inside the `_internal_ask_candidate` method.



If the algorithm is not able to handle parallelization (if `ask` cannot be called multiple times consecutively), the `no_parallelization` **class attribute** must be set to `True`.



### Seeding

Seeding has an important part for the significance and reproducibility of the algorithm benchmarking. We want to ensure the following constraints:
- we expect stochastic algorithms to be actually stochastic, if we set a hard seed inside the implementation this assumption is broken.
- we need the randomness to obtain relevant statistics when benchmarking the algorithms on deterministic functions.
- we should be able to seed from **outside** when we need it: we expect that setting a seed to the global random state should lead to
reproducible results.

In order to facilitate these behaviors, each optimizer has a `random_state` attribute (`np.random.RandomState`), which can be seeded by the
user if need be. All calls to stochastic functions should there be made through this `random_state`.
By default, it will be seeded randomly by drawing a number from the global numpy random state so
that seeding the global numpy random statewill yield reproducible results as well

A unit tests automatically makes sure that all optimizers have repeatable bvehaviors on a simple test case when seeded from outside (see below).


### About type hints

We have used [type hints](https://docs.python.org/3/library/typing.html) throughout `nevergrad` to make it more robust, and the continuous integration will check that everything is correct when pull requests are submitted. However, **we do not want typing to be an annoyance** for contributors who do not care about it, so please feel entirely free to use `# type: ignore` on each line the continuous integration will flag as incorrect, so that the errors disappear. If we consider it useful to have correct typing, we will update the code after your pull request is merged.


### Optimizer families

If it makes sense to create several variations of your optimizer, using different hyperparameters, you can implement an `OptimizerFamily`. The only aim of this class is to create `Optimizers` and set the parameters before returning it. This is still an experimental API which may evolve soon, and an example can be found in the implementation of [differential evolution algorithms](../nevergrad/optimization/differentialevolution.py).

## How to test it

You are welcome to add tests if you want to make sure your implementation is correct. It is however not required since some tests are run on all registered algorithms. They will test two features:
- that all algorithms are able to find the optimum of a simple 2-variable quadratic fitness function.
- that running the algorithms twice after setting a seed lead to the exact same recommendation. This is useful to make sure we will get repeatibility in the benchmarks.

To run these tests, you can use:
```
pytest nevergrad/optimization/test_optimizerlib.py
```

The repeatability test will however crash the first time you run it, since no value for the recommendation of your algorithm exist. This is automatically added when running the tests, and if everything goes well the second time you run them, it means everything is fine. You will see in you diff that an additional line was added to a file containing all expected recommendations.

If for any reason one of this test is not suitable for your algorithm, we'll discuss this in the pull request and decide of the appropriate workaround.

## How to benchmark it

Benchmarks are implemented in two files [experiments.py](../nevergrad/benchmark/experiments.py) and [frozenexperiments.py](../nevergrad/benchmark/frozenexperiments.py).
While the former can be freely modified (benchmarks will be regularly added and removed), the latter file implements experiments which should not be modified when adding an algorithm, because they are used in tests, or for reproducibility of published results.

Providing some benchmark results along your pull requests will highlight the interest of your algorithm. It is however not required. For now, there is no standard apprroach for benchmarking your algorithm. You can implement your own benchmark, or copy an existing one and add your algorithm. Feel free to propose other solutions.

### How benchmarks are implemented

A benchmark is made of many `Experiment` instances.  An `Experiment` is basically the combination of a test function, and settings for the optimization (optimizer, budget, etc...).

Benchmarks are specified using a generator of `Experiment` instances. See examples in [experiments.py](../nevergrad/benchmark/experiments.py). If you want to make sure your benchmark is perfectly reproducible, you will need to be careful of properly seeding the functions and/or the experiments.
