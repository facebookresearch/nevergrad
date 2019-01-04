# Adding an algorithm

These guidelines are for people who want to add an algorithm to `nevergrad`. Feel free to update them if you find them unclear or think they should evolve.

## Where to add the algorithm?

All optimizers are implemented in the `nevergrad.optimization` subpackage, and all optimizer classes are available either in the `nevergrad.optimization.optimizerlib` module, or through the optimizer registry: `nevergrad.optimization.registry`.

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

It records all evaluated points through the `archive` attribute, which is of type:
```
Dict[Tuple[float,...], Value]
```
The key tuple if the point location, and `Value` is a class with attributes:
- `count`: number of evaluations at this point.
- `mean`: mean value of the evaluations at this point.
- `variance`: variance of the evaluations at this point.

For more detauls, see the implementation in [utils.py](../nevergrad/optimization/utils.py).

Through the archive, you can therefore access most useful information about past evaluations.

The base `Optimizer` class also tracks the best optimistic and pessimistic points through the `current_bests` attribute which is of type:
```
Dict[str, Point]
```
The key string is either `optimistic` or `pessimistic`, and the `Point` value is a `Value` with an additional `x` attribute, recording the location of the point.


### Methods and attributes

4 methods are designed to be overriden:
- `__init__`: for the initialization of your algorithm
- `_internal_ask__`: to fetch the next point to be evaluated. This function is the only one that is absolutely required to be overriden. The default `ask` method calls this method (please do not override the default `ask`).
- `_internal_tell`: to update your algorithm with the new point. The default `tell` method calls this internal method after updating the archive (see paragraph above), please do not override it.
- `_internal_provide_recommendation`: to provide the final recommendation. By default, the recommendation is the pessimistic best point.

If the algorithm is not able to handle parallelization (if `ask` cannot be called multiple times consecutively), the `no_parallelization` **class attribute** must be set to `True`.


### Seeding

Seeding has an important part for the significance and reproducibility of the algorithm benchmarking. For this to work, it is however important to **avoid seeding from inside** the algorithm. Indeed:
- we expect stochastic algorithms to be actually stochastic, if we set a seed inside the implementation this assumption is broken.
- we need the randomness to obtain relevant statistics when benchmarking the algorithms on deterministic functions.
- we can seed anyway from **outside** when we need it. This is what happens in the benchmarks: in this case we do want each independent run to be repeatable.

For consistency and simplicity's sake, please prefer `numpy`'s random generator over the standard one.
Also, you may instanciate a random generator for each optimizer and using it afterwards. This way it makes the optimizer independent of other uses of the default random generator.
Still, please seed it with the standard numpy random generator so that seeding the standard generator will produce deterministic outputs:
```python
self._rng = np.ranndom.RandomState(np.random.randint(2**32))
```

A unit tests automatically makes sure that all optimizers have repeatable bvehaviors on a simple test case when seeded from outside (see below).


### About type hints

We have used [type hints](https://docs.python.org/3/library/typing.html) throughout `nevergrad` to make it more robust, and the continuous integration will check that everything is correct when pull requests are submitted. However, **we do not want typing to be an annoyance** for contributors who do not care about it, so please feel entirely free to use `# type: ignore` on each line the continuous integration will flag as incorrect, so that the errors disappear. If we consider it useful to have correct typing, we will update the code after your pull request is merged.

## How to test it

You are welcome to add tests if you want to make sure your implementation is correct. It is however not required since some tests are run on all registered algorithms. They will test two features:
- that all algorithms are able to find the optimum of a simple 2-variable quadratic fitness function.
- that running the algorithms twice after setting a seed lead to the exact same recommendation. This is useful to make sure we will get repeatibility in the benchmarks.

To run these tests, you can use:
```
nosetests nevergrad/optimization/test_optimizerlib.py
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
