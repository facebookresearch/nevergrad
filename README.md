[![CircleCI](https://circleci.com/gh/facebookresearch/nevergrad/tree/master.svg?style=svg)](https://circleci.com/gh/facebookresearch/nevergrad/tree/master)

# Nevergrad - A gradient-free optimization platform

![Nevergrad](docs/resources/Nevergrad-LogoMark.png)


`nevergrad` is a Python 3.6+ library. It can be installed with:

```
pip install nevergrad
```

More installation options and complete instructions are available in the "Getting started" section of the [**documentation**](https://facebookresearch.github.io/nevergrad/).

You can join Nevergrad users Facebook group [here](https://www.facebook.com/groups/nevergradusers/).

Minimizing a function using an optimizer (here `OnePlusOne`) is straightforward:

```python
import nevergrad as ng

def square(x):
    return sum((x - .5)**2)

optimizer = ng.optimizers.OnePlusOne(parametrization=2, budget=100)
recommendation = optimizer.minimize(square)
print(recommendation)  # optimal args and kwargs
>>> Array{(2,)}[recombination=average,sigma=1.0]:[0.49971112 0.5002944 ]
```

A slightly more complicated example, with one variable in R2, one variable in Z, and one categorical variable with possible values a and b and c:
```
import nevergrad as ng
from nevergrad.parametrization import parameter as par

# Our objective function depends on:
# - x (in R^2)
# - y (in N)
# - z (in {"a", "b", "c"})
#
# The optimum is x=(0.5, 0.5), y=3, z=a.
def square(x, y, z):
    return sum((x - .5)**2) + (y - 3)**2 + (0 if z == "a" else 1)

parametrization = par.Instrumentation(
    x=par.Array(shape=(2,)),
    y=par.Scalar(1).set_integer_casting(),  # Can be negative.
    z=par.Choice(("a", "b", "c"))
        )

optimizer = ng.optimizers.OnePlusOne(parametrization=parametrization, budget=1000)
recommendation = optimizer.minimize(square)

print(recommendation)
```


![Example of optimization](docs/resources/TwoPointsDE.gif)

*Convergence of a population of points to the minima with two-points DE.*


## Documentation

Check out our [**documentation**](https://facebookresearch.github.io/nevergrad/)! It's still a work in progress, don't hesitate to submit issues and/or PR to update it and make it clearer!


## Citing

```bibtex
@misc{nevergrad,
    author = {J. Rapin and O. Teytaud},
    title = {{Nevergrad - A gradient-free optimization platform}},
    year = {2018},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://GitHub.com/FacebookResearch/Nevergrad}},
}
```

## License

`nevergrad` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.
