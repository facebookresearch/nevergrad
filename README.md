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
