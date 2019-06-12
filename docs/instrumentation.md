# Instrumentation

**Please note that instrumentation is still a work in progress. We will try to update it to make it simpler and simpler to use (all feedbacks are welcome ;) ), with the side effect that their will be breaking changes.**

The aim of instrumentation is to turn a piece of code with parameters you want to optimize into a function defined on an n-dimensional continuous data space in which the optimization can easily be performed. For this, discrete/categorial arguments must be transformed to continuous variables, and all variables concatenated. The instrumentation subpackage will help you do thanks to:
- the `variables` modules providing priors that can be used to define each argument.
- the `Instrumentation`, and `InstrumentedFunction` classes which provide an interface for converting any arguments into the data space used for optimization, and convert from data space back to the arguments space.
- the `FolderFunction` which helps transform any code into a Python function in a few lines. This can be especially helpful to optimize parameters in non-Python 3.6+ code (C++, Octave, etc...) or parameters in scripts.


## Variables

5 types of variables are currently provided:
- `SoftmaxCategorical(items)`: converts a list of `n` (unordered) categorial items into an `n`-dimensional space. The returned element will be sampled as the softmax of the values on these dimensions. Be cautious: this process is non-deterministic and makes the function evaluation noisy.
- `OrderedDiscrete(items)`: converts a list of (ordered) discrete items into a 1-dimensional variable. The returned value will depend on the value on this dimension: low values corresponding to first elements of the list, and high values to the last.
- `Gaussian(mean, std)`: normalizes a `n`-dimensional variable with independent Gaussian priors (1-dimension per value).
- `Array(dim1, ...)`: casts the data from the optimizaton space into a `np.ndarray` of any shape, to which some transforms can be applied
  (see `asscalar`, `affined`, `exponentiated`, `bounded`). This makes it a very flexible type of variable. Eg. `Array(2, 3).bounded(0, 2)` encodes for an array of shape `(2, 3)`, with values bounded between 0 and 2.
- `Scalar(dtype)`: casts the data from the optimization space into a float (the default) or an int. It is equivalent to `Array(1).asscalar(dtype)`
  and all `Array` methods are therefore available. For instance, one can use it for a logarithmicly distributed value between
  0.001 and 1.: `Scalar().bounded(0, 3).exponentiated(base=10, coeff=-1)`.  Also, note that `Gaussian(a, b)` is equivalent to `Scalar().affined(a, b)`.


## Instrumentation

Instrumentation helps you convert a set of arguments into variables in the data space which can be optimized. The core class performing this conversion is called `Instrumentation`. It provides arguments conversion through the `arguments_to_data` and `data_to_arguments` methods. Since `data_to_arguments` can be stochastic, the instrumentation holds a random state (`instrumentation.random_state`) which is also used by optimizers.


```python
import nevergrad as ng

# argument transformation
arg1 = ng.var.OrderedDiscrete(["a", "b"])  # 1st arg. = positional discrete argument
arg2 = ng.var.SoftmaxCategorical(["a", "c", "e"])  # 2nd arg. = positional discrete argument
value = ng.var.Gaussian(mean=1, std=2)  # the 4th arg. is a keyword argument with Gaussian prior

# create the instrumented function
instrum = ng.Instrumentation(arg1, arg2, "blublu", value=value)
# the 3rd arg. is a positional arg. which will be kept constant to "blublu"
print(instrum.dimension)  # 5 dimensional space

# The dimension is 5 because:
# - the 1st discrete variable has 1 possible values, represented by a hard thresholding in
#   a 1-dimensional space, i.e. we add 1 coordinate to the continuous problem
# - the 2nd discrete variable has 3 possible values, represented by softmax, i.e. we add 3 coordinates to the continuous problem
# - the 3rd variable has no uncertainty, so it does not introduce any coordinate in the continuous problem
# - the 4th variable is a real number, represented by single coordinate.


print(instrum.data_to_arguments([1, -80, -80, 80, 3]))
# prints (args, kwargs): (('b', 'e', 'blublu'), {'value': 7})
# b is selected because 1 > 0 (the threshold is 0 here since there are 2 values.
# e is selected because proba(e) = exp(80) / (exp(80) + exp(-80) + exp(-80))
# value=7 because 3 * std + mean = 7
```


For convenience and until a better way is implemented (see future notice), we provide an `InstrumentedFunction` class converting a function of any parameter space into the data space. Here is a basic example of its use:

**Future notice**: `InstrumentedFunction` may come to disappear (or at least we discourage its use) since a new API for instrumenting on the optimizer side has been added in v0.2.0.

You can then directly perform optimization on a function given its instrumentation:
```python
def myfunction(arg1, arg2, arg3, value=3):
    print(arg1, arg2, arg3)
    return value**2

optimizer = ng.optimizers.OnePlusOne(instrumentation=instrum, budget=100)
recommendation = optimizer.minimize(ifunc)
```

When you have performed optimization on this function and want to trace back to what should your values be, use:
```python
# let us instantiate a fake recommendation for the sake of this tutorial
recommendation = optimizer.create_candidate.from_data([1, -80, -80, 80, -.5], deterministic=True)
print(recommendation.args)    # should print ["b", "e", "blublu"]
print(recommendation.kwargs)  # should print {"value": 0} because -.5 * std + mean = 0

# but be careful, since some variables are stochastic (SoftmaxCategorical ones are), setting deterministic=False may yield different results
# The following will print more information on the conversion to your arguments:
print(instrum.get_summary(recommendation.data))
```



## External code instantiation

Sometimes it is completely impractical or impossible to have a simple Python3.6+ function to optimize. This may happen when the code you want to optimize is a script. Even more so if the code you want to optimize is not Python3.6+.

We provide tooling for this situation. Go through this steps to instrument your code:
 - **identify the variables** (parameters, constants...) you want to optimize.
 - **add placeholders** to your code. Placeholders are just tokens of the form `NG_ARG{name|comment}` where you can modify the name and comment. The name you set will be the one you will need to use as your function argument. In order to avoid breaking your code, the line containing the placeholders can be commented. To notify that the line should be uncommented for instrumentation, you'll need to add "@nevergrad@" at the start of the comment. Here is an example in C which will notify that we want to obtain a function with a `step` argument which will inject values into the `step_size` variable of the code:
```c
int step_size = 0.1
// @nevergrad@ step_size = NG_ARG{step|any comment}
```
- **prepare the command to execute** that will run your code. Make sure that the last printed line is just a float, which is the value to base the optimization upon. We will be doing minimization here, so this value must decrease for better results.
- **instantiate** your code into a function using the `FolderFunction` class:
```python
from nevergrad.instrumentation import FolderFunction
folder = "nevergrad/instrumentation/examples" # folder containing the code
command = ["python", "examples/script.py"]  # command to run from right outside the provided folder
func = FolderFunction(folder, command, clean_copy=True)
print(func.placeholders)  # will print the number of variables of the function
# prints: [Placeholder('value1', 'this is a comment'), Placeholder('value2', None), Placeholder('string', None)]
print(func(value1=2, value2=3, string="blublu"))
# prints: 12.0
```
- **instrument** the function, (see Instrumentation section just above).


## Tips and caveats

 - using `FolderFunction` argument `clean_copy=True` will copy your folder so that tempering with it during optimization will run different versions of your code.
 - under the hood, with or without `clean_copy=True`, when calling the function, `FolderFunction` will create symlink copy of the initial folder, remove the files that have tokens, and create new ones with appropriate values. Symlinks are used in order to avoid duplicating large projects, but they have some drawbacks, see next point ;)
 - one can add a compilation step to `FolderFunction` (the compilation just has to be included in the script). However, be extra careful that if the initial folder contains some build files, they could be modified by the compilation step, because of the symlinks. Make sure that during compilation, you remove the build symlinks first! **This feature has not been fool proofed yet!!!**
 - the following external file types are registered by default: `[".c", ".h", ".cpp", ".hpp", ".py", ".m"]`. Custom file types can be registered using `instrumentation.register_file_type` by providing the relevant file suffix as well as the characters that indicate a comment. However, for now, variables which can provide a vector or values (`Gaussian` when providing a `shape`) will inject code with a Python format (list) by default, which may not be suitable.
