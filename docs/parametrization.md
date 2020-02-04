# Parametrization

**Please note that parametrization is still a work in progress and changes are on their way (including for this documentation)! We are trying to update it to make it simpler and simpler to use (all feedbacks are welcome ;) ), with the side effect that there will be breaking changes.**

The aim of parametrization is to specify what are the parameters that the optimization should be performed upon.
The parametrization subpackage will help you do thanks to:
- the `parameter` modules (accessed by the shortcut `nevergrad.p`) providing classes that should be used to specify each parameter.
- the `FolderFunction` which helps transform any code into a Python function in a few lines. This can be especially helpful to optimize parameters in non-Python 3.6+ code (C++, Octave, etc...) or parameters in scripts.

## Variables

7 types of variables are currently provided:
- `Choice(items)`: describes a parameter which can take values within the provided list of (usually unordered categorical) items, and for which transitions are global (from one item to any other item). The returned element will be sampled as the softmax of the values on these dimensions. Be cautious: this process is non-deterministic and makes the function evaluation noisy.
- `TransitionChoice(items)`: describes a parameter which can take values within the provided list of (usually ordered) items, and for which transitions are local (from one item to close items).
- `Array(shape)`: describes a `np.ndarray` of any shape. The bounds of the array and the mutation of this array can be specified (see `set_bounds`, `set_mutation`). This makes it a very flexible type of variable. Eg. `Array(shape=(2, 3)).set_bounds(0, 2)` encodes for an array of shape `(2, 3)`, with values bounded between 0 and 2.
- `Scalar(dtype)`: describes a float (the default) or an int.
  and all `Array` methods are therefore available. Note that `Gaussian(a, b)` is equivalent to `Scalar().affined(a, b)`.
- `Log(a_min, a_max)`: describes log distributed data between two bounds. Under the hood this uses an `Scalar` with appropriate specifications for bounds and mutations.
- `Instrumentation(*args, **kwargs)`: a container for other parameters. Values of parameters in the `args` will be returned as a `tuple` by `param.args`, and
  values of parameters in the `kwargs` will be returned as a `dict` by `param.kwargs` (in practice, `param.value == (param.args, param.kwargs)`).
  This serves to parametrize functions taking multiple arguments, since you can then call the function with `func(*param.args, **param.kwargs)`.

## Parametrization

Parametrization helps you define the parameters you want to optimize upon.
Currently most algorithms make use of it to help convert the parameters into the "standardized data" space (a vector space spanning all the real values),
where it is easier to define operations.

Let's define the parametrization for a function taking 3 positional arguments and one keyword argument `value`.
- `arg1 = ng.p.TransitionChoice(["a", "b"])` is the first positional argument, it encodes the choice through a single index which can mutate in a continuous way.
- `arg2 = ng.p.Choice(["a", "c", "e"])` is the second one, which can take 3 possible values, without any order, the selection is made stochasticly through the sampling of a softmax. It is encoded by 3 values (the softmax weights) in the "standardized space"
- third argument will be kept constant to ` blublu`
- `value = ng.p.Scalar()` which represents a scalar both in the parameter space, and in the "standardized space"

We then define a parameter holding all these parameters, with a standardized space of dimension 5 (as the sum of the dimensions above):
```python
instru = ng.p.Instrumentation(arg1, arg2, "blublu", value=value)
print(instru.dimension)
>>> 5
```


You can then directly perform optimization on a function given its parametrization:
```python
def myfunction(arg1, arg2, arg3, value=3):
    print(arg1, arg2, arg3)
    return value**2

optimizer = ng.optimizers.OnePlusOne(parametrization=instru, budget=100)
recommendation = optimizer.minimize(myfunction)
print(recommendation.value)
>>> (('b', 'e', 'blublu'), {'value': -0.00014738768964717153})
```



Here is a glipse of what happens on the optimization space:
```python
instru.set_standardized_data([1, -80, -80, 80, 3])
print(instru.args, instru.kwargs)
>>> (('b', 'e', 'blublu'), {'value': 3.0})
```
With this code:
- b is selected because 1 > 0 (the index is 1 for values above 0, and 0 for values under 0 since there are 2 values).
- e is selected because proba(e) = exp(80) / (exp(80) + exp(-80) + exp(-80)) = 1
- `value=3` because the last value of the standardized space (i.e. 3) corresponds to the value of the last kwargs.
```


## External code instantiation

Sometimes it is completely impractical or impossible to have a simple Python3.6+ function to optimize. This may happen when the code you want to optimize is a script. Even more so if the code you want to optimize is not Python3.6+.

We provide tooling for this situation. Go through these steps to instrument your code:
 - **identify the variables** (parameters, constants...) you want to optimize.
 - **add placeholders** to your code. Placeholders are just tokens of the form `NG_ARG{name|comment}` where you can modify the name and comment. The name you set will be the one you will need to use as your function argument. In order to avoid breaking your code, the line containing the placeholders can be commented. To notify that the line should be uncommented for parametrization, you'll need to add "@nevergrad@" at the start of the comment. Here is an example in C which will notify that we want to obtain a function with a `step` argument which will inject values into the `step_size` variable of the code:
```c
int step_size = 0.1
// @nevergrad@ step_size = NG_ARG{step|any comment}
```
- **prepare the command to execute** that will run your code. Make sure that the last printed line is just a float, which is the value to base the optimization upon. We will be doing minimization here, so this value must decrease for better results.
- **instantiate** your code into a function using the `FolderFunction` class:
```python
from nevergrad.parametrization import FolderFunction
folder = "nevergrad/parametrization/examples" # folder containing the code
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
 - the following external file types are registered by default: `[".c", ".h", ".cpp", ".hpp", ".py", ".m"]`. Custom file types can be registered using `FolderFunction.register_file_type` by providing the relevant file suffix as well as the characters that indicate a comment. However, for now, variables which can provide a vector or values (`Gaussian` when providing a `shape`) will inject code with a Python format (list) by default, which may not be suitable.
