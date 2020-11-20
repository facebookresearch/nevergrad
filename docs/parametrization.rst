.. _parametrizing:

Parametrizing your optimization
===============================

**Please note that parametrization is still a work in progress and changes are on their way (including for this documentation)! We are trying to update it to make it simpler and simpler to use (all feedbacks are welcome ;) ), with the side effect that there will be breaking changes.**

The aim of parametrization is to specify what are the parameters that the optimization should be performed upon.
The parametrization subpackage will help you do thanks to:

- the :code:`parameter` modules (accessed by the shortcut :code:`nevergrad.p`) providing classes that should be used to specify each parameter.
- the :code:`FolderFunction` which helps transform any code into a Python function in a few lines. This can be especially helpful to optimize parameters in non-Python 3.6+ code (C++, Octave, etc...) or parameters in scripts.


Preliminary examples
--------------------

The code below defines a parametrization that creates a :code:`dict` with 3 elements:

- a log-distributed scalar.
- an array of size 2.
- a random letter, which is either :code:`a`, :code:`b` or :code:`c`,

The main attribute for users is the :code:`value` attribute, which provides the value of the defined :code:`dict`:

.. literalinclude:: ../nevergrad/parametrization/test_param_doc.py
    :language: python
    :dedent: 4
    :start-after: DOC_PARAM_0
    :end-before: DOC_PARAM_1


Parametrization is central in :code:`nevergrad` since it is the interface between the optimization domain
defined by the users, and the standardized representations used by the optimizers. The snippet below
shows how to duplicate the parameter (i.e. spawn a child from the parameter), manually updating the value,
and export to the standardized space. Unless implementing an algorithm, users should not have any need
for this export. However, setting the :code:`value` manually can be used to provide an initial prior to
the optimizer.

.. literalinclude:: ../nevergrad/parametrization/test_param_doc.py
    :language: python
    :dedent: 4
    :start-after: DOC_PARAM_10
    :end-before: DOC_PARAM_11

Similarly, optimizers can use :code:`mutate` and :code:`recombine` methods to update the value of parameters.
You can easily check how parameters mutate, and mutation of :code:`Array` variables can be adapted to your need:

.. literalinclude:: ../nevergrad/parametrization/test_param_doc.py
    :language: python
    :dedent: 4
    :start-after: DOC_PARAM_100
    :end-before: DOC_PARAM_101

Note that optimizer :code:`freeze` candidates to avoid modification their modification and border effects, however you can always spawn new parameters from them.

Parametrization is also responsible for the randomness in nevergrad. They have a :code:`random_state` which can be set for reproducibility

.. literalinclude:: ../nevergrad/parametrization/test_param_doc.py
    :language: python
    :dedent: 4
    :start-after: DOC_PARAM_1000
    :end-before: DOC_PARAM_1001

Parameters
----------

7 types of parameters are currently provided:

- :code:`Choice(items)`: describes a parameter which can take values within the provided list of (usually unordered categorical) items, and for which transitions are global (from one item to any other item). The returned element will be sampled as the softmax of the values on these dimensions. Be cautious: this process is non-deterministic and makes the function evaluation noisy.
- :code:`TransitionChoice(items)`: describes a parameter which can take values within the provided list of (usually ordered) items, and for which transitions are local (from one item to close items).
- :code:`Array(shape=shape)`: describes a :code:`np.ndarray` of any shape. The bounds of the array and the mutation of this array can be specified (see :code:`set_bounds`, :code:`set_mutation`). This makes it a very flexible type of parameter. Eg. :code:`Array(shape=(2, 3)).set_bounds(0, 2)` encodes for an array of shape :code:`(2, 3)`, with values bounded between 0 and 2. It can be also set to an array of integers (see :code:`set_integer_casting`)
- :code:`Scalar()`: describes a scalar. This parameter inherits from all :code:`Array` methods, so it can be bounded, projected to integers and mutation rate can be customized.
- :code:`Log(lower, upper)`: describes log distributed data between two bounds. Under the hood this uses an :code:`Scalar` with appropriate specifications for bounds and mutations.
- :code:`Instrumentation(*args, **kwargs)`: a container for other parameters. Values of parameters in the :code:`args` will be returned as a :code:`tuple` by :code:`param.args`, and
  values of parameters in the :code:`kwargs` will be returned as a :code:`dict` by :code:`param.kwargs` (in practice, :code:`param.value == (param.args, param.kwargs)`).
  This serves to parametrize functions taking multiple arguments, since you can then call the function with :code:`func(*param.args, **param.kwargs)`.


Follow the link to the API reference for more details and initialization options:

.. autosummary::
    nevergrad.p.Array
    nevergrad.p.Scalar
    nevergrad.p.Log
    nevergrad.p.Dict
    nevergrad.p.Tuple
    nevergrad.p.Instrumentation
    nevergrad.p.Choice
    nevergrad.p.TransitionChoice


Parametrization
---------------

Parametrization helps you define the parameters you want to optimize upon.
Currently most algorithms make use of it to help convert the parameters into the "standardized data" space (a vector space spanning all the real values),
where it is easier to define operations.

Let's define the parametrization for a function taking 3 positional arguments and one keyword argument :code:`value`.

- :code:`arg1 = ng.p.Choice([["Helium", "Nitrogen", "Oxygen"]])` is the first positional argument, which can take 3 possible values, without any order, the selection is made stochasticly through the sampling of a softmax. It is encoded by 3 values (the softmax weights) in the "standardized space".
- :code:`arg2 = ng.p.TransitionChoice(["Solid", "Liquid", "Gas"])` is the second one, it encodes the choice (i.e. 1 dimension) through a single index which can mutate in a continuous way.
- third argument will be kept constant to :code:`blublu`
- :code:`values = ng.p.Tuple(ng.p.Scalar().set_integer_casting(), ng.p.Scalar())` which represents a tuple of two scalars with different numeric types in the parameter space, and in the "standardized space"

We then define a parameter holding all these parameters, with a standardized space of dimension 6 (as the sum of the dimensions above):

.. literalinclude:: ../nevergrad/optimization/test_doc.py
    :language: python
    :dedent: 4
    :start-after: DOC_PARAM_0
    :end-before: DOC_PARAM_1


You can then directly perform optimization on a function given its parametrization:


.. literalinclude:: ../nevergrad/optimization/test_doc.py
    :language: python
    :dedent: 4
    :start-after: DOC_PARAM_1
    :end-before: DOC_PARAM_2


Here is a glimpse of what happens on the optimization space:

.. literalinclude:: ../nevergrad/optimization/test_doc.py
    :language: python
    :dedent: 4
    :start-after: DOC_PARAM_2
    :end-before: DOC_PARAM_3

With this code:

- :code:`Nitrogen` is selected because proba(e) = exp(80) / (exp(80) + exp(-80) + exp(-80)) = 1
- :code:`Liquid` is selected because the index for `Liquid` is around 0 in the standardized space.
- :code:`amount=(3, 5.0)` because the last two values of the standardized space (i.e. 3.0, 5.0) corresponds to the value of the last kwargs.


Parametrizing external code
---------------------------


Sometimes it is completely impractical or impossible to have a simple Python3.6+ function to optimize. This may happen when the code you want to optimize is a script. Even more so if the code you want to optimize is not Python3.6+.

We provide tooling for this situation but this is hacky, so if you can avoid it, **do avoid it**. Otherwise, go through these steps to instrument your code:

 - **identify the variables** (parameters, constants...) you want to optimize.
 - **add placeholders** to your code. Placeholders are just tokens of the form :code:`NG_ARG{name|comment}` where you can modify the name and comment. The name you set will be the one you will need to use as your function argument. In order to avoid breaking your code, the line containing the placeholders can be commented. To notify that the line should be uncommented for parametrization, you'll need to add "@nevergrad@" at the start of the comment. Here is an example in C which will notify that we want to obtain a function with a :code:`step` argument which will inject values into the :code:`step_size` variable of the code:

.. code-block:: C++

    int step_size = 0.1
    // @nevergrad@ step_size = NG_ARG{step|any comment}

- **prepare the command to execute** that will run your code. Make sure that the last printed line is just a float, which is the value to base the optimization upon. We will be doing minimization here, so this value must decrease for better results.
- **instantiate** your code into a function using the :code:`FolderFunction` class:

.. literalinclude:: ../nevergrad/parametrization/test_instantiate.py
    :language: python
    :dedent: 4
    :start-after: DOC_INSTANTIATE_0
    :end-before: DOC_INSTANTIATE_1

- **parametrize** the function (see Parametrization section just above).


Tips and caveats
^^^^^^^^^^^^^^^^

 - using :code:`FolderFunction` argument :code:`clean_copy=True` will copy your folder so that tempering with it during optimization will run different versions of your code.
 - under the hood, with or without :code:`clean_copy=True`, when calling the function, :code:`FolderFunction` will create symlink copy of the initial folder, remove the files that have tokens, and create new ones with appropriate values. Symlinks are used in order to avoid duplicating large projects, but they have some drawbacks, see next point ;)
 - one can add a compilation step to :code:`FolderFunction` (the compilation just has to be included in the script). However, be extra careful that if the initial folder contains some build files, they could be modified by the compilation step, because of the symlinks. Make sure that during compilation, you remove the build symlinks first! **This feature has not been fool proofed yet!!!**
 - the following external file types are registered by default: :code:`[".c", ".h", ".cpp", ".hpp", ".py", ".m"]`. Custom file types can be registered using :code:`FolderFunction.register_file_type` by providing the relevant file suffix as well as the characters that indicate a comment. However, for now, parameters which provide a vector or values (:code:`Array`) will inject code with a Python format (list) by default, which may not be suitable.
