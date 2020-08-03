Examples - Working with Pyomo model
=========================================

Pyomo is an open source software package for modeling and solving mathematical programs in Python [Hart2011]_.
This section gives an example of how to optimize Pyomo models using Nevergrad.
More examples can be found in nevergrad/examples/pyomogallery directory.

Concrete Model
------------------------------------------------------------------------------------
Let us create a :code:`ConcreteModel` instance using Pyomo.
In a :code:`ConcreteModel`, each component is fully initialized.

.. code-block:: python

    import pyomo.environ as pyomo

    def square(m):
        return pyomo.quicksum((m.x[i] - 0.5)**2 for i in m.x)

    model = pyomo.ConcreteModel()
    model.x = pyomo.Var([0, 1], domain=pyomo.Reals)
    model.obj = pyomo.Objective(rule=square)
    model.Constraint1 = pyomo.Constraint(rule=lambda m: m.x[0] >= 1)
    model.Constraint2 = pyomo.Constraint(rule=lambda m: m.x[1] >= 0.8)

In the above model, it is clear that the parameter in the objective function is :code:`x`, which is an indexed variable subjected to two constraints.
Nevergrad has provided an utility to parse the Pyomo model to enable you to create :code:`ExperimentFunction` automatically.
Such :code:`ExperimentFunction` contains the parameters, constraints, and an objective function to be optimized.
Note that only single objective model is supported by the utility.
To do this, you should first import new module.

.. code-block:: python

    import nevergrad as ng
    from nevergrad.functions.pyomo.core import Pyomo


In our example as shown below, :code:`OnePlusOne` optimizer is used to minimize the objective function:

.. code-block:: python

    func = core.Pyomo('The Square of Distance from Point (0.5, 0.5)', model)
    optimizer = ng.optimizers.OnePlusOne(parametrization=func.parametrization, budget=100)
    recommendation = optimizer.minimize(func.function)

Finally, the result can be read using

.. code-block:: python

    print(recommendation.kwargs['x[0]'])
    print(recommendation.kwargs['x[1]'])



Abstract Model
------------------------------------------------------------------------------------
Pyomo model has to be fully constructed before you solve the model.
When you define an :code:`AbstractModel`, construction occurs in two phases.
First of all, you declare and attach components to the model, those components are empty containers and not fully constructed.
Next, you will fill in the containers using the :code:`create_instance()` method.
The :code:`create_instance()` method allows you to take the abstract model and optional data and returns a new :code:`ConcreteModel` instance.
You are recommended to use :code:`DataPortal` in Pyomo to load data in various format.
You may refer to the `Pyomo documentation <https://pyomo.readthedocs.io/en/stable/working_abstractmodels/data/dataportals.html>`_ for the details.


.. code-block:: python

    data = pyomo.DataPortal()
    data.load(filename=data_path, model=model)
    model = abstract_model.create_instance(data)

    # model, which is a ConcreteModel instance, is fully constructed and can be optimized by Nevergrad.



.. [Hart2011] Hart, William E., Jean-Paul Watson, and David L. Woodruff. “Pyomo: modeling and solving mathematical programs in Python.” Mathematical Programming Computation 3, no. 3 (2011): 219-260.