Examples - Working with Pyomo model
=========================================

Pyomo is an open source software package for modeling and solving mathematical programs in Python [Hart2011]_.
This section gives an example of how to optimize Pyomo models using Nevergrad.

Concrete Model
------------------------------------------------------------------------------------
Let us create a :code:`ConcreteModel` instance using Pyomo.
In a :code:`ConcreteModel`, each component is fully initialized.

.. literalinclude:: ../nevergrad/functions/pyomo/test_pyomo_doc.py
    :language: python
    :dedent: 4
    :start-after: DOC_CONCRETE_0
    :end-before: DOC_CONCRETE_1

In the above model, it is clear that the parameter in the objective function is :code:`x`, which is an indexed variable subjected to two constraints.
Nevergrad has provided an utility to parse the Pyomo model to enable you to create :code:`ExperimentFunction` automatically.
Such :code:`ExperimentFunction` contains the parameters, constraints, and an objective function to be optimized.
Note that only single objective model is supported by the utility.
To do this, you should first import new module.

.. literalinclude:: ../nevergrad/functions/pyomo/test_pyomo_doc.py
    :language: python
    :dedent: 4
    :start-after: DOC_CONCRETE_10
    :end-before: DOC_CONCRETE_11

In our example as shown below, :code:`OnePlusOne` optimizer is used to minimize the objective function:

.. literalinclude:: ../nevergrad/functions/pyomo/test_pyomo_doc.py
    :language: python
    :dedent: 4
    :start-after: DOC_CONCRETE_100
    :end-before: DOC_CONCRETE_101

Finally, the result can be read using

.. literalinclude:: ../nevergrad/functions/pyomo/test_pyomo_doc.py
    :language: python
    :dedent: 4
    :start-after: DOC_CONCRETE_1000
    :end-before: DOC_CONCRETE_1001



Abstract Model
------------------------------------------------------------------------------------
Pyomo model has to be fully constructed before you solve the model.
When you define an :code:`AbstractModel`, construction occurs in two phases.
First of all, you declare and attach components to the model, those components are empty containers and not fully constructed.
Next, you will fill in the containers using the :code:`create_instance()` method.
The :code:`create_instance()` method allows you to take the abstract model and optional data and returns a new :code:`ConcreteModel` instance.
You are recommended to use :code:`DataPortal` in Pyomo to load data in various format.
You may refer to the `Pyomo documentation <https://pyomo.readthedocs.io/en/stable>`_ for the details.


.. literalinclude:: ../nevergrad/functions/pyomo/test_pyomo_doc.py
    :language: python
    :dedent: 4
    :start-after: DOC_ABSTRACT_100
    :end-before: DOC_ABSTRACT_101


.. [Hart2011] Hart, William E., Jean-Paul Watson, and David L. Woodruff. “Pyomo: modeling and solving mathematical programs in Python.” Mathematical Programming Computation 3, no. 3 (2011): 219-260.
