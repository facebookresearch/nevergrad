Nevergrad - A gradient-free optimization platform
=================================================

.. image:: ./resources/Nevergrad-LogoMark.png

.. image:: https://img.shields.io/badge/Support-Ukraine-FFD500?style=flat&labelColor=005BBB
   :alt: Support Ukraine - Help Provide Humanitarian Aid to Ukraine.
   :target: https://opensource.fb.com/support-ukraine

This documentation is a work in progress, feel free to help us update/improve/restucture it!

Quick start
-----------

:code:`nevergrad` is a Python 3.6+ library. It can be installed with:

.. code-block:: bash

    pip install nevergrad

You can find other installation options (including for Windows users) in the :ref:`Getting started section <getting_started>`.

Feel free to join `Nevergrad users Facebook group <https://www.facebook.com/groups/nevergradusers/>`_.

Minimizing a function using an optimizer (here :code:`NgIohTuned`, our adaptative optimization algorithm) can be easily run with:

.. literalinclude:: ../nevergrad/optimization/test_doc.py
    :language: python
    :dedent: 4
    :start-after: DOC_SIMPLEST_0
    :end-before: DOC_SIMPLEST_1


.. image:: ./resources/TwoPointsDE.gif
  :width: 400

*Convergence of a population of points to the minima with two-points DE.*

:code:`nevergrad` can also support bounded continuous variables as well as discrete variables, and mixture of those.
To do this, one can specify the input space:

.. literalinclude:: ../nevergrad/parametrization/test_param_doc.py
    :language: python
    :dedent: 4
    :start-after: DOC_README_0
    :end-before: DOC_README_1

Learn more about parametrization in the :ref:`Parametrization section <parametrizing>`!


.. toctree::
   :maxdepth: 3
   :caption: CONTENTS

   getting_started.rst
   windows.md
   optimization.rst
   parametrization.rst
   benchmarking.rst
   contributing.rst
   opencompetition2020.md


.. toctree::
   :maxdepth: 3
   :caption: API REFERENCE

   ref_optimizer.rst
   ref_parametrization.rst
   ref_callbacks.rst


.. toctree::
   :maxdepth: 3
   :caption: EXAMPLES

   machinelearning.rst
   r.rst
   benchmarks.rst
   pyomo.rst
   examples/guiding/Guiding image generation with Nevergrad.md
   examples/diversity/Diversity in image generation with Nevergrad.md
   examples/lognormal/Lognormal mutations in Nevergrad.md
   examples/retrofitting/Retrofitting with Nevergrad.md


.. toctree::
   :maxdepth: 3
   :caption: STATISTICS

   statistics/Statistics.md
   statistics/AgStatistics.md


Citing
------

.. code-block:: bibtex

    @misc{nevergrad,
        author = {J. Rapin and O. Teytaud},
        title = {{Nevergrad - A gradient-free optimization platform}},
        year = {2018},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://GitHub.com/FacebookResearch/Nevergrad}},
    }


License
-------

:code:`nevergrad` is released under the MIT license. See `LICENSE <https://github.com/facebookresearch/nevergrad/blob/main/LICENSE>`_ for additional details about it, as well as our `Terms of Use <https://opensource.facebook.com/legal/terms>`_ and `Privacy Policy <https://opensource.facebook.com/legal/privacy>`_.
Copyright © Meta Platforms, Inc.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
