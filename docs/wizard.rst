.. _wizards:

The case for Wizards in black-box optimization: Nevergrad's wizard NGOpt
========================================================================

After the enormous success of SatZilla and others, SAT competitions became invaded by algorithm selection.
Algorithms, sometimes termed wizards, statically and dynamically choose an algorithm in a vast portfolio of methods:
they can use the dimension, budget, type of variables, any information readily available before starting the run. In some
cases they do active selection, i.e. they use results of small runs for making a decision.

In Nevergrad we do use Wizards: we defined chronologically CMandAS2 (using a lot of active selection), then Shiwa (a lot
of passive selection), then NGOpt (more and more surrogate models, more base algorithms). Currently NGOpt is a pointer
to out current best method.

NGOpt will take into account:
   #. the type of variables
   #. the dimension
   #. the parallelism (num workers)
   #. the budget
   #. any side info you might provide (e.g. whether there is noise).

NGOpt is not perfect. Sometimes, we find a problem in which it did a suboptimal choice. Nonetheless it is frequently
very reasonable, and on average it performs quite well.

By the way, do not trust paper which cite some results about Nevergrad in the BBO competition: they decided to use Nevergrad's random search instead of Nevergrad. They use 15 lines of Nevergrad instead of our enormous optimization wizard. This is ok, but this should not be called "Nevergrad".

Research in black-box optimization and wizards for competition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
`Squirrel <https://arxiv.org/abs/2012.08180>`_, an optimization wizard using a lot of differential evolution (by the way an excellent optimization algorithm,
essentially and unfortunately ignored in the machine learning community) won the BBO Challenge (`BBO Challenge <https://bbochallenge.com/altleaderboard>`_) with its original setup. The setup in the official ranking removed names, which implies that wizards were penalized; Squirrel was still not bad (3rd) but interestingly in the full info case it was ranked 1st. To me, Squirrel has made an excellent point, showing that the prior knowledge is super useful.
Interestingly, the method which ranked best after the change of setup was also using some evolutionary stuff, with `a NSGA component <https://arxiv.org/abs/2012.03826v1>`_. I.e. in both setups, evolution was present in the top result: and in the original setup, wizards were also present. 

Organizing competitions when wizards dominate (and they will dominate in black-box optimization, as much as they are already dominating in combinatorial optimization and scheduling) are more complicated to organize.
Holger Hoos gives an `interesting talk <https://simons.berkeley.edu/talks/tbd-307>`_ about SATzilla and collaborative competitions. In some competitions, portfolios of methods are forbidden (which makes sense, for studying specific  methods); nonetheless, they do perform quite well.

Research in black-box optimization and benchmarking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At Nevergrad we do a lot of benchmarking. We need it for improving our wizard and for checking our codes. We have the following conclusions overall:
   #. Do not trust a comparison on less than 30 objective functions, with several completely different settings (for example, not all with Scikit learn, or not all with Pytorch, etc).
   #. A comparison is just an informed guess if there is not a wide range of settings in terms of dimension and objective functions.
   #. Do not trust positive results from a code which is not properly packaged (e.g. Pypi-packaged). This is just not very reproducible.

I find the results in `MicroPredictions <https://microprediction.github.io/optimizer-elo-ratings/>`_ interesting: I reproduced some of their results so that I am convinced. BOBYQA, and its implementation there, are definitely interesting contributions to the state of the art.

Nevergrad features an enormous `list of benchmarks <https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/benchmark/experiments.py>`_. Importantly, Nevergrad is PyPi-packaged and each benchmark can be run in one line:

.. code-block:: bash

    python -m nevergrad.benchmark illcond --seed=12 --repetitions=50 --num_workers=40 --plot


If it does not work, ping us! The `user group <https://www.facebook.com/groups/nevergradusers>`_ is very active.




