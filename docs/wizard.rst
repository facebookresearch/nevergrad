.. _wizards:

The case for Wizards in black-box optimization
==============================================

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

By the way, do not trust paper who cite some results about Nevergrad in the BBO competition: they decided to use Nevergrad's random search instead of Nevergrad. They use 15 lines of Nevergrad instead of our enormous optimization wizard. This is ok, but this should not be called "Nevergrad".

Research in black-box optimization and wizards for competition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
`Squirrel <https://arxiv.org/abs/2012.08180>`_, an optimization wizard using a lot of differential evolution (by the way an excellent optimization algorithm,
essentially and unfortunately ignored in the machine learning community) won the BBO Challenge (`BBO Challenge <https://bbochallenge.com/altleaderboard>`_) with its original setup. The setup was modified (removal of some prior knowledge typically used by wizards). Squirrel was still not bad, but ranked 3rd instead of first. It was still the best performing method with the original setup. To me, Squirrel has made an excellent point, showing that the prior knowledge is super useful.
Interestingly, the method which ranked best after the change of setup was also using some evolutionary stuff, with a NSGA component.

Organizing competitions when wizards dominate (and they will dominate in black-box optimization, as much as they are already dominating in combinatorial optimization and scheduling) are more complicated to organize.
Holger Hoos gives an `interesting talk <https://simons.berkeley.edu/talks/tbd-307>`_ about SATzilla and collaborative competitions. That sounds good to me, so that
people do not end up overfitting competitions or preferring method A to method B for political reasons.

Research in black-box optimization and benchmarking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At Nevergrad we do a lot of benchmarking. We need it for improving our wizard and for checking our codes. We have the following conclusions overall:
   #. Do not trust a comparison on less than 30 objective functions, with several completely different settings (for example, not all with Scikit learn, or not all with Pytorch, etc).
   #. Do not trust a comparison if the dimensions do not range from 2 to 5000. 
   #. Do not trust a comparison if it is run by people who are the authors of a method involved in the comparison.
   #. Do not trust positive results from a code which is not properly packaged (e.g. Pypi-packaged).

Nevergrad features an enormous `list of benchmarks <https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/benchmark/experiments.py>`_. Importantly, Nevergrad is PyPi-packaged and each benchmark can be run in one line.

.. code-block:: bash

    python -m nevergrad.benchmark illcond --seed=12 --repetitions=50 --num_workers=40 --plot


If it does not work, ping us! The `user group <https://www.facebook.com/groups/nevergradusers>`_ is very active.




