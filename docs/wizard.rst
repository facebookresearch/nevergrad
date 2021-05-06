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
# the type of variables
# the dimension
# the parallelism (num workers)
# the budget
# any side info you might provide (e.g. whether there is noise).

NGOpt is not perfect. Sometimes, we find a problem in which it did a suboptimal choice. Nonetheless it is frequently
very reasonable, and on average it performs quite well.


Research in black-box optimization and wizards for competition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
`Squirrel<https://arxiv.org/abs/2012.08180>`_, an optimization wizard using a lot of differential evolution (by the way an excellent optimization algorithm,
essentially and unfortunately ignored in the machine learning community) initially won the BBO Challenge (`BBO Challenge<https://bbochallenge.com/altleaderboard>`_). 

However the organizers decided to rerun the algorithms a second time after removal of naming information so that wizards which were using such information
would perform worse. Squirrel was still not bad, but ranked 3rd instead of first.

It is a classical issue in benchmarking. People typically have in mind a method they want to investigate: for example the title
of the BBO paper is `` Bayesian Optimization is Superior to Random Search for Machine Learning Hyperparameter Tuning: Analysis of the Black-Box Optimization Challenge 2020''
and the organizers refused to use the optimization wizard of Nevergrad (what they call Nevergrad is actually a random search extracted from Nevergrad). At least the authors should mention that this is not Nevergrad but the random search extracted from Nevergrad.
My opinion is that Squirrel did win the competition but as it was a competition dedicated to Bayesian Optimization things were altered until Bayesian Optimization becomes more visible.

Research in black-box optimization and benchmarking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
At Nevergrad we do a lot of benchmarking. We have the following conclusions overall:
# Do not trust a comparison on less than 30 objective functions, with several completely different settings (for example, not all with Scikit learn, or not all with Pytorch, etc).
# Do not trust a comparison if the dimensions do not range from 2 to 5000. 
# Do not trust a comparison if it is run by people who are the authors of a method involved in the comparison.
# Do not trust positive results from a code which is not properly packaged (e.g. Pypi-packaged).

Nevergrad features an enormous list of benchmarks. Importantly, Nevergrad is PyPi-packaged and each benchmark can be run in one line.

.. code-block:: bash

    python -m nevergrad.benchmark illcond --seed=12 --repetitions=50 --num_workers=40 --plot


If it does not work, ping us! The `user group<https://www.facebook.com/groups/nevergradusers>`_ is very active.




