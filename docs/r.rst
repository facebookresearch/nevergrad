.. _r:

Examples - Nevergrad for R
==========================


.. code-block:: R

     # Library and environment for Reticulate/Nevergrad.
     library("reticulate")
     conda_create("r-reticulate")
     conda_install("r-reticulate", "nevergrad", pip=TRUE)
     use_condaenv("r-reticulate")

     # Only if you use parallelism.
     library(doParallel)

     # Choose your optimization method below.
     optimizer_name <- "NGOpt"
     # optimizer_name <- "DoubleFastGADiscreteOnePlusOne"
     # optimizer_name <- "OnePlusOne"
     # optimizer_name <- "DE"
     # optimizer_name <- "RandomSearch"
     # optimizer_name <- "TwoPointsDE"
     # optimizer_name <- "Powell"
     # optimizer_name <- "MetaModel"  CRASH !!!!
     # optimizer_name <- "SQP"
     # optimizer_name <- "Cobyla"
     # optimizer_name <- "NaiveTBPSA"
     # optimizer_name <- "DiscreteOnePlusOne"
     # optimizer_name <- "cGA"
     # optimizer_name <- "ScrHammersleySearch"

     # Now we can play with Nevergrad as usual.
     # We assume here that we have 17 continuous hyperparameters with values in [0, 1].
     # We can do other instrumentations, as discussed below.
     my_tuple <- tuple(17)
     instrumentation <- ng$p$Array(shape=my_tuple)
     instrumentation$set_bounds(0., 1.)
     num_workers <- 3  # We want to be able to evaluate 3 hyperparametrizations simultaneously.
     num_iterations <- 100 * num_workers  # Let us say we have a budget of 100xnum_workers hyperparameters to evaluate.

     # Let us create a Nevergrad optimization method.
     optimizer <-  ng$optimizers$registry[optimizer_name](instrumentation, budget=num_iterations, num_workers=num_workers)

     # Dummy initializations.
     nevergrad_hp <- 0
     nevergrad_hp_val <- 0
     score <- 0

     for (i in 1:num_iterations) {
         for (j in 1:num_workers) {
            nevergrad_hp[j] <- optimizer$ask()
            nevergrad_hp_val[j] <- nevergrad_hp[j]$value
         }

         # Sequential version.
         # for (j in 1:num_workers) {  # In a perfect world this would be parallel.
         #    score[j] <- norm(nevergrad_hp_val[j]
         # }
         
         # Parallel version.
         # Actually this could be asynchronous, Nevergrad is ok for that, you do not have to
         # do the tell's in the same order as the ask's.
         registerDoParallel(cores=num_workers)
         getDoParWorkers()
         foreach(i=1:num_workers) %dopar% score[j] <- norm(nevergrad_hp_val[j])

         for (j in 1:num_workers) {
            optimizer$tell(nevergrad_hp[j], score[j])
         }
     }
     print(optimizer$recommend()$value)

Don't forget the "pip=TRUE". I wasted so much time because of this :-)

For other instrumentations (discrete variables, logarithmic continuous variables...), please check `Different instrumentations: <https://github.com/facebookresearch/nevergrad/blob/master/docs/parametrization.rst>`. Or for simple examples for machine learning `machine learning <https://github.com/facebookresearch/nevergrad/blob/master/docs/machinelearning.rst>`.
