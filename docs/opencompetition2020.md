# Open Optimization Competition 2020

The Nevergrad and IOHprofiler teams are happy to announce that we have
paired up for the Open Optimization Competition 2020.



## Overview
All kind of contributions (under the form of pull requests) to Nevergrad and IOHprofiler are eligible.
For example, we distinguish two (non exclusive) categories for submission.

### Performance: Improvements for any of the following performance tracks
(classic competition, with existing baselines that are meant to be outperformed)
- One-shot optimization
- Low budget optimization
- Multi-objective optimization
- Discrete optimization
- Structured optimization
- Constrained optimization

We explicitly allow portfolio-based algorithms such as landscape-aware algorithm selection techniques, etc.

### New ideas & others: contributions to the experimental routines, including
- suggestions for performance measures and statistics,
- good benchmark problems,
- modular algorithm frameworks,
- visualization of data,
- parallelization,
- reproducibility,
- etc.

## Submission
All contributions are directly made to either one of the tools, Nevergrad or IOHprofiler, via pull requests on our GitHub pages. You do not have to do anything else than a pull request:
- [https://github.com/facebookresearch/nevergrad](https://github.com/facebookresearch/nevergrad) for Nevergrad
- [https://github.com/IOHprofiler/](https://github.com/IOHprofiler/) for IOHprofiler [for IOHprofiler, please use the competition branch for your submissions]

All supporting material should be uploaded together with the pull request. Links to arXiv papers etc are possible and welcome, but by no means mandatory. Keep in mind that a good description of your contribution increases the chance that jury members will understand and value your contribution.
All pull requests not yet merged on December 1 2019 and opened before June 1, 2020 are eligible for the competition

Nevergrad requires a CLA for contributors (see in the "Contributing to Nevergad" section of the [documentation](https://facebookresearch.github.io/nevergrad/)).

## Key principles
- Open source: no matter how good the results are, if they can not be reproduced or the code can not be checked, this is not in the scope.
- Reproducibility: if the code can not be run easily, it is also not in the scope.

## Dates
**Submission deadline is June 1, 2020, AoE**
All pull requests active between December 1st 2019 and June 1st, 2020 are eligible.

## Awards:
Up to 12 000 euros of awards, to be used for traveling to PPSN or GECCO 2020 or 2021, distributed over several winners.
In addition, a limited number of registration fee waivers are available for PPSN 2020.

## About Nevergrad and IOHprofiler

### Nevergrad
Nevergrad is an open source platform for derivative-free optimization:  it contains a wide range of optimizers, test cases, supports multiobjective optimization and handles constraints. It automatically updates results of all experiments merged in the codebase, hence users do not need computational power for participating and getting results.

### IOHProfiler
IOHprofiler is a tool for benchmarking iterative optimization heuristics such as local search variants, evolutionary algorithms, model-based algorithms, and other sequential optimization techniques. IOHprofiler has two components:
- IOHexperimenter for running empirical evaluations
- IOHanalyzer for the statistical analysis and visualization of the experiment data.
A documentation is available here: [https://iohprofiler.github.io/](https://iohprofiler.github.io/)

## Motivation
We want to build open-source, user-friendly, and community-driven platforms for comparing different optimization techniques. Our key principles are reproducibility, open source, and ease of access. While we have set some first steps towards such platforms, we believe that the tools can greatly benefit from the contributions of the various communities for whom they are built.

## Award Committee Members
The award committee members are:
- Enrique Alba (University of Málaga, Spain)
- Maxim Buzdalov (ITMO University, Russia)
- Josu Cerebio (University of the Basque country, Spain)
- Benjamin Doerr (Ecole Polytechnique, France)
- Tobias Glasmachers (Ruhr-Universität Bochum, Germany)
- Manuel Lopez-Ibanez (University of Manchester, UK)
- Katherine Mary Malan  (University of South Africa)
- Luís Paquete (University of Coimbra, Portugal)
- Jan van Rijn (Leiden University, The Netherlands)
- Marc Schoenauer (Inria Saclay, France)
- Thomas Weise (Hefei University, China)

Our policy re. possible conflict of interest: Award committee members can not propose as winner a person with whom they worked directly during the previous 12 months. There is no restriction for coworkers of other committee members than oneself.
Submissions made by members of the organizing committee or by employees of Facebook can not be awarded. Their coworkers can be rewarded (but only for work that does not involve organizers nor Facebook employees).

## Details about the Submissions
All submissions are based on pull request, which are directly made to either one of the tools, via
- [https://github.com/facebookresearch/nevergrad](https://github.com/facebookresearch/nevergrad) for Nevergrad
- [https://github.com/IOHprofiler/](https://github.com/IOHprofiler/) for IOHprofiler. For submissions to IOHprofiler, please use the competition branch

## Recommended topics (non-exhaustive list)
We identify the following list of topics for which we feel that great contributions are possible.
This does not exclude other forms of contributions; all pull requests are eligible; please just do a pull request, follow the discussion there, you might specify a track in the conversation but this is not mandatory.

### Improvements in any of the following optimization categories:
- One-shot optimization
- Low budget optimization
- Multi-objective optimization
- Discrete optimization, in particular self-adaptation
- Structured optimization (e.g. almost periodic problems with several groups of variables)
- Constraint handling
- Algorithm selection and combination
- As a beta version, we have an automatic recomputing of various benchmark results, at [https://dl.fbaipublicfiles.com/nevergrad/allxps/list.html](https://dl.fbaipublicfiles.com/nevergrad/allxps/list.html), including realworld benchmarks, Yet another black-box optimization benchmark (YABBOB), and many others.

### Improvements in terms of benchmarking:
- Criteria in benchmarking, e.g. robustness criteria over large families of problems
- Visualization of data
- New problems (structured optimization, new classes of problems, real world problems)
- Cross-validation issues in optimization benchmarking
- Statistics

### Software contributions
- Distribution over clusters or grids
- Software contribution in general
- Mathematically justified improvement

The awards will be separated in two tracks:
- Performance: making algorithms better or designing better algorithms;
- Contributions: everything people can think of, which makes the platform better for users and for science.
Please note that tracks are indicative; you do not have to specify a track.

## Organizers
In case of questions, please do not hesitate to contact the organizers of the competition. Please send all inquiries to Carola (Carola.Doerr@lip6.fr) and Olivier (oteytaud@fb.com), who will coordinate your request.
- Thomas Bäck (Leiden University, The Netherlands)
- Carola Doerr (CNRS, Sorbonne Université, Paris, France)
- Antoine Moreau (Université Clermont Auvergne)
- Jeremy Rapin (Facebook Artificial Intelligence Research, Paris, France)
- Baptiste Roziere (Facebook Artificial Intelligence Research, Paris, France)
- Ofer M. Shir (Tel-Hai College and Migal Institute, Israel)
- Olivier Teytaud (Facebook Artificial Intelligence Research, Paris, France)

For all details, please check [https://gecco-2020.sigevo.org/index.html/Workshops](https://gecco-2020.sigevo.org/index.html/Workshops)
