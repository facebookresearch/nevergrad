Open Optimization Competition 2020

The Nevergrad and IOHprofiler teams are happy to announce that we have<span class="Apple-converted-space"></span>

paired up for the Open Optimization Competition 2020.<span class="Apple-converted-space"></span>

<span class="s1">[Overview](%5Cl%20%22h.m467pnvbf91g%22)</span>

<span class="s1">[About Nevergrad and IOHprofiler](%5Cl%20%22h.qy9f65w2qt2z%22)</span>

<span class="s1">[Motivation](%5Cl%20%22h.hu3xkclziowd%22)</span>

<span class="s1">[Award Committee Members](%5Cl%20%22h.jus9rz8fudrg%22)</span>

<span class="s1">[Details about the Submissions](%5Cl%20%22h.uq48p6hhkitb%22)</span>

<span class="s1">[Recommended topics (non-exhaustive list)](%5Cl%20%22h.rbehrmpj6mdf%22)</span>

<span class="s1">[Organizers](%5Cl%20%22h.8hhl8pwbm1pk%22)</span>

Overview<span class="Apple-converted-space"></span>

Key features of the Open Optimization Competition 2020:

*   ● <span class="Apple-tab-span"></span> <span class="s1">**Two tracks:**</span><span class="Apple-converted-space"></span>

*   ○ <span class="Apple-tab-span"></span> **PERFORMANCE TRACK:** Algorithms for any of the following performance tracks  
    (classic competition, with existing baselines that are meant to be outperformed)

*   ■ <span class="Apple-tab-span"></span> One-shot optimization
*   ■ <span class="Apple-tab-span"></span> Low budget optimization
*   ■ <span class="Apple-tab-span"></span> Multi-objective optimization
*   ■ <span class="Apple-tab-span"></span> Discrete optimization
*   ■ <span class="Apple-tab-span"></span> Structured optimization<span class="Apple-converted-space"></span>
*   ■ <span class="Apple-tab-span"></span> Constrained optimization

*   We explicitly allow portfolio-based algorithms such as landscape-aware algorithm selection techniques, etc.<span class="Apple-converted-space"></span>

*   ○ <span class="Apple-tab-span"></span> **IDEAS TRACK:** Contributions to the experimental routines, including<span class="Apple-converted-space"></span>

*   ■ <span class="Apple-tab-span"></span> suggestions for performance measures and statistics,<span class="Apple-converted-space"></span>
*   ■ <span class="Apple-tab-span"></span> good benchmark problems,<span class="Apple-converted-space"></span>
*   ■ <span class="Apple-tab-span"></span> visualization of data,<span class="Apple-converted-space"></span>
*   ■ <span class="Apple-tab-span"></span> parallelization,<span class="Apple-converted-space"></span>
*   ■ <span class="Apple-tab-span"></span> reproducibility, etc.<span class="Apple-converted-space"></span>

*   ● <span class="Apple-tab-span"></span> <span class="s1">**Submissions:**</span><span class="Apple-converted-space"></span>

*   ○ <span class="Apple-tab-span"></span> All contributions are directly made to either one of the tools, Nevergrad or IOHprofiler, via pull requests on our GitHub pages.<span class="Apple-converted-space"></span>

*   <span class="s2">■ <span class="Apple-tab-span"></span> [<span class="s1">https://github.com/facebookresearch/nevergrad</span>](https://github.com/facebookresearch/nevergrad) for Nevergrad<span class="Apple-converted-space"></span></span>
*   ■ <span class="Apple-tab-span"></span> [<span class="s3">https://github.com/IOHprofiler/</span>](https://github.com/IOHprofiler/IOHanalyzer) for IOHprofiler [for IOHprofiler, please use the competition branch for your submissions]

*   All supporting material should be uploaded together with the pull request. Links to arXiv papers etc are possible and welcome, but by no means mandatory. Keep in mind that a good description of your contribution increases the chance that jury members will understand and value your contribution.<span class="Apple-converted-space"></span>

*   ○ <span class="Apple-tab-span"></span> All pull requests not yet merged on December 1 2019 or opened before June 1, 2020 are eligible for <span class="Apple-converted-space"></span> the competition
*   ○ <span class="Apple-tab-span"></span> Key principles:

*   ■ <span class="Apple-tab-span"></span> Open source: no matter how good are the results, if they can not be reproduced or the code can not be checked this is not in the scope.
*   ■ <span class="Apple-tab-span"></span> Reproducibility: if the code can not be run easily, it is also not in the scope.

*   ● <span class="Apple-tab-span"></span> <span class="s1">**Dates:<span class="Apple-converted-space"></span>**</span>
*   Submission deadline is June 1, 2020, AoE
*   ● <span class="Apple-tab-span"></span> <span class="s1">**Awards**</span>:  
    The type of awards, if any, will be announced later.
*   About Nevergrad and IOHprofiler

*   ● <span class="Apple-tab-span"></span> <span class="s1">**Nevergrad** </span>is an open source platform for derivative-free optimization: <span class="Apple-converted-space"></span> it contains a wide range of optimizers, test cases, supports multiobjective optimization and handles constraints. It automatically updates results of all experiments merged in the codebase, hence users do not need computational power for participating and getting results.
*   ● <span class="Apple-tab-span"></span> <span class="s1">**IOHprofiler** </span>is a tool for benchmarking iterative optimization heuristics such as local search variants, evolutionary algorithms, model-based algorithms, and other sequential optimization techniques. IOHprofiler has two components:<span class="Apple-converted-space"></span>

*   ○ <span class="Apple-tab-span"></span> [<span class="s3">**IOHexperimenter**</span> ](https://github.com/IOHprofiler/IOHexperimenter)for running empirical evaluations
*   ○ <span class="Apple-tab-span"></span> [<span class="s3">**IOHanalyzer**</span>](https://github.com/IOHprofiler/IOHanalyzer) for the statistical analysis and visualization of the experiment data.

*   A documentation is available here: [<span class="s3">https://iohprofiler.github.io/</span>](https://iohprofiler.github.io/)<span class="Apple-converted-space"></span>
*   At the moment both tools are rather independent. Note, however, that the nevergrad data (csv files) can be analyzed by IOHanalyzer, and that we are working towards a better compatibility between the tools.  

*   Motivation
*   We want to build open-source, user-friendly, and community-driven platforms for comparing different optimization techniques. Our key principles are reproducibility, open source, and ease of access. While we have set some first steps towards such platforms, we believe that the tools can greatly benefit from the contributions of the various communities for whom they are built.

*   Award Committee Members
*   The award committee members are:

*   - <span class="Apple-tab-span"></span> [<span class="s3">Enrique Alba</span>](http://www.lcc.uma.es/~eat/) (University of Málaga, Spain)
*   - <span class="Apple-tab-span"></span> [<span class="s3">Maxim Buzdalov</span>](https://ctlab.itmo.ru/~mbuzdalov/) (ITMO University, Russia)
*   - <span class="Apple-tab-span"></span> [<span class="s3">Josu Cerebio</span>](http://www.sc.ehu.es/ccwbayes/members/jceberio/home/) (University of the Basque country, Spain)
*   - <span class="Apple-tab-span"></span> [<span class="s3">Benjamin Doerr</span>](https://people.mpi-inf.mpg.de/~doerr/) (Ecole Polytechnique, France)
*   - <span class="Apple-tab-span"></span> [<span class="s3">Tobias Glasmachers</span>](https://www.ini.rub.de/the_institute/people/tobias-glasmachers/) (Ruhr-Universität Bochum, Germany)
*   - <span class="Apple-tab-span"></span> [<span class="s3">Manuel Lopez-Ibanez</span>](http://lopez-ibanez.eu/) (University of Manchester, UK)
*   - <span class="Apple-tab-span"></span> [<span class="s3">Katherine Mary Malan</span>](http://www.kmalan.co.za/) <span class="Apple-converted-space"></span> (University of South Africa)
*   - <span class="Apple-tab-span"></span> [<span class="s3">Luís Paquete</span>](https://apps.uc.pt/mypage/faculty/uc26679/) (Coimbra, Portugal)
*   - <span class="Apple-tab-span"></span> [<span class="s3">Jan van Rijn</span>](http://liacs.leidenuniv.nl/~rijnjnvan/) (Leiden University, The Netherlands)
*   - <span class="Apple-tab-span"></span> [<span class="s3">Marc Schoenauer</span>](https://www.lri.fr/~marc/) (Inria, France)
*   <span class="s4">- <span class="Apple-tab-span"></span> [<span class="s3">Thomas Weise</span>](http://iao.hfuu.edu.cn/team/director)</span> (Hefei University, China)  

*   **Policy re. possible conflict of interest:** Award committee members can not propose as winner a person with whom they worked directly during the previous 12 months. There is no restriction for coworkers of other committee members than oneself.
*   Submissions made by members of the organizing committee or by employees of Facebook can not be awarded. Their coworkers can be rewarded (but only for work that does not involve organizers nor Facebook employees).

*   Details about the Submissions<span class="Apple-converted-space"></span>
*   All submissions are based on pull request, which are directly made to either one of the tools, via

*   <span class="s2">● <span class="Apple-tab-span"></span> [<span class="s1">https://github.com/facebookresearch/nevergrad</span>](https://github.com/facebookresearch/nevergrad) for Nevergrad<span class="Apple-converted-space"></span></span>
*   ● <span class="Apple-tab-span"></span> [<span class="s3">https://github.com/IOHprofiler/</span>](https://github.com/IOHprofiler/IOHanalyzer) for IOHprofiler. For submissions to IOHprofiler, please use the competition branch<span class="Apple-converted-space"></span>
*   Note that in addition to submitting pull requests, you are also invited to submit a 2-page summary of your contribution to our competition branch at [<span class="s3">GECCO</span>](http://gecco-2020.sigevo.org/). Formatting requirements etc can be found on [<span class="s3">https://gecco-2020.sigevo.org/index.html/Papers+Submission+Instructions</span>](https://gecco-2020.sigevo.org/index.html/Papers+Submission+Instructions)<span class="Apple-converted-space"></span>
*   These 2-pagers appear undergo some mild reviewing process, and will be published in the companion material of GECCO, which is indexed by all major databases such as Google scholar, dblp, etc. Submissions in GECCO are a possible addition to a submission by pull-request, and do not form a competition entry by itself. Essentially, this is just a possibility to publish some information about your work in an indexed proceedings.<span class="Apple-converted-space"></span>

*   Recommended topics (non-exhaustive list)
*   We identify the following list of topics for which we feel that great contributions are possible.

*   ● <span class="Apple-tab-span"></span> Improvements in any of the following optimization categories:

*   ○ <span class="Apple-tab-span"></span> One-shot optimization
*   ○ <span class="Apple-tab-span"></span> Low budget optimization
*   ○ <span class="Apple-tab-span"></span> Multi-objective optimization
*   ○ <span class="Apple-tab-span"></span> Discrete optimization, in particular self-adaptation
*   ○ <span class="Apple-tab-span"></span> Structured optimization (e.g. almost periodic problems with several groups of variables)<span class="Apple-converted-space"></span>
*   ○ <span class="Apple-tab-span"></span> Constraint handling
*   ○ <span class="Apple-tab-span"></span> Algorithm selection and combination

*   ● <span class="Apple-tab-span"></span> Improvements in terms of benchmarking:

*   ○ <span class="Apple-tab-span"></span> Criteria in benchmarking, e.g. robustness criteria over large families of problems
*   ○ <span class="Apple-tab-span"></span> Visualization of data
*   ○ <span class="Apple-tab-span"></span> New problems (structured optimization, new classes of problems, real world problems)
*   ○ <span class="Apple-tab-span"></span> Cross-validation issues in optimization benchmarking
*   ○ <span class="Apple-tab-span"></span> Statistics
*   ○ <span class="Apple-tab-span"></span> As a beta version, we have an automatic recomputing of various benchmark results, at [<span class="s3">https://dl.fbaipublicfiles.com/nevergrad/allxps/list.html</span>](https://dl.fbaipublicfiles.com/nevergrad/allxps/list.html), including realworld benchmarks, Yet another black-box optimization benchmark (YABBOB), and many others.

*   ● <span class="Apple-tab-span"></span> Software contributions

*   ○ <span class="Apple-tab-span"></span> Distribution over clusters or grids
*   ○ <span class="Apple-tab-span"></span> Software contribution in general

*   ● <span class="Apple-tab-span"></span> Mathematically justified improvement

*   The awards will be separated in two tracks:

*   - <span class="Apple-tab-span"></span> Performance: making algorithms better or designing better algorithms;
*   - <span class="Apple-tab-span"></span> Contributions: everything people can think of, which makes the platform better for users and for science.
*   Organizers
*   In case of questions, please do not hesitate to contact the organizers of the competition. Please send all inquiries to Carola (Carola.Doerr@lip6.fr) and Olivier (oteytaud@fb.com), who will coordinate your request.<span class="Apple-converted-space"></span>

*   ● <span class="Apple-tab-span"></span> [<span class="s3">Thomas Bäck</span>](https://www.universiteitleiden.nl/en/staffmembers/thomas-back) (Leiden University)
*   <span class="s2">● <span class="Apple-tab-span"></span> [<span class="s1">Carola Doerr</span>](http://www-ia.lip6.fr/~doerr/) (CNRS)</span>
*   ● <span class="Apple-tab-span"></span> [<span class="s3">Antoine Moreau</span>](http://cloud.ip.uca.fr/~moreau/index_en.html) (Université Clermont Auvergne)
*   ● <span class="Apple-tab-span"></span> Jeremy Rapin (Facebook Artificial Intelligence Research, Paris, France)
*   ● <span class="Apple-tab-span"></span> Olivier Teytaud (Facebook Artificial Intelligence Research, Paris, France)
