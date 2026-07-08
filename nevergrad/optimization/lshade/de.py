import numpy as np  # engine for numerical computing

# Abstract class of all optimizers for continuous black-box **minimization**
from .optimizer import Optimizer


class DE(Optimizer):
    """Differential Evolution (DE).

    This is the **abstract** class for all Differential Evolution (`DE`) classes. Please use
    any of its instantiated subclasses to optimize the black-box problem at hand.

    .. note:: `DE` was proposed to solve some challenging real-world black-box problems by
       Kenneth Price and Rainer Storn, `two recipients of IEEE Evolutionary Computation
       Pioneer Award 2017 <https://tinyurl.com/456as566>`_. Although there is *few* significant
       theoretical advance till now (to our knowledge), it is **still widely used in practice**,
       owing to its often attractive search performance on many multimodal black-box functions.
       `"DE borrows the idea from Nelder&Mead of employing information from within the vector
       population to alter the search space."---[Storn&Price, 1997, JGO]
       <https://doi.org/10.1023/A:1008202821328>`_

       The popular and powerful `SciPy <https://www.nature.com/articles/s41592-019-0686-2>`_
       library has provided an open-source Python implementation for `DE` with wide applications.

    For some interesting applications of `DE`, please refer to `[Weichart et al., 2024, Psychological Review]
    <https://psycnet.apa.org/record/2024-83890-001>`_,
    `[LaBerge et al., 2024, Nature Photonics (UT Austin, TU Dresden, Fermilab, etc.)]
    <https://www.nature.com/articles/s41566-024-01475-2>`_, `[Olschewski et al., 2024, PNAS]
    <https://www.pnas.org/doi/abs/10.1073/pnas.2317751121>`_,
    `[DeWolf et al., 2024 (EPFL + MPI-IS + Harvard University)]
    <https://www.biorxiv.org/content/10.1101/2024.09.11.612513v1.abstract>`_,
    `[Higgins et al., 2023, Science]
    <https://www.science.org/doi/10.1126/science.add5190>`_,
    `[Shinn et al., 2023, Nature Neuroscience]
    <https://www.nature.com/articles/s41593-023-01299-3>`_,
    `[Staffell et al., 2023, Nature Energy (Imperial + TU Delft)]
    <https://www.nature.com/articles/s41560-023-01341-5>`_,
    `[Koob et al., 2023, Psychological Review]
    <https://psycnet.apa.org/record/2021-99615-001>`_, `[Barbosa et al., 2021, PAAP]
    <https://link.springer.com/chapter/10.1007/978-981-16-0010-4_15>`_, `[Lawson et al., 2020, AJ]
    <https://doi.org/10.3847/1538-3881/ababa6>`_, `[Event Horizon Telescope Collaboration, 2019, ApJL]
    <https://iopscience.iop.org/article/10.3847/2041-8213/ab1141>`_, `[Lawson et al., 2019, AJ]
    <https://doi.org/10.3847/1538-3881/ab3461>`_, `[Laganowsky et al., 2014, Nature]
    <https://www.nature.com/articles/nature13419>`_, just to name a few.

    Parameters
    ----------
    problem : dict
              problem arguments with the following common settings (`keys`):
                * 'fitness_function' - objective function to be **minimized** (`func`),
                * 'ndim_problem'     - number of dimensionality (`int`),
                * 'upper_boundary'   - upper boundary of search range (`array_like`),
                * 'lower_boundary'   - lower boundary of search range (`array_like`).
    options : dict
              optimizer options with the following common settings (`keys`):
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular setting (`key`):
                * 'n_individuals' - number of offspring, aka offspring population size (`int`, default: `100`).

    Attributes
    ----------
    n_individuals : `int`
                    number of offspring, aka offspring population size. For `DE`, typically a *large* (often >=100)
                    population size is used to better explore for multimodal functions. Obviously the *optimal*
                    population size is problem-dependent, which can be fine-tuned in practice.

    Methods
    -------

    References
    ----------
    Price, K.V., 2013.
    `Differential evolution.
    <https://link.springer.com/chapter/10.1007/978-3-642-30504-7_8>`_
    In Handbook of Optimization (pp. 187-214). Springer.

    Price, K.V., Storn, R.M. and Lampinen, J.A., 2005.
    `Differential evolution: A practical approach to global optimization.
    <https://link.springer.com/book/10.1007/3-540-31306-0>`_
    Springer Science & Business Media.

    https://jacobfilipp.com/DrDobbs/articles/DDJ/1997/9704/9704a/9704a.htm

    Storn, R.M. and Price, K.V. 1997.
    `Differential evolution – a simple and efficient heuristic for
    global optimization over continuous spaces.
    <https://doi.org/10.1023/A:1008202821328>`_
    Journal of Global Optimization, 11(4), pp.341–359.

    Storn, R.M., 1996, May.
    `Differential evolution design of an IIR-filter.
    <https://ieeexplore.ieee.org/document/542373>`_
    In Proceedings of IEEE International Conference on Evolutionary
    Computation (pp. 268-273). IEEE.

    Storn, R.M., 1996, June.
    `On the usage of differential evolution for function optimization.
    <https://ieeexplore.ieee.org/abstract/document/534789>`_
    In Proceedings of North American Fuzzy Information Processing (pp.
    519-523). IEEE.
    """

    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:  # number of offspring, aka offspring population size
            self.n_individuals = 100
        assert self.n_individuals > 0
        self._n_generations = 0  # number of generations
        self._printed_evaluations = self.n_function_evaluations

    def initialize(self):
        raise NotImplementedError

    def mutate(self):
        raise NotImplementedError

    def crossover(self):
        raise NotImplementedError

    def select(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def _print_verbose_info(self, fitness, y, is_print=False):
        if y is not None and self.saving_fitness:
            if not np.isscalar(y):
                fitness.extend(y)
            else:
                fitness.append(y)
        if self.verbose:
            is_verbose = (
                self._printed_evaluations != self.n_function_evaluations
            )  # to avoid repeated printing
            is_verbose_1 = (not self._n_generations % self.verbose) and is_verbose
            is_verbose_2 = self.termination_signal > 0 and is_verbose
            is_verbose_3 = is_print and is_verbose
            if is_verbose_1 or is_verbose_2 or is_verbose_3:
                # info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
                # print(info.format(
                #     self._n_generations,
                #     float(self.best_so_far_y),  # 转换为Python float
                #     float(np.min(y)),  # 转换为Python float
                #     self.n_function_evaluations
                # ))
                self._printed_evaluations = self.n_function_evaluations

    def _collect(self, fitness=None, y=None):
        self._print_verbose_info(fitness, y)
        results = Optimizer._collect(self, fitness)
        results["_n_generations"] = self._n_generations
        return results
