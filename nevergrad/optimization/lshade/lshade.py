import numpy as np  # engine for numerical computing
from scipy.stats import cauchy  # Cauchy continuous random variable

from .jade import JADE  # adaptive differential evolution (JADE)
from .de import DE


class LSHADE(JADE):
    """Success-History based Adaptive Differential Evolution (SHADE).

    Parameters
    ----------
    problem : `dict`
              problem arguments with the following common settings (`keys`):
                * 'fitness_function' - objective function to be **minimized** (`func`),
                * 'ndim_problem'     - number of dimensionality (`int`),
                * 'upper_boundary'   - upper boundary of search range (`array_like`),
                * 'lower_boundary'   - lower boundary of search range (`array_like`).
    options : `dict`
              optimizer options with the following common settings (`keys`):
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'n_individuals' - number of offspring, aka offspring population size (`int`, default: `100`),
                * 'mu'            - mean of normal distribution for adaptation of crossover probability (`float`,
                  default: `0.5`),
                * 'median'        - median of Cauchy distribution for adaptation of mutation factor (`float`,
                  default: `0.5`),
                *  'h'            - length of historical memory (`int`, default: `100`).

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/vm3w7se4>`_ for more details.

    Attributes
    ----------
    h             : `int`
                    length of historical memory.
    median        : `float`
                    median of Cauchy distribution for adaptation of mutation factor.
    mu            : `float`
                    mean of normal distribution for adaptation of crossover probability.
    n_individuals : `int`
                    number of offspring, aka offspring population size.

    References
    ----------
    Tanabe, R. and Fukunaga, A., 2013, June.
    `Success-history based parameter adaptation for differential evolution.
    <https://ieeexplore.ieee.org/document/6557555>`_
    In IEEE Congress on Evolutionary Computation (pp. 71-78). IEEE.
    """

    def __init__(self, problem, options, rand_seed, optimal_value):
        JADE.__init__(self, problem, options)
        self.h = options.get("h", 100)  # length of historical memory
        assert 0 < self.h
        self.m_mu = np.ones(self.h) * self.mu  # means of normal distribution
        self.m_median = np.ones(self.h) * self.median  # medians of Cauchy distribution
        self._k = 0  # index to update
        self.p_min = 2.0 / self.n_individuals
        self.initial_pop_size = self.n_individuals

        # set seed
        self.rng_initialization = np.random.default_rng(rand_seed)
        self.rng_optimization = np.random.default_rng(rand_seed)

        # restart
        self.optimal_value = optimal_value
        self.restart_threshold = 5e4

    def initialize_1(self, args=None):
        pass

    def initialize_2(self, args=None):
        pass

    def initialize_3(self, args=None):
        pass

    def crossover_1(self, args=None):
        pass

    def crossover_2(self, args=None):
        pass

    def select_initialize(self, index=0):
        if index == 0:
            return self.initialize_1()
        elif index == 1:
            return self.initialize_2()
        else:
            return self.initialize_3()

    def mutate(self, x=None, y=None, a=None):
        x_mu = np.empty((self.n_individuals, self.ndim_problem))  # mutated population
        f_mu = np.empty((self.n_individuals,))  # mutated mutation factors
        x_un = np.vstack((np.copy(x), a))  # union of population x and archive a
        r = self.rng_optimization.choice(self.h, (self.n_individuals,))
        order = np.argsort(y)[:]
        p = (0.2 - self.p_min) * self.rng_optimization.random((self.n_individuals,)) + self.p_min
        idx = [order[self.rng_optimization.choice(int(i))] for i in np.ceil(p * self.n_individuals)]
        for k in range(self.n_individuals):
            f_mu[k] = cauchy.rvs(loc=self.m_median[r[k]], scale=0.1, random_state=self.rng_optimization)
            while f_mu[k] <= 0.0:
                f_mu[k] = cauchy.rvs(loc=self.m_median[r[k]], scale=0.1, random_state=self.rng_optimization)
            if f_mu[k] > 1.0:
                f_mu[k] = 1.0
            r1 = self.rng_optimization.choice([i for i in range(self.n_individuals) if i != k])
            r2 = self.rng_optimization.choice([i for i in range(len(x_un)) if i != k and i != r1])
            x_mu[k] = x[k] + f_mu[k] * (x[idx[k]] - x[k]) + f_mu[k] * (x[r1] - x_un[r2])
        return x_mu, f_mu, r

    def crossover(self, x_mu=None, x=None, r=None):
        x_cr = np.copy(x)
        p_cr = np.empty((self.n_individuals,))  # crossover probabilities
        for k in range(self.n_individuals):
            p_cr[k] = self.rng_optimization.normal(self.m_mu[r[k]], 0.1)
            p_cr[k] = np.minimum(np.maximum(p_cr[k], 0.0), 1.0)
            i_rand = self.rng_optimization.integers(self.ndim_problem)
            for i in range(self.ndim_problem):
                if (i == i_rand) or (self.rng_optimization.random() < p_cr[k]):
                    x_cr[k, i] = x_mu[k, i]
        return x_cr, p_cr

    def select(self, args=None, x=None, y=None, x_cr=None, a=None, f_mu=None, p_cr=None):
        # set successful mutation factors, crossover probabilities, fitness differences
        f, p, d = np.empty((0,)), np.empty((0,)), np.empty((0,))
        for k in range(self.n_individuals):
            if self._check_terminations():
                break
            yy = self._evaluate_fitness(x_cr[k], args)
            if yy < y[k]:
                a = np.vstack((a, x[k]))  # archive of inferior solutions
                f = np.hstack((f, f_mu[k]))  # archive of successful mutation factors
                p = np.hstack((p, p_cr[k]))  # archive of successful crossover probabilities
                d = np.hstack((d, y[k] - yy))  # archive of successful fitness differences
                x[k], y[k] = x_cr[k], yy
        if (len(p) != 0) and (len(f) != 0):
            w = d / np.sum(d)
            self.m_mu[self._k] = np.sum(w * p)  # for normal distribution
            self.m_median[self._k] = np.sum(w * np.power(f, 2)) / np.sum(w * f)  # for Cauchy distribution
            self._k = (self._k + 1) % self.h
        return x, y, a

    def change_population(self, x=None, y=None, a=None, args=None):
        max_iterations = max(
            2, self.max_function_evaluations // self.initial_pop_size
        )  # Ensure at least 2 iterations
        reduction_factor = (self.initial_pop_size - 4) / (max_iterations - 1)
        self.n_individuals = max(4, int(self.initial_pop_size - self._n_generations * reduction_factor))

        # Select the best individuals to form the new population
        if len(a) > self.n_individuals:
            indices = np.argsort(y)[: self.n_individuals]
            x = x[indices]
            y = y[indices]
            a = np.delete(a, self.rng_optimization.choice(len(a), (len(a) - self.n_individuals,), False), 0)
        else:
            # If the archive size is less than the new population size, keep it as is
            pass
        return x, y, a

    def iterate(self, x=None, y=None, a=None, args=None):
        x_mu, f_mu, r = self.mutate(x.copy(), y.copy(), a.copy())
        if self.max_function_evaluations <= 500000:
            x_cr, p_cr = self.crossover_2(x_mu.copy(), x.copy(), r.copy())
        else:
            x_cr, p_cr = self.crossover_1(x_mu.copy(), x.copy(), r.copy())
        x_cr = self.bound(x_cr, x)
        x, y, a = self.select(args, x, y, x_cr, a, f_mu, p_cr)
        x, y, a = self.change_population(x.copy(), y.copy(), a.copy())
        self._n_generations += 1
        return x, y, a

    def check_not_improving(self):
        if self.counter_early_stopping >= self.restart_threshold:
            self.counter_early_stopping = 0
            self.base_early_stopping = -np.inf
            return True
        else:
            return False

    def optimize(self, fitness_function=None, args=None):
        fitness = DE.optimize(self, fitness_function)

        if self.max_function_evaluations <= 500000:
            x, y, a = self.select_initialize(0)
            while not self._check_terminations():
                self._print_verbose_info(fitness, y)
                if self.check_not_improving():
                    x, y, a = self.select_initialize(0)
                x, y, a = self.iterate(x, y, a, args)
        else:
            x, y, a = self.select_initialize(1)
            has_fallback = False
            while not self._check_terminations():
                self._print_verbose_info(fitness, y)
                if not has_fallback and self.check_not_improving():
                    x, y, a = self.select_initialize(0)
                    has_fallback = True
                x, y, a = self.iterate(x, y, a, args)

        return self._collect(fitness, y)
