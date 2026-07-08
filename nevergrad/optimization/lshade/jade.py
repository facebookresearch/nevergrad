import numpy as np  # engine for numerical computing
from scipy.stats import cauchy  # Cauchy continuous random variable

from .de import DE  # abstract class of all differential evolution (DE)


class JADE(DE):
    """Adaptive Differential Evolution (JADE).

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
                * 'p'             - level of greediness of mutation strategy (`float`, default: `0.05`),
                * 'c'             - life span (`float`, default: `0.1`),
                * 'is_bound'      - flag to limit all samplings inside the search range (`boolean`, default: `False`).

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/bddsba5s>`_ for more details.

    Attributes
    ----------
    c             : `float`
                    life span.
    is_bound      : `boolean`
                    flag to limit all samplings inside the search range.
    median        : `float`
                    median of Cauchy distribution for adaptation of mutation factor.
    mu            : `float`
                    mean of normal distribution for adaptation of crossover probability.
    n_individuals : `int`
                    number of offspring, offspring population size.
    p             : `float`
                    level of greediness of mutation strategy.

    References
    ----------
    Zhang, J., and Sanderson, A. C. 2009.
    `JADE: Adaptive differential evolution with optional external archive.
    <https://ieeexplore.ieee.org/document/5208221/>`_
    IEEE Transactions on Evolutionary Computation, 13(5), pp.945–958.
    """

    def __init__(self, problem, options):
        DE.__init__(self, problem, options)
        self.mu = options.get(
            "mu", 0.5
        )  # mean of normal distribution for adaptation of crossover probabilities
        self.median = options.get(
            "median", 0.5
        )  # location of Cauchy distribution for adaptation of mutation factor
        self.p = options.get("p", 0.05)  # level of greediness of the mutation strategy
        assert 0.0 <= self.p <= 1.0
        self.c = options.get("c", 0.1)  # life span
        assert 0.0 <= self.c <= 1.0
        self.is_bound = options.get("is_bound", False)

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(
            self.initial_lower_boundary,
            self.initial_upper_boundary,
            size=(self.n_individuals, self.ndim_problem),
        )  # population
        y = np.empty((self.n_individuals,))  # fitness
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        a = np.empty((0, self.ndim_problem))  # set of archived inferior solutions
        return x, y, a

    def bound(self, x=None, xx=None):
        if not self.is_bound:
            return x
        for k in range(self.n_individuals):
            idx = np.array(x[k] < self.lower_boundary)
            if idx.any():
                x[k][idx] = (self.lower_boundary + xx[k])[idx] / 2.0
            idx = np.array(x[k] > self.upper_boundary)
            if idx.any():
                x[k][idx] = (self.upper_boundary + xx[k])[idx] / 2.0
        return x

    def mutate(self, x=None, y=None, a=None):
        x_mu = np.empty((self.n_individuals, self.ndim_problem))  # mutated population
        f_mu = np.empty((self.n_individuals,))  # mutated mutation factors
        order = np.argsort(y)[
            : int(np.ceil(self.p * self.n_individuals))
        ]  # index of the [100*p]% best individuals
        x_p = x[self.rng_optimization.choice(order, (self.n_individuals,))]
        x_un = np.vstack((np.copy(x), a))  # archive
        for k in range(self.n_individuals):
            f_mu[k] = cauchy.rvs(loc=self.median, scale=0.1, random_state=self.rng_optimization)
            while f_mu[k] <= 0.0:
                f_mu[k] = cauchy.rvs(loc=self.median, scale=0.1, random_state=self.rng_optimization)
            if f_mu[k] > 1.0:
                f_mu[k] = 1.0
            r1 = self.rng_optimization.choice([i for i in range(self.n_individuals) if i != k])
            r2 = self.rng_optimization.choice([i for i in range(len(x_un)) if i != k and i != r1])
            x_mu[k] = x[k] + f_mu[k] * (x_p[k] - x[k]) + f_mu[k] * (x[r1] - x_un[r2])
        return x_mu, f_mu

    def crossover(self, x_mu=None, x=None):
        x_cr = np.copy(x)
        p_cr = self.rng_optimization.normal(self.mu, 0.1, (self.n_individuals,))  # crossover probabilities
        # truncate to [0, 1]
        p_cr = np.minimum(np.maximum(p_cr, 0.0), 1.0)
        for k in range(self.n_individuals):
            i_rand = self.rng_optimization.integers(self.ndim_problem)
            for i in range(self.ndim_problem):
                if (i == i_rand) or (self.rng_optimization.random() < p_cr[k]):
                    x_cr[k, i] = x_mu[k, i]
        return x_cr, p_cr

    def select(self, args=None, x=None, y=None, x_cr=None, a=None, f_mu=None, p_cr=None):
        f = np.empty((0,))  # set of all successful mutation factors
        p = np.empty((0,))  # set of all successful crossover probabilities
        for k in range(self.n_individuals):
            if self._check_terminations():
                break
            yy = self._evaluate_fitness(x_cr[k], args)
            if yy < y[k]:
                a = np.vstack((a, x[k]))  # archive of the inferior solution
                f = np.hstack((f, f_mu[k]))  # archive of the successful mutation factor
                p = np.hstack((p, p_cr[k]))  # archive of the successful crossover probability
                x[k] = x_cr[k]
                y[k] = yy
        if len(p) != 0:  # for mean update of normal distribution
            self.mu = (1.0 - self.c) * self.mu + self.c * np.mean(p)
        if len(f) != 0:  # for location update of Cauchy distribution
            self.median = (1.0 - self.c) * self.median + self.c * np.sum(np.power(f, 2)) / np.sum(f)
        return x, y, a

    def iterate(self, x=None, y=None, a=None, args=None):
        x_mu, f_mu = self.mutate(x, y, a)
        x_cr, p_cr = self.crossover(x_mu, x)
        x_cr = self.bound(x_cr, x)
        x, y, a = self.select(args, x, y, x_cr, a, f_mu, p_cr)
        # randomly remove solutions to keep the archive size fixed
        if len(a) > self.n_individuals:
            a = np.delete(a, self.rng_optimization.choice(len(a), (len(a) - self.n_individuals,), False), 0)
        self._n_generations += 1
        return x, y, a

    def optimize(self, fitness_function=None, args=None):
        fitness = DE.optimize(self, fitness_function)
        x, y, a = self.initialize(args)
        while not self._check_terminations():
            self._print_verbose_info(fitness, y)
            x, y, a = self.iterate(x, y, a, args)
        return self._collect(fitness, y)
