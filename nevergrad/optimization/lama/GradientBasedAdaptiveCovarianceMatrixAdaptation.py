import numpy as np


class GradientBasedAdaptiveCovarianceMatrixAdaptation:
    def __init__(
        self,
        budget,
        population_size=50,
        elite_fraction=0.2,
        initial_sigma=0.3,
        c_c=0.1,
        c_s=0.3,
        c_1=0.2,
        c_mu=0.3,
        damps=1.0,
        learning_rate=0.01,
    ):
        self.budget = budget
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.initial_sigma = initial_sigma
        self.c_c = c_c  # cumulation for C
        self.c_s = c_s  # cumulation for sigma control
        self.c_1 = c_1  # learning rate for rank-one update
        self.c_mu = c_mu  # learning rate for rank-mu update
        self.damps = damps  # damping for step-size
        self.learning_rate = learning_rate  # learning rate for gradient-based local search

    def __adaptive_covariance_matrix_adaptation(self, func, pop, mean, C, sigma):
        n_samples = self.population_size
        dim = pop.shape[1]

        new_pop = np.zeros_like(pop)
        new_scores = np.zeros(n_samples)

        for i in range(n_samples):
            z = np.random.randn(dim)
            try:
                y = np.dot(np.linalg.cholesky(C), z)
            except np.linalg.LinAlgError:
                y = np.dot(np.linalg.cholesky(C + 1e-8 * np.eye(dim)), z)
            candidate = np.clip(mean + sigma * y, -5.0, 5.0)
            new_pop[i] = candidate
            new_scores[i] = func(candidate)

        return new_pop, new_scores

    def __gradient_local_search(self, func, x):
        eps = 1e-8
        grad = np.zeros_like(x)
        fx = func(x)

        for i in range(len(x)):
            x_eps = np.copy(x)
            x_eps[i] += eps
            grad[i] = (func(x_eps) - fx) / eps

        return x - self.learning_rate * grad

    def __call__(self, func):
        np.random.seed(0)
        dim = 5
        lower_bound = -5.0
        upper_bound = 5.0

        # Initialize population
        pop = np.random.uniform(lower_bound, upper_bound, (self.population_size, dim))
        scores = np.array([func(ind) for ind in pop])

        # Global best initialization
        best_idx = np.argmin(scores)
        global_best_position = pop[best_idx]
        global_best_score = scores[best_idx]

        evaluations = self.population_size
        max_iterations = self.budget // self.population_size

        # Initialize mean, covariance matrix, and sigma
        mean = np.mean(pop, axis=0)
        C = np.cov(pop.T)
        sigma = self.initial_sigma

        # Evolution path
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim**2))

        for iteration in range(max_iterations):
            # Perform adaptive covariance matrix adaptation step
            pop, scores = self.__adaptive_covariance_matrix_adaptation(func, pop, mean, C, sigma)

            # Update global best
            best_idx = np.argmin(scores)
            if scores[best_idx] < global_best_score:
                global_best_score = scores[best_idx]
                global_best_position = pop[best_idx]

            # Apply gradient-based local search to the best individual in the population
            local_best = self.__gradient_local_search(func, global_best_position)
            local_best_score = func(local_best)
            evaluations += 1

            if local_best_score < global_best_score:
                global_best_score = local_best_score
                global_best_position = local_best

            # Update mean, covariance matrix, and sigma
            elite_count = int(self.population_size * self.elite_fraction)
            elite_idx = np.argsort(scores)[:elite_count]
            elite_pop = pop[elite_idx]
            mean_new = np.dot(np.ones(elite_count) / elite_count, elite_pop)

            ps = (1 - self.c_s) * ps + np.sqrt(self.c_s * (2 - self.c_s)) * (mean_new - mean) / sigma
            hsig = (
                np.linalg.norm(ps) / np.sqrt(1 - (1 - self.c_s) ** (2 * evaluations / self.population_size))
            ) < (1.4 + 2 / (dim + 1))
            pc = (1 - self.c_c) * pc + hsig * np.sqrt(self.c_c * (2 - self.c_c)) * (mean_new - mean) / sigma

            artmp = (elite_pop - mean) / sigma
            C = (
                (1 - self.c_1 - self.c_mu) * C
                + self.c_1 * np.outer(pc, pc)
                + self.c_mu * np.dot(artmp.T, artmp) / elite_count
            )
            sigma *= np.exp((np.linalg.norm(ps) / chi_n - 1) * self.damps)

            mean = mean_new

            evaluations += self.population_size
            if evaluations >= self.budget:
                break

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt
