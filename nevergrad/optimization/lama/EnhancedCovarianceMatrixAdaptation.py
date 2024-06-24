import numpy as np


class EnhancedCovarianceMatrixAdaptation:
    def __init__(self, budget, population_size=50, elite_fraction=0.2, initial_sigma=0.3):
        self.budget = budget
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.initial_sigma = initial_sigma

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

        for iteration in range(max_iterations):
            # Perform adaptive covariance matrix adaptation step
            pop, scores = self.__adaptive_covariance_matrix_adaptation(func, pop, mean, C, sigma)

            # Update global best
            best_idx = np.argmin(scores)
            if scores[best_idx] < global_best_score:
                global_best_score = scores[best_idx]
                global_best_position = pop[best_idx]

            # Update mean, covariance matrix, and sigma
            elite_count = int(self.population_size * self.elite_fraction)
            elite_idx = np.argsort(scores)[:elite_count]
            elite_pop = pop[elite_idx]
            mean = np.mean(elite_pop, axis=0)
            C = np.cov(elite_pop.T)

            # Adaptive sigma adjustment based on elite score improvements
            if iteration > 0 and np.mean(scores) < np.mean(prev_scores):
                sigma *= 1.2  # Increase sigma if improvement
            else:
                sigma *= 0.8  # Decrease sigma otherwise

            prev_scores = scores
            evaluations += self.population_size
            if evaluations >= self.budget:
                break

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt
