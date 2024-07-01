import numpy as np


class RobustCovarianceMatrixAdaptationMemeticSearch:
    def __init__(
        self, budget, population_size=50, memetic_rate=0.5, elite_fraction=0.2, learning_rate=0.01, sigma=0.3
    ):
        self.budget = budget
        self.population_size = population_size
        self.memetic_rate = memetic_rate
        self.elite_fraction = elite_fraction
        self.learning_rate = learning_rate
        self.sigma = sigma

    def gradient_estimation(self, func, x, h=1e-8):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x1 = np.copy(x)
            x2 = np.copy(x)
            x1[i] += h
            x2[i] -= h
            grad[i] = (func(x1) - func(x2)) / (2 * h)
        return grad

    def local_search(self, func, x, score):
        grad = self.gradient_estimation(func, x)
        candidate = np.clip(x - self.learning_rate * grad, -5.0, 5.0)
        f = func(candidate)
        if f < score:
            return candidate, f
        return x, score

    def covariance_matrix_adaptation(self, func, pop, scores, mean, C):
        n_samples = len(pop)
        dim = pop.shape[1]

        new_pop = np.zeros_like(pop)
        new_scores = np.zeros(n_samples)

        for i in range(n_samples):
            z = np.random.randn(dim)
            try:
                y = np.dot(np.linalg.cholesky(C), z)
            except np.linalg.LinAlgError:
                y = np.dot(np.linalg.cholesky(C + 1e-8 * np.eye(dim)), z)
            candidate = np.clip(mean + self.sigma * y, -5.0, 5.0)
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

        # Initialize mean and covariance matrix
        mean = np.mean(pop, axis=0)
        C = np.cov(pop.T)

        for iteration in range(max_iterations):
            # Perform covariance matrix adaptation step
            pop, scores = self.covariance_matrix_adaptation(func, pop, scores, mean, C)

            # Perform memetic local search
            for i in range(self.population_size):
                if np.random.rand() < self.memetic_rate:
                    pop[i], scores[i] = self.local_search(func, pop[i], scores[i])

            # Update global best
            best_idx = np.argmin(scores)
            if scores[best_idx] < global_best_score:
                global_best_score = scores[best_idx]
                global_best_position = pop[best_idx]

            # Update mean and covariance matrix
            elite_count = int(self.population_size * self.elite_fraction)
            elite_idx = np.argsort(scores)[:elite_count]
            elite_pop = pop[elite_idx]
            mean = np.mean(elite_pop, axis=0)
            C = np.cov(elite_pop.T)

            evaluations += self.population_size
            if evaluations >= self.budget:
                break

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt
