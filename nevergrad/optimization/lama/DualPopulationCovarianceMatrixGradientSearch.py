import numpy as np


class DualPopulationCovarianceMatrixGradientSearch:
    def __init__(
        self,
        budget,
        population_size=50,
        elite_fraction=0.2,
        initial_sigma=0.5,
        c_c=0.1,
        c_s=0.3,
        c_1=0.2,
        c_mu=0.3,
        damps=1.0,
        learning_rate=0.001,
        gradient_steps=10,
        gradient_search_fraction=0.5,
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
        self.gradient_steps = gradient_steps  # number of gradient descent steps
        self.gradient_search_fraction = (
            gradient_search_fraction  # fraction of budget allocated to gradient search
        )

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

    def __gradient_local_search(self, func, x, budget):
        eps = 1e-8
        for _ in range(self.gradient_steps):
            if budget <= 0:
                break

            grad = np.zeros_like(x)
            fx = func(x)
            budget -= 1

            for i in range(len(x)):
                x_eps = np.copy(x)
                x_eps[i] += eps
                grad[i] = (func(x_eps) - fx) / eps
                budget -= 1
                if budget <= 0:
                    break

            x -= self.learning_rate * grad
            x = np.clip(x, -5.0, 5.0)

        return x, budget

    def __hierarchical_selection(self, pop, scores):
        elite_count = int(self.population_size * self.elite_fraction)
        elite_idx = np.argsort(scores)[:elite_count]
        elite_pop = pop[elite_idx]

        diverse_count = int(self.population_size * (1 - self.elite_fraction))
        diverse_idx = np.argsort(scores)[-diverse_count:]
        diverse_pop = pop[diverse_idx]

        return elite_pop, diverse_pop

    def __call__(self, func):
        np.random.seed(0)
        dim = 5
        lower_bound = -5.0
        upper_bound = 5.0

        # Initialize populations
        pop_main = np.random.uniform(lower_bound, upper_bound, (self.population_size, dim))
        scores_main = np.array([func(ind) for ind in pop_main])

        pop_grad = np.copy(pop_main)
        scores_grad = np.copy(scores_main)

        evaluations = self.population_size * 2
        max_iterations = self.budget // (self.population_size * 2)

        # Initialize global best
        global_best_score = np.inf
        global_best_position = None

        for pop, scores in [(pop_main, scores_main), (pop_grad, scores_grad)]:
            best_idx = np.argmin(scores)
            if scores[best_idx] < global_best_score:
                global_best_score = scores[best_idx]
                global_best_position = pop[best_idx]

        for iteration in range(max_iterations):
            # Main population update
            mean = np.mean(pop_main, axis=0)
            C = np.cov(pop_main.T)
            sigma = self.initial_sigma

            pop_main, scores_main = self.__adaptive_covariance_matrix_adaptation(
                func, pop_main, mean, C, sigma
            )

            best_idx = np.argmin(scores_main)
            if scores_main[best_idx] < global_best_score:
                global_best_score = scores_main[best_idx]
                global_best_position = pop_main[best_idx]

            # Gradient-based local search population update
            elite_pop, _ = self.__hierarchical_selection(pop_main, scores_main)
            budget_remaining = int(self.gradient_search_fraction * (self.budget - evaluations))

            for i in range(len(elite_pop)):
                elite_pop[i], budget_remaining = self.__gradient_local_search(
                    func, elite_pop[i], budget_remaining
                )
                scores_grad[i] = func(elite_pop[i])

            best_idx = np.argmin(scores_grad)
            if scores_grad[best_idx] < global_best_score:
                global_best_score = scores_grad[best_idx]
                global_best_position = elite_pop[best_idx]

            # Hierarchical selection
            elite_pop, diverse_pop = self.__hierarchical_selection(pop_main, scores_main)

            # Update mean, covariance matrix, and sigma
            mean_new = np.mean(elite_pop, axis=0)

            if iteration == 0:
                dim = elite_pop.shape[1]
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                chi_n = np.sqrt(dim) * (1 - 1 / (4.0 * dim) + 1 / (21.0 * dim**2))

            ps = (1 - self.c_s) * ps + np.sqrt(self.c_s * (2 - self.c_s)) * (mean_new - mean) / sigma
            hsig = (
                np.linalg.norm(ps) / np.sqrt(1 - (1 - self.c_s) ** (2 * evaluations / self.population_size))
            ) < (1.4 + 2 / (dim + 1))
            pc = (1 - self.c_c) * pc + hsig * np.sqrt(self.c_c * (2 - self.c_c)) * (mean_new - mean) / sigma

            artmp = (elite_pop - mean) / sigma
            C = (
                (1 - self.c_1 - self.c_mu) * C
                + self.c_1 * np.outer(pc, pc)
                + self.c_mu * np.dot(artmp.T, artmp) / elite_pop.shape[0]
            )
            sigma *= np.exp((np.linalg.norm(ps) / chi_n - 1) * self.damps)

            mean = mean_new

            evaluations += self.population_size * 2

            if evaluations >= self.budget:
                break

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt
