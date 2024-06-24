import numpy as np


class EnhancedCovarianceGradientSearchV2:
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
    ):
        self.budget = budget
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.initial_sigma = initial_sigma
        self.c_c = c_c
        self.c_s = c_s
        self.c_1 = c_1
        self.c_mu = c_mu
        self.damps = damps
        self.learning_rate = learning_rate
        self.gradient_steps = gradient_steps

    def __adaptive_covariance_matrix_adaptation(self, func, mean, C, sigma):
        n_samples = self.population_size
        dim = mean.shape[0]

        new_pop = np.zeros((n_samples, dim))
        new_scores = np.zeros(n_samples)
        cholesky_success = False

        for _ in range(10):  # Retry up to 10 times if Cholesky fails
            try:
                chol_decomp = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                cholesky_success = True
                break
            except np.linalg.LinAlgError:
                C += 1e-6 * np.eye(dim)

        if not cholesky_success:
            chol_decomp = np.eye(dim)

        for i in range(n_samples):
            if self.budget_remaining <= 0:
                break
            z = np.random.randn(dim)
            y = np.dot(chol_decomp, z)
            candidate = np.clip(mean + sigma * y, -5.0, 5.0)
            new_pop[i] = candidate
            new_scores[i] = func(candidate)
            self.budget_remaining -= 1

        return new_pop, new_scores

    def __gradient_local_search(self, func, x):
        eps = 1e-8
        for _ in range(self.gradient_steps):
            if self.budget_remaining <= 0:
                break

            grad = np.zeros_like(x)
            fx = func(x)
            self.budget_remaining -= 1

            for i in range(len(x)):
                if self.budget_remaining <= 0:
                    break

                x_eps = np.copy(x)
                x_eps[i] += eps
                grad[i] = (func(x_eps) - fx) / eps
                self.budget_remaining -= 1

            x -= self.learning_rate * grad
            x = np.clip(x, -5.0, 5.0)

        return x

    def __hierarchical_selection(self, pop, scores):
        elite_count = int(self.population_size * self.elite_fraction)
        elite_idx = np.argsort(scores)[:elite_count]
        elite_pop = pop[elite_idx]

        diverse_count = int(self.population_size * (1 - self.elite_fraction))
        diverse_idx = np.random.choice(np.arange(len(pop)), size=diverse_count, replace=False)
        diverse_pop = pop[diverse_idx]

        return elite_pop, diverse_pop

    def __call__(self, func):
        np.random.seed(0)
        dim = 5
        lower_bound = -5.0
        upper_bound = 5.0

        self.budget_remaining = self.budget

        # Initialize populations
        pop_main = np.random.uniform(lower_bound, upper_bound, (self.population_size, dim))
        scores_main = np.array([func(ind) for ind in pop_main])
        self.budget_remaining -= self.population_size

        # Initialize global best
        global_best_score = np.min(scores_main)
        global_best_position = pop_main[np.argmin(scores_main)]

        # Initialize CMA-ES parameters
        mean = np.mean(pop_main, axis=0)
        C = np.eye(dim)
        sigma = self.initial_sigma
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4.0 * dim) + 1 / (21.0 * dim**2))

        while self.budget_remaining > 0:
            if self.budget_remaining <= 0:
                break

            # Main population update
            new_pop, new_scores = self.__adaptive_covariance_matrix_adaptation(func, mean, C, sigma)

            # Update population and scores
            pop_main = np.vstack((pop_main, new_pop))
            scores_main = np.hstack((scores_main, new_scores))

            best_idx = np.argmin(scores_main)
            if scores_main[best_idx] < global_best_score:
                global_best_score = scores_main[best_idx]
                global_best_position = pop_main[best_idx]

            # Gradient-based local search on elitist solutions
            elite_pop, _ = self.__hierarchical_selection(pop_main, scores_main)
            for i in range(len(elite_pop)):
                if self.budget_remaining <= 0:
                    break

                elite_pop[i] = self.__gradient_local_search(func, elite_pop[i])
                new_score = func(elite_pop[i])
                self.budget_remaining -= 1

                # Update if new solution is better
                idx = np.where((pop_main == elite_pop[i]).all(axis=1))[0]
                if len(idx) > 0 and new_score < scores_main[idx[0]]:
                    scores_main[idx[0]] = new_score

            best_idx = np.argmin(scores_main)
            if scores_main[best_idx] < global_best_score:
                global_best_score = scores_main[best_idx]
                global_best_position = pop_main[best_idx]

            # Hierarchical selection for diversity
            elite_pop, diverse_pop = self.__hierarchical_selection(pop_main, scores_main)

            # Update mean, covariance matrix, and sigma
            mean_new = np.mean(elite_pop, axis=0)

            ps = (1 - self.c_s) * ps + np.sqrt(self.c_s * (2 - self.c_s)) * (mean_new - mean) / sigma
            hsig = (
                np.linalg.norm(ps)
                / np.sqrt(
                    1 - (1 - self.c_s) ** (2 * (self.budget - self.budget_remaining) / self.population_size)
                )
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

            # Ensure numerical stability
            C = np.nan_to_num(C, nan=1e-10, posinf=1e-10, neginf=1e-10)
            sigma = max(1e-10, sigma)

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt
