import numpy as np


class MultiScaleAdaptiveHybridOptimization:
    def __init__(self, budget, population_size=30):
        self.budget = budget
        self.population_size = population_size

    def gradient_estimation(self, func, x):
        epsilon = 1e-8
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_pos = np.copy(x)
            x_neg = np.copy(x)
            x_pos[i] += epsilon
            x_neg[i] -= epsilon
            grad[i] = (func(x_pos) - func(x_neg)) / (2 * epsilon)
        return grad

    def differential_evolution(self, func, pop, scores):
        new_pop = np.copy(pop)
        new_scores = np.copy(scores)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.mutation_rate * (b - c), -5.0, 5.0)
            cross_points = np.random.rand(len(mutant)) < self.crossover_rate
            if not np.any(cross_points):
                cross_points[np.random.randint(0, len(mutant))] = True
            trial = np.where(cross_points, mutant, pop[i])
            f = func(trial)
            if f < scores[i]:
                new_scores[i] = f
                new_pop[i] = trial
        return new_pop, new_scores

    def local_search(self, func, pop, scores):
        new_pop = np.copy(pop)
        new_scores = np.copy(scores)
        for i in range(self.population_size):
            grad = self.gradient_estimation(func, pop[i])
            candidate = np.clip(pop[i] - self.learning_rate * grad, -5.0, 5.0)
            f = func(candidate)
            if f < scores[i]:
                new_scores[i] = f
                new_pop[i] = candidate
        return new_pop, new_scores

    def adaptive_parameters(self, iteration, max_iterations):
        self.mutation_rate = 0.5 + 0.3 * (1 - np.cos(np.pi * iteration / max_iterations))
        self.crossover_rate = 0.5 + 0.3 * (1 - np.cos(np.pi * iteration / max_iterations))
        self.learning_rate = 0.01 * np.exp(-iteration / max_iterations)

    def multiscale_sampling(self, pop, scale):
        new_pop = np.copy(pop)
        for i in range(self.population_size):
            perturbation = np.random.normal(0, scale, size=pop[i].shape)
            new_pop[i] = np.clip(pop[i] + perturbation, -5.0, 5.0)
        return new_pop

    def __call__(self, func):
        np.random.seed(0)
        dim = 5
        lower_bound = -5.0
        upper_bound = 5.0

        # Initialize population
        pop = np.random.uniform(lower_bound, upper_bound, (self.population_size, dim))
        scores = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(scores)
        global_best_position = pop[best_idx]
        global_best_score = scores[best_idx]

        evaluations = self.population_size
        max_iterations = (self.budget // self.population_size) * 2

        iteration = 0
        while evaluations < self.budget:
            self.adaptive_parameters(iteration, max_iterations)

            scale = 1.0 - (iteration / max_iterations)  # Multiscale sampling rate

            # Apply multiscale sampling
            pop = self.multiscale_sampling(pop, scale)
            scores = np.array([func(ind) for ind in pop])

            # Perform differential evolution step
            pop, scores = self.differential_evolution(func, pop, scores)
            evaluations += self.population_size

            # Update global best from differential evolution
            best_idx = np.argmin(scores)
            if scores[best_idx] < global_best_score:
                global_best_score = scores[best_idx]
                global_best_position = pop[best_idx]

            if evaluations >= self.budget:
                break

            # Perform local search step
            pop, scores = self.local_search(func, pop, scores)
            evaluations += self.population_size

            # Update global best from local search
            best_idx = np.argmin(scores)
            if scores[best_idx] < global_best_score:
                global_best_score = scores[best_idx]
                global_best_position = pop[best_idx]

            iteration += 1

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt
