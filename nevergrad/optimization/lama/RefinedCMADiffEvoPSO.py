import numpy as np


class RefinedCMADiffEvoPSO:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.initial_pop_size = 60
        self.min_pop_size = 20
        self.initial_F = 0.5
        self.initial_CR = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.7
        self.restart_threshold = 50
        self.sigma = 0.3
        self.diversity_threshold = 0.1  # Threshold for population diversity

    def initialize_population(self, bounds):
        population = np.random.uniform(bounds.lb, bounds.ub, (self.initial_pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.initial_pop_size, self.dim))
        return population, velocities

    def select_parents(self, population):
        idxs = np.random.choice(range(population.shape[0]), 3, replace=False)
        return population[idxs]

    def mutate(self, parent1, parent2, parent3, F):
        return parent1 + F * (parent2 - parent3)

    def crossover(self, target, mutant, CR):
        j_rand = np.random.randint(self.dim)
        trial = np.array(
            [mutant[j] if np.random.rand() < CR or j == j_rand else target[j] for j in range(self.dim)]
        )
        return trial

    def cma_update(self, population, mean, cov_matrix):
        new_samples = np.random.multivariate_normal(mean, cov_matrix, size=population.shape[0])
        return np.clip(new_samples, -5.0, 5.0)

    def diversity(self, population):
        return np.mean(np.std(population, axis=0))

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        bounds = func.bounds

        population, velocities = self.initialize_population(bounds)
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        evaluations = self.initial_pop_size

        no_improvement_counter = 0

        mean = np.mean(population, axis=0)
        cov_matrix = np.cov(population, rowvar=False)

        while evaluations < self.budget:
            current_pop_size = max(
                self.min_pop_size, int(self.initial_pop_size * ((self.budget - evaluations) / self.budget))
            )
            new_population = np.zeros_like(population[:current_pop_size])
            fitness = np.zeros(current_pop_size)

            for i in range(current_pop_size):
                parent1, parent2, parent3 = self.select_parents(population)
                F = np.random.uniform(0.4, 0.9)
                CR = np.random.uniform(0.6, 1.0)
                mutant = self.mutate(parent1, parent2, parent3, F)
                trial = self.crossover(population[i], mutant, CR)

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < personal_best_scores[i]:
                    personal_best_positions[i] = trial
                    personal_best_scores[i] = trial_fitness

                if personal_best_scores[i] < global_best_score:
                    global_best_position = personal_best_positions[i]
                    global_best_score = personal_best_scores[i]
                    no_improvement_counter = 0
                else:
                    no_improvement_counter += 1

                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * np.random.rand(self.dim) * (personal_best_positions[i] - population[i])
                    + self.c2 * np.random.rand(self.dim) * (global_best_position - population[i])
                )
                new_population[i] = population[i] + velocities[i]
                new_population[i] = np.clip(new_population[i], bounds.lb, bounds.ub)
                fitness[i] = func(new_population[i])
                evaluations += 1

            population = new_population
            personal_best_positions = personal_best_positions[:current_pop_size]
            personal_best_scores = personal_best_scores[:current_pop_size]
            velocities = velocities[:current_pop_size]

            if np.min(fitness) < self.f_opt:
                self.f_opt = np.min(fitness)
                self.x_opt = population[np.argmin(fitness)]

            # Check for diversity
            if (
                self.diversity(population) < self.diversity_threshold
                or no_improvement_counter >= self.restart_threshold
            ):
                population, velocities = self.initialize_population(bounds)
                personal_best_positions = np.copy(population)
                personal_best_scores = np.array([func(ind) for ind in population])
                global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
                global_best_score = np.min(personal_best_scores)
                no_improvement_counter = 0
                evaluations += self.initial_pop_size

            # CMA Update
            mean = np.mean(population, axis=0)
            cov_matrix = np.cov(population, rowvar=False)
            population = self.cma_update(population, mean, cov_matrix)

        return self.f_opt, self.x_opt
