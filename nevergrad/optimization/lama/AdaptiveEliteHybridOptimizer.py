import numpy as np


class AdaptiveEliteHybridOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.pop_size = 50
        self.min_pop_size = 10
        self.max_pop_size = 100
        self.initial_F = 0.5
        self.initial_CR = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.7
        self.elite_fraction = 0.1
        self.diversity_threshold = 0.1
        self.tau1 = 0.1
        self.tau2 = 0.1
        self.crossover_rate = 0.9

    def initialize_population(self, bounds):
        population = np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        return population, velocities

    def select_parents(self, population):
        idxs = np.random.choice(range(population.shape[0]), 3, replace=False)
        return population[idxs]

    def mutate(self, parent1, parent2, parent3, F):
        return np.clip(parent1 + F * (parent2 - parent3), -5.0, 5.0)

    def crossover(self, target, mutant, CR):
        j_rand = np.random.randint(self.dim)
        return np.array(
            [mutant[j] if np.random.rand() < CR or j == j_rand else target[j] for j in range(self.dim)]
        )

    def diversity(self, population):
        return np.mean(np.std(population, axis=0))

    def adapt_parameters(self, F, CR):
        if np.random.rand() < self.tau1:
            F = np.clip(np.random.normal(F, 0.1), 0, 1)
        if np.random.rand() < self.tau2:
            CR = np.clip(np.random.normal(CR, 0.1), 0, 1)
        return F, CR

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        bounds = func.bounds

        population, velocities = self.initialize_population(bounds)
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        evaluations = self.pop_size

        F = self.initial_F
        CR = self.initial_CR

        while evaluations < self.budget:
            current_pop_size = max(
                self.min_pop_size, int(self.pop_size * ((self.budget - evaluations) / self.budget))
            )
            new_population = np.zeros_like(population[:current_pop_size])
            fitness = np.zeros(current_pop_size)

            for i in range(current_pop_size):
                parent1, parent2, parent3 = self.select_parents(population)
                F, CR = self.adapt_parameters(F, CR)
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

            # Elitism
            elite_count = max(1, int(self.elite_fraction * current_pop_size))
            elite_indices = np.argsort(fitness)[:elite_count]
            elite_population = population[elite_indices]
            elite_velocities = velocities[elite_indices]

            # Check for diversity
            if self.diversity(population) < self.diversity_threshold:
                population, velocities = self.initialize_population(bounds)
                personal_best_positions = np.copy(population)
                personal_best_scores = np.array([func(ind) for ind in population])
                global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
                global_best_score = np.min(personal_best_scores)
                evaluations += self.pop_size
            else:
                population[:elite_count] = elite_population
                velocities[:elite_count] = elite_velocities

        return self.f_opt, self.x_opt
