import numpy as np


class DifferentialEvolutionPSOHybrid:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.pop_size = 50
        self.F = 0.5  # Mutation factor
        self.CR = 0.9  # Crossover rate
        self.c1 = 1.5  # Cognitive parameter
        self.c2 = 1.5  # Social parameter
        self.w = 0.5  # Inertia weight

    def initialize_population(self, bounds):
        population = np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        return population, velocities

    def select_parents(self, population):
        idxs = np.random.choice(range(self.pop_size), 3, replace=False)
        return population[idxs]

    def mutate(self, parent1, parent2, parent3):
        return parent1 + self.F * (parent2 - parent3)

    def crossover(self, target, mutant):
        j_rand = np.random.randint(self.dim)
        trial = np.array(
            [mutant[j] if np.random.rand() < self.CR or j == j_rand else target[j] for j in range(self.dim)]
        )
        return trial

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

        while evaluations < self.budget:
            new_population = np.zeros_like(population)
            fitness = np.zeros(self.pop_size)

            for i in range(self.pop_size):
                parent1, parent2, parent3 = self.select_parents(population)
                mutant = self.mutate(parent1, parent2, parent3)
                trial = self.crossover(population[i], mutant)

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
            if np.min(fitness) < self.f_opt:
                self.f_opt = np.min(fitness)
                self.x_opt = population[np.argmin(fitness)]

        return self.f_opt, self.x_opt
