import numpy as np


class AdaptiveDifferentialEvolutionPlus:
    def __init__(self, budget=10000, population_size=20):
        self.budget = budget
        self.population_size = population_size
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        eval_count = self.population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        # Add adaptiveness to mutation factor and crossover probability
        adaptive_rate = 0.1  # rate of adaptation
        while eval_count < self.budget:
            for i in range(self.population_size):
                # Mutation
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.bounds[0], self.bounds[1])

                # Crossover
                trial = np.copy(population[i])
                crossover_mask = np.random.rand(self.dim) < self.crossover_probability
                trial[crossover_mask] = mutant[crossover_mask]

                # Selection
                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update global best
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial
                        # Adapt mutation factor and crossover probability
                        self.mutation_factor = min(1.0, self.mutation_factor + adaptive_rate)
                        self.crossover_probability = min(1.0, self.crossover_probability + adaptive_rate)
                else:
                    self.mutation_factor = max(0.1, self.mutation_factor - adaptive_rate)
                    self.crossover_probability = max(0.1, self.crossover_probability - adaptive_rate)

                if eval_count >= self.budget:
                    break

        return self.f_opt, self.x_opt
