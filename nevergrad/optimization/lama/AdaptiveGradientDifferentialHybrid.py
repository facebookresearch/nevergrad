import numpy as np


class AdaptiveGradientDifferentialHybrid:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        population_size = 100
        elite_size = 5
        mutation_factor = 0.85
        crossover_rate = 0.8
        adaptive_factor = 0.1

        # Initialize population and evaluate fitness
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        evaluations = population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        while evaluations < self.budget:
            new_population = []
            for i in range(population_size):
                # Differential mutation with adaptive factor
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + mutation_factor * (b - c), self.lb, self.ub)
                mutant = np.clip(mutant + adaptive_factor * (self.x_opt - mutant), self.lb, self.ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < crossover_rate
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate trial solution
                trial_fitness = func(trial)
                evaluations += 1

                # Adaptive selection
                if trial_fitness < fitness[i]:
                    new_population.append(trial)
                    fitness[i] = trial_fitness

                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial
                else:
                    new_population.append(population[i])

            population = np.array(new_population)

            # Adaptive mutation factor update
            mutation_factor = np.clip(mutation_factor - 0.01 * (1 - np.mean(fitness) / self.f_opt), 0.5, 1)

            # Elitism
            elite_indices = np.argsort(fitness)[:elite_size]
            for elite in elite_indices:
                population[np.random.randint(population_size)] = population[elite]

        return self.f_opt, self.x_opt
