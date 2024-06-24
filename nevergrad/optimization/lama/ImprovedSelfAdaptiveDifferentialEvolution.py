import numpy as np


class ImprovedSelfAdaptiveDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # fixed dimensionality

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        bounds = np.array([-5.0, 5.0])
        population_size = 20
        base_F = 0.8  # Differential weight
        base_CR = 0.9  # Crossover probability

        # Initialize population
        population = np.random.uniform(bounds[0], bounds[1], (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        evaluations = population_size

        # Initialize self-adaptive parameters
        F_values = np.full(population_size, base_F)
        CR_values = np.full(population_size, base_CR)

        # Adaptive parameters for diversity control
        stagnation_limit = 50
        no_improvement_counter = 0
        diversity_threshold = 1e-5

        while evaluations < self.budget:
            new_population = np.zeros_like(population)
            new_fitness = np.zeros(population_size)
            new_F_values = np.zeros(population_size)
            new_CR_values = np.zeros(population_size)

            for i in range(population_size):
                # Adaptation of F and CR
                if np.random.rand() < 0.1:
                    F_values[i] = 0.1 + 0.9 * np.random.rand()
                if np.random.rand() < 0.1:
                    CR_values[i] = np.random.rand()

                # Mutation
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F_values[i] * (b - c), bounds[0], bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) < CR_values[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                evaluations += 1

                if f_trial < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial
                    new_F_values[i] = F_values[i]
                    new_CR_values[i] = CR_values[i]

                    # Update the best found solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        no_improvement_counter = 0
                    else:
                        no_improvement_counter += 1
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]
                    new_F_values[i] = F_values[i]
                    new_CR_values[i] = CR_values[i]

                # Handle stagnation by enhancing exploration
                if no_improvement_counter >= stagnation_limit:
                    population_variance = np.var(population, axis=0)
                    if np.all(population_variance < diversity_threshold):
                        new_population = np.random.uniform(bounds[0], bounds[1], (population_size, self.dim))
                        new_fitness = np.array([func(ind) for ind in new_population])
                        evaluations += population_size
                        no_improvement_counter = 0

                if evaluations >= self.budget:
                    break

            population, fitness = new_population, new_fitness
            F_values, CR_values = new_F_values, new_CR_values

        return self.f_opt, self.x_opt
