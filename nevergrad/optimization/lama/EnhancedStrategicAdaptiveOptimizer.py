import numpy as np


class EnhancedStrategicAdaptiveOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Parameters and initial conditions optimized
        population_size = 200
        mutation_factor = 0.8
        crossover_rate = 0.7
        sigma = 0.2  # Mutation step size is reduced to control excessive exploration
        elite_size = int(0.05 * population_size)  # Reduced elite size to encourage diversity

        # Initial population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(x) for x in population])
        evaluations = population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        # Evolutionary loop
        while evaluations < self.budget:
            new_population = []

            # Elitism
            elite_indices = np.argsort(fitness)[:elite_size]
            for idx in elite_indices:
                new_population.append(population[idx])

            # Main evolutionary process
            while len(new_population) < population_size:
                # Differential Mutation and Crossover
                for _ in range(int(population_size / 2)):
                    target_idx = np.random.randint(0, population_size)
                    candidates = np.random.choice(
                        np.delete(np.arange(population_size), target_idx), 3, replace=False
                    )
                    x1, x2, x3 = (
                        population[candidates[0]],
                        population[candidates[1]],
                        population[candidates[2]],
                    )

                    mutant = x1 + mutation_factor * (x2 - x3)
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                    # Crossover
                    cross_points = np.random.rand(self.dim) < crossover_rate
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, population[target_idx])

                    trial_fitness = func(trial)
                    evaluations += 1

                    # Selection
                    if trial_fitness < fitness[target_idx]:
                        new_population.append(trial)
                        if trial_fitness < best_fitness:
                            best_solution = trial
                            best_fitness = trial_fitness
                    else:
                        new_population.append(population[target_idx])

            population = np.array(new_population)
            fitness = np.array([func(x) for x in population])

            # Adaptive mutation and crossover rates
            mutation_factor = max(0.5, min(1.0, mutation_factor + np.random.uniform(-0.05, 0.05)))
            crossover_rate = max(0.5, min(1.0, crossover_rate + np.random.uniform(-0.05, 0.05)))
            sigma *= np.exp(0.05 * np.random.randn())

        return best_fitness, best_solution
