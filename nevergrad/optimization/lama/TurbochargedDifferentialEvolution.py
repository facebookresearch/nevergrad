import numpy as np


class TurbochargedDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimension fixed as per the problem
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Increased population size for better search space coverage
        population_size = 150
        mutation_factor = 0.5  # Initializing with a lower mutation factor
        crossover_prob = 0.7  # Higher crossover probability for increased diversity

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(x) for x in population])

        best_idx = np.argmin(fitness)
        best_value = fitness[best_idx]
        best_solution = population[best_idx].copy()

        # Adaptive mutation and crossover approach with direct feedback from performance
        performance_feedback = 0.1

        for _ in range(self.budget // population_size):
            new_population = np.empty_like(population)
            new_fitness = np.zeros(population_size)

            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Mutate: Dithering mutation strategy
                local_mutation = mutation_factor + performance_feedback * (np.random.rand() - 0.5)
                mutant = a + local_mutation * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover: Binomial
                cross_points = np.random.rand(self.dim) < crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = trial_fitness
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]

            population = new_population
            fitness = new_fitness

            # Update the best solution found
            current_best_idx = np.argmin(fitness)
            current_best_value = fitness[current_best_idx]
            if current_best_value < best_value:
                best_value = current_best_value
                best_solution = population[current_best_idx].copy()

            # Dynamic adaptation based on feedback
            if current_best_value < best_value:
                performance_feedback *= 1.05  # Increase mutation if performance is improving
            else:
                performance_feedback *= 0.95  # Decrease mutation if performance stagnates

        return best_value, best_solution
