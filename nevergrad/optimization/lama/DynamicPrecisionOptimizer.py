import numpy as np


class DynamicPrecisionOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Defined dimensionality of the problem
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initial strategic setup
        current_budget = 0
        population_size = 100
        num_elites = 10
        mutation_rate = 0.8
        crossing_probability = 0.7

        # Initialize population randomly
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        current_budget += population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        # Evolutionary loop
        while current_budget < self.budget:
            new_population = np.empty_like(population)
            new_fitness = np.empty_like(fitness)

            # Elite preservation strategy
            elite_indices = np.argsort(fitness)[:num_elites]
            new_population[:num_elites] = population[elite_indices]
            new_fitness[:num_elites] = fitness[elite_indices]

            # Generate new solutions via mutation and crossover
            for i in range(num_elites, population_size):
                if current_budget >= self.budget:
                    break

                # Selection of parents for breeding based on fitness
                parents_indices = np.random.choice(population_size, 3, replace=False)
                x1, x2, x3 = population[parents_indices]

                # Mutation: differential evolution mutation strategy
                mutant = x1 + mutation_rate * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                cross_points = np.random.rand(self.dim) < crossing_probability
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate trial solution
                trial_fitness = func(trial)
                current_budget += 1

                # Greedy selection
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = trial_fitness
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]

                # Update best solution found
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness

            # Update population and fitness
            population = new_population
            fitness = new_fitness

            # Dynamically adjust mutation rate and crossing probability based on progress
            progress = current_budget / self.budget
            mutation_rate = max(0.5, 1 - progress)  # Decrease mutation rate over time
            crossing_probability = min(1.0, 0.5 + progress * 0.5)  # Increase crossover probability

        return best_fitness, best_solution
