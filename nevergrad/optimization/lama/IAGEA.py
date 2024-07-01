import numpy as np


class IAGEA:
    def __init__(self, budget, population_size=100, crossover_prob=0.9, mutation_factor=0.5, adaptive=True):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.crossover_prob = crossover_prob
        self.mutation_factor = mutation_factor
        self.adaptive = adaptive

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx]

        num_evals = self.population_size

        # Adaptive strategy parameters
        success_rate = 0.1
        learning_rate = 0.1

        while num_evals < self.budget:
            new_population = []
            num_successes = 0

            for i in range(self.population_size):
                # Mutation: DE/rand/1/bin with adaptive mutation factor
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = x1 + self.mutation_factor * (x2 - x3)
                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover
                cross_points = np.random.rand(self.dimension) < self.crossover_prob
                trial_vector = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial_vector)
                num_evals += 1

                if trial_fitness < fitness[i]:
                    new_population.append(trial_vector)
                    fitness[i] = trial_fitness
                    num_successes += 1

                    # Update the best found solution
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial_vector
                else:
                    new_population.append(population[i])

                if num_evals >= self.budget:
                    break

            population = np.array(new_population)

            # Adapt mutation factor based on success rate
            if self.adaptive:
                success_rate = num_successes / self.population_size
                if success_rate > 0.2:
                    self.mutation_factor *= 1 + learning_rate
                elif success_rate < 0.2:
                    self.mutation_factor *= 1 - learning_rate
                self.mutation_factor = np.clip(self.mutation_factor, 0.1, 1.0)

        return best_fitness, best_individual
