import numpy as np


class SAGEA:
    def __init__(self, budget, population_size=150, crossover_prob=0.8, mutation_factor=0.6):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.crossover_prob = crossover_prob
        self.mutation_factor = mutation_factor

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx]

        num_evals = self.population_size

        # Evolutionary loop
        while num_evals < self.budget:
            new_population = []
            for i in range(self.population_size):
                # Mutation: DE/rand/1/bin with scaled mutation factor
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                scale = 1.0 - (num_evals / self.budget)  # Decrease scale over time
                mutant = x1 + self.mutation_factor * scale * (x2 - x3)
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

                    # Update the best found solution
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial_vector
                else:
                    new_population.append(population[i])

                if num_evals >= self.budget:
                    break

            population = np.array(new_population)

        return best_fitness, best_individual
