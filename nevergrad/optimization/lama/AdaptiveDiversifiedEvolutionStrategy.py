import numpy as np


class AdaptiveDiversifiedEvolutionStrategy:
    def __init__(self, budget, dimension=5, lower_bound=-5.0, upper_bound=5.0):
        self.budget = budget
        self.dimension = dimension
        self.bounds = {"lb": lower_bound, "ub": upper_bound}
        self.population_size = 20
        self.mutation_rate = 1.0 / dimension
        self.mutation_scale = 0.1
        self.crossover_probability = 0.7

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(
            self.bounds["lb"], self.bounds["ub"], (self.population_size, self.dimension)
        )
        fitness = np.array([func(individual) for individual in population])
        f_opt = np.min(fitness)
        x_opt = population[np.argmin(fitness)]

        # Evolutionary loop
        iterations = self.budget // self.population_size
        for _ in range(iterations):
            new_population = []

            for idx in range(self.population_size):
                # Mutation
                if np.random.rand() < self.mutation_rate:
                    mutant = population[idx] + np.random.normal(0, self.mutation_scale, self.dimension)
                    mutant = np.clip(mutant, self.bounds["lb"], self.bounds["ub"])
                else:
                    mutant = population[idx]

                # Crossover
                if np.random.rand() < self.crossover_probability:
                    partner_idx = np.random.randint(self.population_size)
                    crossover_point = np.random.randint(self.dimension)
                    offspring = np.concatenate(
                        (population[idx][:crossover_point], population[partner_idx][crossover_point:])
                    )
                else:
                    offspring = mutant

                # Selection
                offspring_fitness = func(offspring)
                if offspring_fitness < fitness[idx]:
                    new_population.append(offspring)
                    fitness[idx] = offspring_fitness

                    # Update the best solution found
                    if offspring_fitness < f_opt:
                        f_opt = offspring_fitness
                        x_opt = offspring
                else:
                    new_population.append(population[idx])

            population = np.array(new_population)

        return f_opt, x_opt
