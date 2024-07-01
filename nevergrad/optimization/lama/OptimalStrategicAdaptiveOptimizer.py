import numpy as np


class OptimalStrategicAdaptiveOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Parameters and initial conditions
        population_size = 150
        mutation_rate = 0.7
        crossover_rate = 0.6
        sigma = 0.3  # Initial mutation step size
        elite_size = int(0.1 * population_size)  # Elite proportion

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
                # Crossover
                if np.random.rand() < crossover_rate:
                    parents = np.random.choice(population_size, 2, replace=False)
                    crossover_point = np.random.randint(1, self.dim)
                    offspring = np.concatenate(
                        (population[parents[0]][:crossover_point], population[parents[1]][crossover_point:])
                    )
                    offspring = np.clip(offspring, self.lower_bound, self.upper_bound)
                    offspring_fitness = func(offspring)
                    evaluations += 1

                    # Selection
                    if offspring_fitness < fitness[parents[1]]:
                        new_population.append(offspring)
                        if offspring_fitness < best_fitness:
                            best_solution = offspring
                            best_fitness = offspring_fitness
                    else:
                        new_population.append(population[parents[1]])

                # Mutation
                idx = np.random.choice(population_size)
                individual = population[idx]
                mutant = individual + sigma * np.random.randn(self.dim)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                mutant_fitness = func(mutant)
                evaluations += 1

                # Selection
                if mutant_fitness < fitness[idx]:
                    new_population.append(mutant)
                    if mutant_fitness < best_fitness:
                        best_solution = mutant
                        best_fitness = mutant_fitness
                else:
                    new_population.append(individual)

            population = np.array(new_population)
            fitness = np.array([func(x) for x in population])

            # Adaptive strategy
            mutation_rate = min(1.0, mutation_rate + np.random.uniform(-0.05, 0.05))
            crossover_rate = min(1.0, crossover_rate + np.random.uniform(-0.05, 0.05))
            sigma *= np.exp(0.05 * np.random.randn())

        return best_fitness, best_solution
