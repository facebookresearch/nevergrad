import numpy as np


class EnhancedAdaptiveMultiStrategicOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Parameters and initial conditions
        population_size = 200
        mutation_rate = 0.8
        recombination_rate = 0.2
        sigma = 0.2  # Initial mutation step size
        elite_size = int(0.05 * population_size)  # Reduced elite proportion

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
            indices = np.arange(population_size)

            # Elitism
            elite_indices = np.argsort(fitness)[:elite_size]
            for idx in elite_indices:
                new_population.append(population[idx])

            # Main evolutionary process
            while len(new_population) < population_size:
                # Mutation and recombination strategy based on adaptive thresholds
                if np.random.rand() < mutation_rate:
                    # Mutation strategy
                    idx = np.random.choice(indices)
                    individual = population[idx]
                    mutant = individual + sigma * np.random.randn(self.dim)
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                    mutant_fitness = func(mutant)
                    evaluations += 1

                    # Acceptance of new mutant
                    if mutant_fitness < fitness[idx]:
                        new_population.append(mutant)
                        if mutant_fitness < best_fitness:
                            best_solution = mutant
                            best_fitness = mutant_fitness
                    else:
                        new_population.append(individual)
                else:
                    # Recombination
                    parents = np.random.choice(indices, 2, replace=False)
                    alpha = np.random.rand()
                    offspring = alpha * population[parents[0]] + (1 - alpha) * population[parents[1]]
                    offspring = np.clip(offspring, self.lower_bound, self.upper_bound)
                    offspring_fitness = func(offspring)
                    evaluations += 1

                    # Acceptance of new offspring
                    if offspring_fitness < fitness[parents[0]] and offspring_fitness < fitness[parents[1]]:
                        new_population.append(offspring)
                        if offspring_fitness < best_fitness:
                            best_solution = offspring
                            best_fitness = offspring_fitness
                    else:
                        new_population.append(population[parents[0]])

            population = np.array(new_population)
            fitness = np.array([func(x) for x in population])

            # Adaptive mutation rate and sigma adjustment
            mutation_rate = min(1.0, mutation_rate + np.random.uniform(-0.05, 0.05))  # Smoother adjustment
            sigma = max(
                0.001, sigma * np.exp(0.05 * (np.mean(fitness) - best_fitness) / best_fitness)
            )  # Smoother sigma adjustment

        return best_fitness, best_solution
