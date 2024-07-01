import numpy as np


class AdaptiveSigmaCrossoverEvolution:
    def __init__(self, budget, dimension=5, population_size=50, sigma_init=1.0, crossover_prob=0.9):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.sigma = np.full(self.population_size, sigma_init)  # Initial sigma for each individual
        self.crossover_prob = crossover_prob  # Probability of crossover

    def __call__(self, func):
        # Initialize population within bounds [-5.0, 5.0]
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(x) for x in population])
        best_idx = np.argmin(fitness)
        f_opt = fitness[best_idx]
        x_opt = population[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Parent selection
                parent1_idx, parent2_idx = np.random.choice(self.population_size, 2, replace=False)
                parent1, parent2 = population[parent1_idx], population[parent2_idx]

                # Crossover
                if np.random.rand() < self.crossover_prob:
                    cross_points = np.random.rand(self.dimension) < 0.5
                    offspring = np.where(cross_points, parent1, parent2)
                else:
                    offspring = parent1.copy()

                # Mutation
                offspring += self.sigma[i] * np.random.randn(self.dimension)
                offspring = np.clip(offspring, -5.0, 5.0)  # Ensure offspring is within bounds

                # Evaluate offspring
                offspring_fitness = func(offspring)
                evaluations += 1

                # Selection
                if offspring_fitness < fitness[i]:
                    population[i] = offspring
                    fitness[i] = offspring_fitness
                    self.sigma[i] *= 0.95  # Reduce sigma if improvement
                else:
                    self.sigma[i] *= 1.05  # Increase sigma if no improvement

                # Update optimum if found a new best
                if offspring_fitness < f_opt:
                    f_opt = offspring_fitness
                    x_opt = offspring

                if evaluations >= self.budget:
                    break

        return f_opt, x_opt
