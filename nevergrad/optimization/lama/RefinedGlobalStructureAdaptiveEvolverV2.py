import numpy as np


class RefinedGlobalStructureAdaptiveEvolverV2:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = np.full(self.dim, -5.0)
        self.ub = np.full(self.dim, 5.0)

    def __call__(self, func):
        population_size = 300
        elite_size = 50
        evaluations = 0
        mutation_scale = 0.15
        adaptive_factor = 0.9
        recombination_prob = 0.65
        innovators_factor = 0.15  # Increase to improve exploration

        # Initialize population more strategically
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations += population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        while evaluations < self.budget:
            # Select elites based on a roulette wheel selection mechanism for diversity
            fitness_prob = 1 / (1 + fitness - fitness.min())
            fitness_prob /= fitness_prob.sum()
            elite_indices = np.random.choice(population_size, elite_size, replace=False, p=fitness_prob)
            elite_individuals = population[elite_indices]
            elite_fitness = fitness[elite_indices]

            # Generate new candidates
            new_population = []
            new_fitness = []
            for _ in range(population_size - elite_size):
                if np.random.rand() < recombination_prob:
                    indices = np.random.choice(elite_size, 3, replace=False)
                    x0, x1, x2 = elite_individuals[indices]
                    child = x0 + mutation_scale * (x1 - x2)
                    child = np.clip(child, self.lb, self.ub)
                else:
                    idx = np.random.choice(elite_size)
                    child = elite_individuals[idx] + np.random.normal(0, mutation_scale, self.dim)
                    child = np.clip(child, self.lb, self.ub)

                child_fitness = func(child)
                evaluations += 1

                if child_fitness < self.f_opt:
                    self.f_opt = child_fitness
                    self.x_opt = child

                new_population.append(child)
                new_fitness.append(child_fitness)

            # Add innovators to explore more of the search space
            innovators = np.random.uniform(
                self.lb, self.ub, (int(population_size * innovators_factor), self.dim)
            )
            innovator_fitness = np.array([func(ind) for ind in innovators])
            evaluations += len(innovators)

            # Form the new population from elite, new candidates, and innovators
            population = np.vstack((elite_individuals, new_population, innovators))
            fitness = np.hstack((elite_fitness, new_fitness, innovator_fitness))

            # Adaptive mutation scale adjustment
            mutation_scale *= adaptive_factor
            if mutation_scale < 0.05:
                mutation_scale = 0.15  # Reset mutation scale

            # Retain best solutions based on fitness
            best_indices = np.argsort(fitness)[:population_size]
            population = population[best_indices]
            fitness = fitness[best_indices]

        return self.f_opt, self.x_opt
