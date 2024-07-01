import numpy as np


class AdaptiveQuorumWithStrategicMutation:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=100,
        elite_fraction=0.1,
        initial_mutation_scale=0.5,
        quorum_size=5,
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(population_size * elite_fraction)
        self.initial_mutation_scale = initial_mutation_scale
        self.quorum_size = quorum_size

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(individual) for individual in population])
        evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        mutation_scale = self.initial_mutation_scale

        while evaluations < self.budget:
            new_population = np.empty_like(population)
            for i in range(self.population_size):
                # Select quorum of individuals, find their elite
                quorum_indices = np.random.choice(self.population_size, self.quorum_size, replace=False)
                elite_idx = quorum_indices[np.argmin(fitness[quorum_indices])]
                elite = population[elite_idx]

                # Strategic mutation based on best and local elites
                direction = best_individual - elite
                mutation = np.random.normal(0, 1, self.dimension) * mutation_scale + direction * 0.1
                child = np.clip(elite + mutation, -5.0, 5.0)
                child_fitness = func(child)
                evaluations += 1

                # Update best if necessary
                if child_fitness < best_fitness:
                    best_fitness = child_fitness
                    best_individual = child

                new_population[i] = child

                if evaluations >= self.budget:
                    break

            # Update population and fitness
            population = new_population
            fitness = np.array([func(ind) for ind in population])

            # Adapt mutation scale
            mutation_scale = max(0.01, mutation_scale * 0.99)  # Decay mutation scale

        return best_fitness, best_individual
