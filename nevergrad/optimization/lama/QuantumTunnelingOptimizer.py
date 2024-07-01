import numpy as np


class QuantumTunnelingOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is set as per the problem statement

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Initialize parameters
        population_size = 20  # Modest population size for balance
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        # Quantum parameters
        gamma = 0.1  # Initial quantum fluctuation magnitude
        gamma_decay = 0.99  # Decay rate for gamma

        # Evolutionary parameters
        crossover_rate = 0.85
        mutation_strength = 0.5

        evaluations_left = self.budget - population_size

        while evaluations_left > 0:
            # Update quantum fluctuation magnitude
            gamma *= gamma_decay

            new_population = []
            new_fitness = []

            for i in range(population_size):
                # Selection of parents for crossover
                parents_indices = np.random.choice(population_size, 2, replace=False)
                parent1, parent2 = population[parents_indices]

                # Crossover
                mask = np.random.rand(self.dim) < crossover_rate
                offspring = np.where(mask, parent1, parent2)

                # Mutation
                mutation = np.random.normal(0, mutation_strength, self.dim)
                offspring += mutation
                offspring = np.clip(offspring, -5.0, 5.0)

                # Quantum tunneling effect
                if np.random.rand() < gamma:
                    tunnel_point = np.random.uniform(-5.0, 5.0, self.dim)
                    offspring = gamma * offspring + (1 - gamma) * tunnel_point

                # Evaluate offspring
                f_offspring = func(offspring)
                evaluations_left -= 1
                if evaluations_left <= 0:
                    break

                # Insert offspring into new population
                new_population.append(offspring)
                new_fitness.append(f_offspring)

                if f_offspring < self.f_opt:
                    self.f_opt = f_offspring
                    self.x_opt = offspring

            # Update the population
            population = np.array(new_population)
            fitness = np.array(new_fitness)

        return self.f_opt, self.x_opt
