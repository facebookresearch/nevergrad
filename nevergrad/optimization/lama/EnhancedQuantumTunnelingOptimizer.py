import numpy as np


class EnhancedQuantumTunnelingOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is set as per the problem statement

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Initialize parameters
        population_size = 50  # Increased population size for better coverage
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        # Quantum parameters
        gamma = 0.1  # Initial quantum fluctuation magnitude
        gamma_decay = 0.985  # Slightly slower decay to maintain diversity longer

        # Evolutionary parameters
        crossover_rate = 0.9  # Increased crossover rate for better mixing
        mutation_strength = 0.3  # Reduced mutation strength to refine local search
        elite_count = 5  # Number of elites to preserve between generations

        evaluations_left = self.budget - population_size

        while evaluations_left > 0:
            # Update quantum fluctuation magnitude
            gamma *= gamma_decay

            # Select elite individuals
            elite_indices = np.argsort(fitness)[:elite_count]
            elite_individuals = population[elite_indices]

            new_population = list(elite_individuals)  # Start with elite individuals
            new_fitness = list(fitness[elite_indices])

            for _ in range(population_size - elite_count):
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
