import numpy as np


class QuantumTunnelingOptimizerV4:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality given in the problem statement

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Hyperparameters
        population_size = 200  # Increased population size for enhanced diversity
        gamma = 0.2  # Increased initial quantum fluctuation magnitude
        gamma_min = 0.01  # Slightly increased minimum quantum fluctuation
        gamma_decay = 0.95  # Reduced decay rate for longer exploration maintenance
        elite_count = 20  # Higher number of elites for better performance retention
        mutation_strength = 0.1  # Lower mutation strength for more precise adjustments
        crossover_probability = 0.85  # Slightly adjusted crossover probability

        # Initialize population
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations_left = self.budget - population_size

        while evaluations_left > 0:
            # Update quantum fluctuation magnitude
            gamma = max(gamma * gamma_decay, gamma_min)

            # Elitism: carry forward best individuals
            elite_indices = np.argsort(fitness)[:elite_count]
            elite_individuals = population[elite_indices]
            new_population = [population[i] for i in elite_indices]
            new_fitness = [fitness[i] for i in elite_indices]

            # Generate new individuals from existing population
            while len(new_population) < population_size:
                parent1_idx, parent2_idx = np.random.choice(population_size, 2, replace=False)
                parent1, parent2 = population[parent1_idx], population[parent2_idx]

                if np.random.random() < crossover_probability:
                    crossover_point = np.random.randint(1, self.dim)
                    offspring = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                else:
                    offspring = parent1 if np.random.random() < 0.5 else parent2

                # Mutation
                mutation = np.random.normal(0, mutation_strength, self.dim)
                offspring += mutation
                offspring = np.clip(offspring, -5.0, 5.0)

                # Quantum tunneling effect
                if np.random.rand() < gamma:
                    tunnel_point = np.random.uniform(-5.0, 5.0, self.dim)
                    offspring = gamma * offspring + (1 - gamma) * tunnel_point

                offspring_fitness = func(offspring)
                evaluations_left -= 1

                if evaluations_left <= 0:
                    break

                new_population.append(offspring)
                new_fitness.append(offspring_fitness)

                if offspring_fitness < self.f_opt:
                    self.f_opt = offspring_fitness
                    self.x_opt = offspring

            population = np.array(new_population)
            fitness = np.array(new_fitness)

        return self.f_opt, self.x_opt
