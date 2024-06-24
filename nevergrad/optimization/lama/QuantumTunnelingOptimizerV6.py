import numpy as np


class QuantumTunnelingOptimizerV6:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Hyperparameters adjustment
        population_size = 300  # Further increased population for more diversity
        gamma = 0.3  # Starting quantum fluctuation magnitude
        gamma_min = 0.001  # Lower minimum quantum fluctuation
        gamma_decay = 0.95  # Slower decay to maintain exploration longer
        elite_count = 30  # Increased elite count for better elite retention
        mutation_strength = 0.02  # Reduced mutation strength for finer adjustments
        crossover_probability = 0.92  # Increased crossover probability for more mixing

        # Initialize population uniformly within bounds
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations_left = self.budget - population_size

        while evaluations_left > 0:
            # Update gamma according to decay schedule
            gamma = max(gamma * gamma_decay, gamma_min)

            # Elitism: keep the best performing individuals
            elite_indices = np.argsort(fitness)[:elite_count]
            elite_individuals = population[elite_indices]
            new_population = [population[i] for i in elite_indices]
            new_fitness = [fitness[i] for i in elite_indices]

            # Breed new individuals via crossover and mutation
            while len(new_population) < population_size:
                parent1_idx, parent2_idx = np.random.choice(elite_indices, 2, replace=False)
                parent1, parent2 = population[parent1_idx], population[parent2_idx]

                # Crossover
                if np.random.random() < crossover_probability:
                    crossover_point = np.random.randint(1, self.dim)
                    offspring = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                else:
                    offspring = np.array(parent1)  # Clone one parent if no crossover

                # Mutation
                mutation = np.random.normal(0, mutation_strength, self.dim)
                offspring += mutation
                offspring = np.clip(offspring, -5.0, 5.0)  # Ensure offspring remain within bounds

                # Introduce quantum tunneling effect with updated dynamics
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
