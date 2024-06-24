import numpy as np


class QuantumTunnelingOptimizerV17:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Tuning hyperparameters based on previous performance and feedback
        population_size = 200  # Further reduced population for increased individual iteration
        gamma = 0.2  # Increased initial tunneling coefficient for more pronounced explorative jumps
        gamma_min = 0.0001  # Lower minimum for finer late-stage exploration
        gamma_decay = 0.992  # Slower decay to preserve exploration capabilities longer
        elite_count = 10  # Reduced elite count to focus on top-performing individuals
        mutation_strength = 0.025  # Higher mutation strength to encourage diversity
        crossover_probability = 0.85  # Higher probability for effective recombination
        tunneling_frequency = 0.7  # Increased tunneling frequency for better local optima escaping

        # Initialize population within bounds
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations_left = self.budget - population_size

        while evaluations_left > 0:
            # Update gamma according to exponential decay
            gamma = max(gamma * gamma_decay, gamma_min)

            # Elitism: retain the top performers
            elite_indices = np.argsort(fitness)[:elite_count]
            elite_individuals = population[elite_indices]
            new_population = [population[i] for i in elite_indices]
            new_fitness = [fitness[i] for i in elite_indices]

            # Reproduction process: crossover and mutation
            while len(new_population) < population_size:
                parent1_idx, parent2_idx = np.random.choice(elite_indices, 2, replace=False)
                parent1, parent2 = population[parent1_idx], population[parent2_idx]

                # Crossover
                if np.random.random() < crossover_probability:
                    crossover_point = np.random.randint(1, self.dim)
                    offspring = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                else:
                    offspring = np.array(parent1)  # If crossover does not occur, clone parent1

                # Mutation
                mutation = np.random.normal(0, mutation_strength, self.dim)
                offspring += mutation
                offspring = np.clip(offspring, -5.0, 5.0)  # Ensure offspring remain within the problem bounds

                # Quantum tunneling effect
                if np.random.rand() < tunneling_frequency:
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
