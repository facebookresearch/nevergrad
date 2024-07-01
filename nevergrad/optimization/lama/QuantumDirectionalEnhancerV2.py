import numpy as np


class QuantumDirectionalEnhancerV2:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Strategy parameters
        population_size = 50  # Further increased population size
        gamma_initial = 0.9  # Initial gamma for broad initial exploration
        gamma_final = 0.001  # Final gamma for fine-tuned exploitation
        gamma_decay = (gamma_final / gamma_initial) ** (1 / self.budget)  # Exponential decay
        elite_fraction = 0.1  # Reduced elite fraction for greater diversity
        mutation_strength = 0.1  # Increased mutation strength
        mutation_decay = 0.995  # Introduce mutation decay
        crossover_probability = 0.85  # Increased crossover probability
        tunneling_frequency = 0.05  # Reduced tunneling frequency for focused jumps

        # Initialize population uniformly within bounds
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations_left = self.budget - population_size

        while evaluations_left > 0:
            gamma = gamma_initial * (gamma_decay ** (self.budget - evaluations_left))
            mutation_strength *= mutation_decay  # Decaying mutation strength

            # Elite selection based on elite fraction
            elite_count = int(population_size * elite_fraction)
            elite_indices = np.argsort(fitness)[:elite_count]
            elite_individuals = population[elite_indices]
            new_population = list(elite_individuals)
            new_fitness = list(fitness[elite_indices])

            # Reproduction: crossover and mutation
            while len(new_population) < population_size:
                if evaluations_left <= 0:
                    break

                idxs = np.random.choice(elite_count, 2, replace=False)
                parent1, parent2 = new_population[idxs[0]], new_population[idxs[1]]

                # Crossover
                if np.random.random() < crossover_probability:
                    offspring = np.where(np.random.rand(self.dim) < 0.5, parent1, parent2)
                else:
                    offspring = np.array(parent1)

                # Mutation
                mutation = np.random.normal(0, mutation_strength, self.dim)
                offspring += mutation
                offspring = np.clip(offspring, -5.0, 5.0)

                # Quantum tunneling with directional bias towards the best individual in the population
                if np.random.rand() < tunneling_frequency:
                    best_idx = np.argmin(new_fitness)
                    best_individual = new_population[best_idx]
                    direction = best_individual - offspring
                    offspring += gamma * direction

                offspring_fitness = func(offspring)
                evaluations_left -= 1

                new_population.append(offspring)
                new_fitness.append(offspring_fitness)

                if offspring_fitness < self.f_opt:
                    self.f_opt = offspring_fitness
                    self.x_opt = offspring

            population = np.array(new_population)
            fitness = np.array(new_fitness)

        return self.f_opt, self.x_opt
