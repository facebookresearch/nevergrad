import numpy as np


class QuantumDirectionalEnhancerV16:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # The dimensionality is fixed at 5

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Refinement of strategy parameters for improved performance
        population_size = 450  # Further reduction for more focused search
        gamma_initial = 1.1  # Slightly increased initial gamma for more aggressive early exploration
        gamma_final = 0.00005  # Lower final gamma for finer exploitation
        gamma_decay = np.exp(
            np.log(gamma_final / gamma_initial) / (self.budget * 0.95)
        )  # Extended decay period
        elite_fraction = 0.25  # Further refined elite fraction to balance exploration-exploitation
        mutation_strength = 0.0002  # Adjusted mutation strength for more effective mutations
        mutation_decay = 0.9997  # Adjusted decay to maintain effectiveness for longer
        crossover_probability = 0.85  # Slight reduction to favor elite propagation
        tunneling_frequency = 0.85  # Adjusted to achieve better balance between updates and quantum leaps
        directional_weight = 1.05  # Slightly increased to enhance aggressive pursuit of best regions
        feedback_rate = 0.08  # Reduced to enhance stability in feedback mechanism

        # Initialize population and evaluate fitness
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations_left = self.budget - population_size

        while evaluations_left > 0:
            gamma = gamma_initial * (gamma_decay ** (self.budget - evaluations_left))
            mutation_strength *= mutation_decay  # Continue to adjust mutation strength

            # Select elite individuals
            elite_count = int(population_size * elite_fraction)
            elite_indices = np.argsort(fitness)[:elite_count]
            elite_individuals = population[elite_indices]
            new_population = list(elite_individuals)
            new_fitness = list(fitness[elite_indices])

            # Reproduce through crossover and mutation
            while len(new_population) < population_size:
                if evaluations_left <= 0:
                    break

                # Parent selection
                parents = np.random.choice(elite_count, 2, replace=False)
                parent1, parent2 = new_population[parents[0]], new_population[parents[1]]

                # Crossover
                if np.random.random() < crossover_probability:
                    offspring = np.concatenate((parent1[: self.dim // 2], parent2[self.dim // 2 :]))
                else:
                    offspring = np.array(parent1)

                # Mutation
                mutation = np.random.normal(0, mutation_strength, self.dim)
                offspring += mutation
                offspring = np.clip(offspring, -5.0, 5.0)

                # Quantum tunneling
                if np.random.rand() < tunneling_frequency:
                    best_idx = np.argmin(new_fitness)
                    best_individual = new_population[best_idx]
                    direction = best_individual - offspring
                    feedback = np.mean([f - self.f_opt for f in new_fitness]) * feedback_rate
                    offspring += (gamma * directional_weight + feedback) * direction

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
