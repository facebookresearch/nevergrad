import numpy as np


class QuantumDirectionalEnhancerV15:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Adjusted strategy parameters for improved performance
        population_size = 500  # Reduced population for better convergence
        gamma_initial = 1.0  # Higher initial gamma for more aggressive early exploration
        gamma_final = 0.0001  # Very low final gamma for precise exploitation
        gamma_decay = np.exp(
            np.log(gamma_final / gamma_initial) / (self.budget * 0.9)
        )  # Adjusted for longer decay
        elite_fraction = 0.3  # Slightly reduced elite fraction to balance exploration-exploitation
        mutation_strength = 0.0001  # Lower mutation strength for more precise updates
        mutation_decay = 0.9996  # Slower decay to maintain effectiveness
        crossover_probability = 0.88  # Reduced to favor elite propagation
        tunneling_frequency = 0.88  # Adjusted frequency to balance regular updates and quantum leaps
        directional_weight = 1.0  # Full weight on direction for aggressive pursuit
        feedback_rate = 0.1  # Slight decrease to stabilize feedback mechanism

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
