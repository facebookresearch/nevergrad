import numpy as np


class QuantumDirectionalEnhancerV12:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Strategy parameters
        population_size = 450  # Slightly increased population for broader search initially
        gamma_initial = 0.9  # Slightly increased initial gamma for stronger early exploration
        gamma_final = 0.00001  # Further reduced final gamma for refined exploitation
        gamma_decay = np.exp(
            np.log(gamma_final / gamma_initial) / (self.budget * 0.65)
        )  # Fine-tuned decay rate
        elite_fraction = 0.25  # Increased elite fraction to maintain a stronger genetic pool
        mutation_strength = 0.0008  # Further reduced mutation strength for more precise adjustments
        mutation_decay = 0.998  # Slower mutation decay for extended effectiveness
        crossover_probability = 0.88  # Slightly increased crossover probability
        tunneling_frequency = 0.8  # Increased tunneling frequency for enhanced quantum behavior
        directional_weight = 0.95  # Adjusted weight for tunneling direction
        feedback_rate = 0.12  # Increased feedback rate for more aggressive adaptation

        # Initialize population uniformly within bounds
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations_left = self.budget - population_size

        while evaluations_left > 0:
            gamma = gamma_initial * (gamma_decay ** (self.budget - evaluations_left))
            mutation_strength *= mutation_decay  # Dynamically adjust mutation strength

            # Elite selection
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
                    crossover_point = np.random.randint(1, self.dim)
                    offspring = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
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
