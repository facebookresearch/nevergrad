import numpy as np


class QuantumDirectionalEnhancerV5:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Strategy parameters
        population_size = 150  # Further increased population size for more diversity
        gamma_initial = 0.6  # Slightly increased initial gamma for a balanced initial exploration
        gamma_final = 0.001  # Fine exploitation
        gamma_decay = np.exp(np.log(gamma_final / gamma_initial) / self.budget)  # Exponential decay
        elite_fraction = 0.05  # Even smaller elite fraction to encourage diversity
        mutation_strength = 0.05  # Reduced mutation strength
        mutation_decay = 0.992  # Gradual mutation decay
        crossover_probability = 0.95  # Maintain high crossover probability
        tunneling_frequency = 0.2  # Increased tunneling frequency for aggressive exploration
        directional_weight = 0.4  # Weight factor for tunneling direction towards best found solution

        # Initialize population uniformly within bounds
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations_left = self.budget - population_size

        while evaluations_left > 0:
            gamma = gamma_initial * (gamma_decay ** (self.budget - evaluations_left))
            mutation_strength *= mutation_decay  # Decaying mutation strength

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
                    crossover_point = np.random.randint(self.dim)
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
                    offspring += gamma * directional_weight * direction

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
