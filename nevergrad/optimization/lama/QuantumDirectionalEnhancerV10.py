import numpy as np


class QuantumDirectionalEnhancerV10:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Strategy parameters
        population_size = 500  # Adjusted population size for better control
        gamma_initial = 1.0  # Lower initial gamma for focused exploration
        gamma_final = 0.0001  # Finer final gamma for improved exploitation
        gamma_decay = np.exp(
            np.log(gamma_final / gamma_initial) / (self.budget * 0.5)
        )  # Faster initial decay
        elite_fraction = 0.1  # Increased elite fraction for better elitism
        mutation_strength = 0.005  # Reduced mutation strength for finer granularity
        mutation_decay = 0.99  # Adjusted mutation decay
        crossover_probability = 0.8  # Increased crossover probability
        tunneling_frequency = 0.6  # Increased tunneling frequency
        directional_weight = 0.95  # Increased weight for tunneling direction

        # Initialize population uniformly within bounds
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations_left = self.budget - population_size

        while evaluations_left > 0:
            gamma = gamma_initial * (gamma_decay ** (self.budget - evaluations_left))
            mutation_strength *= mutation_decay  # Adjust mutation strength dynamically

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
