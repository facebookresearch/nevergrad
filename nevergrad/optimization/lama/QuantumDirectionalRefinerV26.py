import numpy as np


class QuantumDirectionalRefinerV26:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Adjustments based on the feedback from V25 and aiming to improve performance
        population_size = 700  # Further decreased population size for increased focus
        gamma_initial = 0.2  # Start with a lower initial gamma for gentle initial exploration
        gamma_final = 0.00005  # Lower final gamma for more precise final adjustments
        gamma_decay = np.exp(
            np.log(gamma_final / gamma_initial) / (self.budget * 0.8)
        )  # Adjusting decay rate
        elite_fraction = 0.15  # Further reduced elite fraction to promote diversity
        mutation_strength = 0.002  # Slightly reduced mutation strength
        mutation_decay = 0.9999  # Slower mutation decay for consistent exploration
        crossover_probability = 0.65  # Further reduced for more heritage preservation
        tunneling_frequency = 0.95  # Increased tunneling to aggressively escape local optima
        directional_weight = 12.0  # Increase weight for stronger elite attraction
        feedback_rate = 0.7  # Increase feedback sensitivity

        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations_left = self.budget - population_size

        while evaluations_left > 0:
            gamma = gamma_initial * (gamma_decay ** (self.budget - evaluations_left))
            mutation_strength *= mutation_decay

            elite_count = int(population_size * elite_fraction)
            elite_indices = np.argsort(fitness)[:elite_count]
            elite_individuals = population[elite_indices]
            new_population = list(elite_individuals)
            new_fitness = list(fitness[elite_indices])

            while len(new_population) < population_size:
                if evaluations_left <= 0:
                    break

                parents = np.random.choice(elite_count, 2, replace=False)
                parent1, parent2 = elite_individuals[parents[0]], elite_individuals[parents[1]]

                if np.random.random() < crossover_probability:
                    offspring = np.where(np.random.rand(self.dim) < 0.5, parent1, parent2)
                else:
                    offspring = np.array(parent1)

                mutation = np.random.normal(0, mutation_strength, self.dim)
                offspring += mutation
                offspring = np.clip(offspring, -5.0, 5.0)

                if np.random.rand() < tunneling_frequency:
                    best_idx = np.argmin(new_fitness)
                    best_individual = new_population[best_idx]
                    direction = best_individual - offspring
                    feedback = np.mean([f - self.f_opt for f in new_fitness]) * feedback_rate
                    offspring += gamma * directional_weight * direction + feedback

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
