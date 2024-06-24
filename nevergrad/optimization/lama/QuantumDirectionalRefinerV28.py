import numpy as np


class QuantumDirectionalRefinerV28:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        # Revising parameters based on the feedback from V27 results
        population_size = 300  # Smaller population size for more refined searching
        gamma_initial = 0.1  # Lower start for gamma
        gamma_final = 0.00001  # Maintained aggressive final gamma
        gamma_decay = np.exp(np.log(gamma_final / gamma_initial) / (self.budget * 0.9))  # Slower decay
        elite_fraction = 0.05  # Reduced elite fraction to focus on the very best
        mutation_strength = 0.002  # Reduced mutation strength
        mutation_decay = 0.99997  # Slightly slower decay
        crossover_probability = 0.75  # Increased crossover probability for genetic diversity
        tunneling_frequency = 0.95  # Reduced frequency to focus more on direct improvements
        directional_weight = 10.0  # Reduced directional influence
        feedback_rate = 0.85  # Increased feedback sensitivity

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
