import numpy as np


class QuantumDirectionalRefinerV27:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Further refinements based on the performance of V26
        population_size = 500  # Further decreased population size for focused elite performance
        gamma_initial = 0.15  # Further refinement of initial gamma for a smoother start
        gamma_final = 0.00001  # More aggressive final gamma for precise exploitation
        gamma_decay = np.exp(np.log(gamma_final / gamma_initial) / (self.budget * 0.85))  # More gradual decay
        elite_fraction = 0.1  # Narrower elite selection to intensify competition
        mutation_strength = 0.003  # Adjust mutation strength for better exploration
        mutation_decay = 0.99995  # Fine-tuned mutation decay for better balance
        crossover_probability = 0.7  # Slightly increased to improve genetic diversity
        tunneling_frequency = 0.98  # More frequent tunneling to intensify escape from local minima
        directional_weight = 15.0  # Stronger directional influence for faster convergence
        feedback_rate = 0.8  # Enhanced feedback sensitivity

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
