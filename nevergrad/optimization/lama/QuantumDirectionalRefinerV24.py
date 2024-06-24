import numpy as np


class QuantumDirectionalRefinerV24:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Tuning parameters based on feedback from previous version
        population_size = 1000  # Expanded population size for more coverage
        gamma_initial = 0.5  # Softer initial quantum weight for gentle exploration
        gamma_final = 0.00005  # Miniscule final quantum weight for ultra-fine tuning
        gamma_decay = np.exp(np.log(gamma_final / gamma_initial) / (self.budget * 0.6))  # Custom decay rate
        elite_fraction = 0.25  # Increased elite fraction to maintain more high-quality solutions
        mutation_strength = 0.005  # Reduced mutation strength to prevent disruptive mutations
        mutation_decay = 0.9997  # Extended impact of mutations throughout more generations
        crossover_probability = 0.75  # Reduced crossover probability to emphasize mutations and elitism
        tunneling_frequency = 0.85  # Increased frequency for quantum jumps to escape local minima
        directional_weight = 8.0  # Increased weight to forcefully pull towards elites
        feedback_rate = 0.55  # Enhanced feedback sensitivity for reacting to function landscape

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
