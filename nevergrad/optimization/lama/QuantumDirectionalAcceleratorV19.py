import numpy as np


class QuantumDirectionalAcceleratorV19:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # The dimensionality is fixed at 5

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Strategic parameters further refined for enhanced performance
        population_size = 250  # Reduced population for quicker generations
        gamma_initial = 0.8  # Reduced initial exploration intensity
        gamma_final = 0.00005  # Lower final gamma for finer exploitation
        gamma_decay = np.exp(
            np.log(gamma_final / gamma_initial) / (self.budget * 0.7)
        )  # Faster decay for earlier fine-tuning
        elite_fraction = 0.05  # Further reduced elite fraction to increase competition
        mutation_strength = 0.01  # Adjusted mutation strength for broader exploration
        mutation_decay = 0.99975  # Slower decay to sustain mutation relevance
        crossover_probability = 0.9  # Increased to promote more genetic variation
        tunneling_frequency = 0.98  # Increased frequency for more frequent quantum effects
        directional_weight = 1.5  # Increased weight to improve directional exploitation
        feedback_rate = 0.25  # Increased feedback for stronger adaptive responses

        # Initialize and evaluate the initial population
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

                # Select parents from elites
                parents = np.random.choice(elite_count, 2, replace=False)
                parent1, parent2 = elite_individuals[parents[0]], elite_individuals[parents[1]]

                # Crossover
                if np.random.random() < crossover_probability:
                    offspring = np.where(np.random.rand(self.dim) < 0.5, parent1, parent2)
                else:
                    offspring = np.array(parent1)  # No crossover, direct copy

                # Mutation
                mutation = np.random.normal(0, mutation_strength, self.dim)
                offspring += mutation
                offspring = np.clip(offspring, -5.0, 5.0)

                # Quantum tunneling to potentially escape local minima
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
