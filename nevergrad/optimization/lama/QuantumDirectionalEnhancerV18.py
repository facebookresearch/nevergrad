import numpy as np


class QuantumDirectionalEnhancerV18:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # The dimensionality is fixed at 5

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Strategy parameters with further enhancements
        population_size = 300  # Reduced population for faster computation while maintaining diversity
        gamma_initial = 0.9  # Reduced initial exploration intensity
        gamma_final = 0.0001  # Maintained low final gamma for fine exploitation
        gamma_decay = np.exp(
            np.log(gamma_final / gamma_initial) / (self.budget * 0.8)
        )  # Adjusted decay rate for an earlier shift to fine-tuning
        elite_fraction = 0.1  # Further reduced elite fraction to stimulate more competition
        mutation_strength = 0.005  # Adjusted mutation strength for broader exploration early on
        mutation_decay = 0.9998  # Adjusted decay to sustain mutation relevance longer
        crossover_probability = 0.85  # Increased to encourage more genetic mixing
        tunneling_frequency = 0.95  # Increased frequency to leverage quantum effects more widely
        directional_weight = 1.2  # Higher weight to enhance directional exploitation
        feedback_rate = 0.2  # Increased feedback for stronger adaptive responses

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
