import numpy as np


class QuantumDirectionalFusionOptimizerV2:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality as per problem specifications

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        # Configuration settings
        population_size = 25  # Slightly larger population for better exploration
        gamma_initial = 0.1  # Increased initial gamma for broader initial search scope
        gamma_final = 0.0001  # Fine tuned final refinement
        gamma_decay = np.exp(np.log(gamma_final / gamma_initial) / (self.budget * 0.8))
        elite_fraction = 0.4  # Reduced elite fraction to promote diversity
        mutation_strength = 0.01  # Increased mutation for greater exploration
        mutation_decay = 0.99  # Slower decay to maintain diversity throughout
        crossover_probability = 0.95  # Increased crossover to enhance beneficial gene mixing
        tunneling_frequency = 0.05  # Reduced frequency to focus on top-performing individuals
        directional_weight = 100.0  # Increased emphasis on directional exploitation
        feedback_rate = 0.1  # Increased feedback rate to quickly adapt to promising regions

        # Initialize the population uniformly within the search space
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

                # Select two parents from the elite pool
                parents = np.random.choice(elite_count, 2, replace=False)
                parent1, parent2 = elite_individuals[parents[0]], elite_individuals[parents[1]]

                # Perform crossover based on probability
                if np.random.random() < crossover_probability:
                    offspring = np.where(np.random.rand(self.dim) < 0.5, parent1, parent2)
                else:
                    offspring = np.array(parent1)

                # Apply mutation
                mutation = np.random.normal(0, mutation_strength, self.dim)
                offspring += mutation
                offspring = np.clip(offspring, -5.0, 5.0)

                # Quantum tunneling with directional bias less frequently
                if np.random.rand() < tunneling_frequency:
                    best_idx = np.argmin(new_fitness)
                    best_individual = new_population[best_idx]
                    direction = best_individual - offspring
                    feedback = np.mean([f - self.f_opt for f in new_fitness]) * feedback_rate
                    offspring += gamma * directional_weight * direction + feedback

                # Evaluate the new candidate
                offspring_fitness = func(offspring)
                evaluations_left -= 1

                new_population.append(offspring)
                new_fitness.append(offspring_fitness)

                # Update the optimum if the current candidate is better
                if offspring_fitness < self.f_opt:
                    self.f_opt = offspring_fitness
                    self.x_opt = offspring

            population = np.array(new_population)
            fitness = np.array(new_fitness)

        return self.f_opt, self.x_opt
