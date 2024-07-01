import numpy as np


class QuantumDirectionalEnhancerV13:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Strategy parameters
        population_size = 500  # Increased population size for a broader genetic base
        gamma_initial = 0.95  # Slightly increased initial gamma for stronger initial exploration
        gamma_final = 0.000001  # Smaller final gamma for finer exploitation at the end
        gamma_decay = np.exp(
            np.log(gamma_final / gamma_initial) / (self.budget * 0.75)
        )  # Fine-tuned decay rate
        elite_fraction = 0.3  # Increased elite fraction for a more robust selection process
        mutation_strength = 0.0005  # Lower mutation strength for finer perturbations
        mutation_decay = 0.999  # Slower decay to maintain mutation effectiveness longer
        crossover_probability = 0.9  # Increased crossover probability for enhanced genetic mixing
        tunneling_frequency = 0.85  # Increased tunneling frequency for more frequent quantum leaps
        directional_weight = 0.98  # Increased weight for tunneling direction for better target pursuit
        feedback_rate = 0.15  # Increased feedback rate to enhance reactive adaptation

        # Initialize the population and evaluate fitness
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations_left = self.budget - population_size

        while evaluations_left > 0:
            gamma = gamma_initial * (gamma_decay ** (self.budget - evaluations_left))
            mutation_strength *= mutation_decay  # Adjust mutation strength dynamically

            # Select elite individuals
            elite_count = int(population_size * elite_fraction)
            elite_indices = np.argsort(fitness)[:elite_count]
            elite_individuals = population[elite_indices]
            new_population = list(elite_individuals)
            new_fitness = list(fitness[elite_indices])

            # Reproduction through crossover and mutation
            while len(new_population) < population_size:
                if evaluations_left <= 0:
                    break

                idxs = np.random.choice(elite_count, 2, replace=False)
                parent1, parent2 = new_population[idxs[0]], new_population[idxs[1]]

                # Perform crossover
                if np.random.random() < crossover_probability:
                    crossover_point = np.random.randint(1, self.dim)
                    offspring = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                else:
                    offspring = np.array(parent1)

                # Mutation
                mutation = np.random.normal(0, mutation_strength, self.dim)
                offspring += mutation
                offspring = np.clip(offspring, -5.0, 5.0)

                # Quantum tunneling with directional bias
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
