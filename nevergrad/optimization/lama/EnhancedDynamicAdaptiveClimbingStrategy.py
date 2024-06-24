import numpy as np


class EnhancedDynamicAdaptiveClimbingStrategy:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = np.full(self.dim, -5.0)
        self.ub = np.full(self.dim, 5.0)

    def __call__(self, func):
        population_size = 250
        elite_size = 30
        evaluations = 0

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations += population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        # Strategy parameters
        mutation_scale = 0.1
        adaptive_factor = 0.95
        recombination_prob = 0.85  # Increased probability for recombination

        # Enhancing exploration and exploitation
        last_best_fitness = np.inf

        while evaluations < self.budget:
            success_count = 0
            new_population = []
            new_fitness = []

            for i in range(population_size):
                if np.random.rand() < recombination_prob:
                    parents_indices = np.random.choice(population_size, 4, replace=False)  # Use 4 parents
                    parent1, parent2, parent3, parent4 = population[parents_indices]
                    child = (parent1 + parent2 + parent3 + parent4) / 4  # Average of 4 parents
                else:
                    parent_idx = np.random.choice(population_size)
                    child = population[parent_idx].copy()

                distance_to_best = np.linalg.norm(population[best_idx] - child)
                individual_mutation_scale = mutation_scale * adaptive_factor ** (distance_to_best)
                mutation = np.random.normal(0, individual_mutation_scale, self.dim)
                child += mutation
                child = np.clip(child, self.lb, self.ub)

                child_fitness = func(child)
                evaluations += 1

                if child_fitness < fitness[i]:
                    new_population.append(child)
                    new_fitness.append(child_fitness)
                    success_count += 1
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness[i])

            population = np.array(new_population)
            fitness = np.array(new_fitness)

            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.f_opt:
                self.f_opt = fitness[current_best_idx]
                self.x_opt = population[current_best_idx]

                if fitness[current_best_idx] < last_best_fitness:
                    last_best_fitness = fitness[current_best_idx]
                    success_rate = success_count / population_size
                    adaptive_factor = max(0.8, adaptive_factor - 0.05 * success_rate)
                    mutation_scale = mutation_scale + 0.02 * (1 - success_rate)

            # Elite reinforcement for better global optima stabilization
            if evaluations % 200 == 0:
                elite_indices = np.argsort(fitness)[:elite_size]
                elite_individuals = population[elite_indices]
                for idx in range(population_size):
                    if idx not in elite_indices:
                        population[idx] = elite_individuals[np.random.choice(elite_size)]

        return self.f_opt, self.x_opt
