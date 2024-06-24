import numpy as np


class RefinedAdaptiveDimensionalClimbingStrategy:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = np.full(self.dim, -5.0)
        self.ub = np.full(self.dim, 5.0)

    def __call__(self, func):
        # Population settings
        population_size = 200
        elite_size = 20
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
        recombination_prob = 0.7

        while evaluations < self.budget:
            new_population = []
            new_fitness = []

            for i in range(population_size):
                if np.random.rand() < recombination_prob:
                    # Recombination of three parents for better exploration
                    parents_indices = np.random.choice(population_size, 3, replace=False)
                    parent1, parent2, parent3 = population[parents_indices]
                    child = (parent1 + parent2 + parent3) / 3
                else:
                    # Clone with a reduced mutation effect
                    parent_idx = np.random.choice(population_size)
                    child = population[parent_idx].copy()

                # Adaptive mutation considering distance to global best
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
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness[i])

            population = np.array(new_population)
            fitness = np.array(new_fitness)

            # Update the best solution found
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.f_opt:
                self.f_opt = fitness[current_best_idx]
                self.x_opt = population[current_best_idx]

            # Incorporate elitism more effectively
            if evaluations % 250 == 0:
                elite_indices = np.argsort(fitness)[:elite_size]
                elite_individuals = population[elite_indices]
                for idx in elite_indices:
                    population[idx] = elite_individuals[np.random.choice(elite_size)]

        return self.f_opt, self.x_opt
