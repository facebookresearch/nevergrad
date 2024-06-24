import numpy as np


class PrecisionIncrementalEvolutionStrategy:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = np.full(self.dim, -5.0)
        self.ub = np.full(self.dim, 5.0)

    def __call__(self, func):
        population_size = 100
        mutation_scale = 0.1
        elite_size = 10
        recombination_weight = 0.7
        evaluations = 0

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations += population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        while evaluations < self.budget:
            new_population = []
            new_fitness = []

            # Generate new candidates
            for i in range(population_size):
                parents_indices = np.random.choice(population_size, 2, replace=False)
                parent1, parent2 = population[parents_indices]

                # Blend recombination
                child = recombination_weight * parent1 + (1 - recombination_weight) * parent2

                # Mutation
                mutation = np.random.normal(0, mutation_scale, self.dim)
                child += mutation
                child = np.clip(child, self.lb, self.ub)

                # Evaluate new candidate
                child_fitness = func(child)
                evaluations += 1

                # Selection step
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

            # Introduce elitism
            if evaluations % 500 == 0:
                elite_indices = np.argsort(fitness)[:elite_size]
                elite_individuals = population[elite_indices]
                replace_indices = np.random.choice(population_size, elite_size, replace=False)
                population[replace_indices] = elite_individuals

        return self.f_opt, self.x_opt
