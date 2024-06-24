import numpy as np


class EnhancedDynamicEvolutionStrategy:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        population_size = 200
        mutation_rate = 0.08
        mutation_scale = 0.2
        crossover_rate = 0.7
        elite_size = 20

        # Initialize population uniformly within the bounds
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        evaluations = population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        while evaluations < self.budget:
            new_population = []
            elite_indices = np.argsort(fitness)[:elite_size]
            elites = population[elite_indices]

            # Safe roulette wheel selection using max fitness scaling
            max_fitness = np.max(fitness)
            adjusted_fitness = max_fitness - fitness + 1e-9  # Adding small constant to avoid zero probability
            probabilities = adjusted_fitness / adjusted_fitness.sum()

            chosen_parents = np.random.choice(
                population_size, size=population_size - elite_size, p=probabilities
            )
            parents = population[chosen_parents]

            # Crossover and mutation
            np.random.shuffle(parents)
            for i in range(0, len(parents) - 1, 2):
                parent1, parent2 = parents[i], parents[i + 1]
                if np.random.random() < crossover_rate:
                    cross_point = np.random.randint(1, self.dim)
                    child1 = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
                    child2 = np.concatenate([parent2[:cross_point], parent1[cross_point:]])
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                new_population.extend([child1, child2])

            # Mutation in the new population
            new_population = np.array(new_population)
            mutation_masks = np.random.rand(len(new_population), self.dim) < mutation_rate
            mutations = np.random.normal(0, mutation_scale, (len(new_population), self.dim))
            new_population = np.clip(new_population + mutation_masks * mutations, self.lb, self.ub)

            # Evaluate new population
            new_fitness = np.array([func(ind) for ind in new_population])
            evaluations += len(new_population)

            # Replace the worst with new individuals
            population = np.vstack((elites, new_population))
            fitness = np.concatenate([fitness[elite_indices], new_fitness])

            # Update best solution found
            current_best_idx = np.argmin(fitness)
            current_best_f = fitness[current_best_idx]
            if current_best_f < self.f_opt:
                self.f_opt = current_best_f
                self.x_opt = population[current_best_idx]

        return self.f_opt, self.x_opt
