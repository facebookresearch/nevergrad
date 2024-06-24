import numpy as np


class RefinedAdaptiveEvolutionStrategy:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        population_size = 100  # Adapted population size
        mutation_rate = 0.1  # Adjusted mutation rate
        mutation_scale = 0.3  # Increased mutation scale
        crossover_rate = 0.7  # Adjusted crossover rate
        elite_size = 10  # Reduced elite size

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        evaluations = population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        while evaluations < self.budget:
            # Selection: Roulette Wheel Selection
            fitness_scores = fitness.max() - fitness  # Convert fitness to a maximization problem
            if fitness_scores.sum() > 0:
                probabilities = fitness_scores / fitness_scores.sum()
            else:
                probabilities = np.ones_like(fitness_scores) / len(fitness_scores)
            selected_indices = np.random.choice(
                np.arange(population_size), size=population_size - elite_size, p=probabilities, replace=True
            )
            mating_pool = population[selected_indices]

            # Crossover: Uniform crossover
            children = []
            for i in range(0, len(mating_pool) - 1, 2):
                child1, child2 = np.copy(mating_pool[i]), np.copy(mating_pool[i + 1])
                for d in range(self.dim):
                    if np.random.rand() < crossover_rate:
                        child1[d], child2[d] = child2[d], child1[d]
                children.append(child1)
                children.append(child2)

            # Mutation
            children = np.array(children)
            mutation_mask = np.random.rand(children.shape[0], self.dim) < mutation_rate
            mutations = np.random.normal(0, mutation_scale, children.shape)
            children = np.clip(children + mutation_mask * mutations, self.lb, self.ub)

            # Evaluate new individuals
            new_fitness = np.array([func(x) for x in children])
            evaluations += len(children)

            # Elitism and replacement
            elites_indices = np.argsort(fitness)[:elite_size]
            elites = population[elites_indices]
            elite_fitness = fitness[elites_indices]

            combined_population = np.vstack([elites, children])
            combined_fitness = np.concatenate([elite_fitness, new_fitness])

            # Select the next generation
            sorted_indices = np.argsort(combined_fitness)
            population = combined_population[sorted_indices[:population_size]]
            fitness = combined_fitness[sorted_indices[:population_size]]

            # Update the best solution found
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.f_opt:
                self.f_opt = fitness[current_best_idx]
                self.x_opt = population[current_best_idx]

        return self.f_opt, self.x_opt
