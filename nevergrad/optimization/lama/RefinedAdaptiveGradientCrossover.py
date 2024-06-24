import numpy as np


class RefinedAdaptiveGradientCrossover:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        population_size = 100
        mutation_rate = 0.1
        mutation_scale = 0.2
        crossover_rate = 0.85
        elite_size = 15

        # Initialize population and calculate fitness
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        evaluations = population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        while evaluations < self.budget:
            # Tournament Selection
            tournament_size = 3
            parents = []
            for _ in range(population_size):
                idx = np.random.choice(range(population_size), size=tournament_size, replace=False)
                parents.append(population[idx[np.argmin(fitness[idx])]])
            parents = np.array(parents)

            # Crossover: Blend Crossover (BLX-alpha)
            alpha = 0.5
            children = []
            for i in range(0, parents.shape[0], 2):
                if i + 1 >= parents.shape[0]:
                    continue
                p1, p2 = parents[i], parents[i + 1]
                gamma = (1 + 2 * alpha) * np.random.random(self.dim) - alpha
                child1 = gamma * p1 + (1 - gamma) * p2
                child2 = gamma * p2 + (1 - gamma) * p1
                children.extend([child1, child2])
            children = np.clip(children, self.lb, self.ub)

            # Mutation: Uniform mutation
            for child in children:
                if np.random.rand() < mutation_rate:
                    mutate_dims = np.random.randint(0, self.dim)
                    child[mutate_dims] += mutation_scale * np.random.randn()

            # Ensure all mutations are within bounds
            children = np.clip(children, self.lb, self.ub)

            # Evaluate children
            children_fitness = np.array([func(child) for child in children])
            evaluations += len(children)

            # Elitism and new population formation
            combined_population = np.vstack([population, children])
            combined_fitness = np.concatenate([fitness, children_fitness])

            # Select the best to form the next generation
            elite_indices = np.argsort(combined_fitness)[:elite_size]
            non_elite_indices = np.argsort(combined_fitness)[elite_size:population_size]

            population = np.vstack(
                [
                    combined_population[elite_indices],
                    combined_population[non_elite_indices][: population_size - elite_size],
                ]
            )
            fitness = np.concatenate(
                [
                    combined_fitness[elite_indices],
                    combined_fitness[non_elite_indices][: population_size - elite_size],
                ]
            )

            # Update the best solution found
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.f_opt:
                self.f_opt = fitness[current_best_idx]
                self.x_opt = population[current_best_idx]

        return self.f_opt, self.x_opt
