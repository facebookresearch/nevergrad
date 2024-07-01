import numpy as np


class RefinedEvolutionaryTuningStrategy:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        population_size = 100  # Adjusted population size for better handling in various landscapes
        mutation_rate = 0.1  # Adjusted mutation rate for maintaining diversity
        mutation_scale = 0.1  # Mutation scale maintained for precise mutations
        crossover_rate = 0.7  # Adjusted crossover rate for optimal mixing
        elite_size = 10  # Percentage of population to maintain as elite

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        evaluations = population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        while evaluations < self.budget:
            # Selection via tournament selection
            tournament_size = 5
            selected_indices = []
            for _ in range(population_size - elite_size):
                participants = np.random.choice(population_size, tournament_size, replace=False)
                best_participant = participants[np.argmin(fitness[participants])]
                selected_indices.append(best_participant)

            mating_pool = population[selected_indices]

            # Crossover
            np.random.shuffle(mating_pool)
            children = []
            for i in range(0, len(selected_indices) - 1, 2):
                if np.random.random() < crossover_rate:
                    cross_point = np.random.randint(1, self.dim)
                    child1 = np.concatenate((mating_pool[i][:cross_point], mating_pool[i + 1][cross_point:]))
                    child2 = np.concatenate((mating_pool[i + 1][:cross_point], mating_pool[i][cross_point:]))
                else:
                    child1, child2 = mating_pool[i], mating_pool[i + 1]
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

            # Track the best found solution
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.f_opt:
                self.f_opt = fitness[current_best_idx]
                self.x_opt = population[current_best_idx]

        return self.f_opt, self.x_opt
