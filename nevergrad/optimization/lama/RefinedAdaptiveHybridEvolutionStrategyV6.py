import numpy as np
import scipy.stats as stats


class RefinedAdaptiveHybridEvolutionStrategyV6:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        population_size = 500
        mutation_rate = 0.15  # Slightly increased mutation rate
        mutation_scale = lambda t: 0.1 * np.exp(-0.0001 * t)  # Gradual decrease in mutation scale
        crossover_rate = 0.9  # Adjusted crossover rate for balance
        elite_size = int(0.2 * population_size)  # Increased elite size

        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(x) for x in population])
        evaluations = population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        while evaluations < self.budget:
            new_population = []
            elite_indices = np.argsort(fitness)[:elite_size]
            elites = population[elite_indices]

            # Using tournament selection for parent selection
            for _ in range(population_size - elite_size):
                tournament = np.random.choice(population_size, 5, replace=False)
                t1, t2 = np.argmin(fitness[tournament][:3]), np.argmin(fitness[tournament][3:])
                parent1, parent2 = population[tournament[t1]], population[tournament[t2]]

                if np.random.random() < crossover_rate:
                    cross_point = np.random.randint(1, self.dim)
                    child = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
                else:
                    child = parent1.copy()

                if np.random.random() < mutation_rate:
                    mutation = np.random.normal(0, mutation_scale(evaluations), self.dim)
                    child = np.clip(child + mutation, self.lb, self.ub)

                new_population.append(child)

            new_fitness = np.array([func(x) for x in new_population])
            evaluations += len(new_population)

            population = np.vstack((elites, new_population))
            fitness = np.concatenate([fitness[elite_indices], new_fitness])

            current_best_idx = np.argmin(fitness)
            current_best_f = fitness[current_best_idx]
            if current_best_f < self.f_opt:
                self.f_opt = current_best_f
                self.x_opt = population[current_best_idx]

        return self.f_opt, self.x_opt
