import numpy as np
from scipy.spatial import distance


class AdvancedDynamicCrowdedDE:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # fixed dimensionality

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        bounds = np.array([-5.0, 5.0])
        init_population_size = 20
        F = 0.8  # Initial Differential weight
        CR = 0.9  # Crossover probability
        restart_threshold = 0.1 * self.budget  # Restart after 10% of budget if no improvement

        def initialize_population(size):
            population = np.random.uniform(bounds[0], bounds[1], (size, self.dim))
            fitness = np.array([func(ind) for ind in population])
            return population, fitness

        def adaptive_parameters(F_values, CR_values):
            for i in range(len(F_values)):
                if np.random.rand() < 0.1:
                    F_values[i] = 0.1 + 0.9 * np.random.rand()
                if np.random.rand() < 0.1:
                    CR_values[i] = np.random.rand()
            return F_values, CR_values

        def local_restart(best_ind):
            std_dev = np.std(population, axis=0)
            new_population = best_ind + np.random.normal(scale=std_dev, size=(population_size, self.dim))
            new_population = np.clip(new_population, bounds[0], bounds[1])
            new_fitness = np.array([func(ind) for ind in new_population])
            return new_population, new_fitness

        def crowding_distance_sort(population, fitness):
            distances = distance.cdist(population, population, "euclidean")
            sorted_indices = np.argsort(fitness)
            crowding_distances = np.zeros(len(population))
            crowding_distances[sorted_indices[0]] = np.inf
            crowding_distances[sorted_indices[-1]] = np.inf

            for i in range(1, len(population) - 1):
                crowding_distances[sorted_indices[i]] = distances[
                    sorted_indices[i - 1], sorted_indices[i + 1]
                ]

            return np.argsort(crowding_distances)

        def mutation_strategy_1(population, i, F):
            indices = list(range(len(population)))
            indices.remove(i)
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            return np.clip(a + F * (b - c), bounds[0], bounds[1])

        def mutation_strategy_2(population, i, F):
            indices = list(range(len(population)))
            indices.remove(i)
            a, b = population[np.random.choice(indices, 2, replace=False)]
            global_best = population[np.argmin(fitness)]
            return np.clip(a + F * (global_best - a) + F * (b - population[i]), bounds[0], bounds[1])

        def select_mutation_strategy(score):
            return mutation_strategy_1 if score < 0.5 else mutation_strategy_2

        def archive_mutation(archive, population, F):
            if len(archive) > 0:
                idx = np.random.randint(0, len(archive))
                return np.clip(population + F * (archive[idx] - population), bounds[0], bounds[1])
            else:
                return population

        population, fitness = initialize_population(init_population_size)
        evaluations = init_population_size
        archive = []

        F_values = np.full(init_population_size, F)
        CR_values = np.full(init_population_size, CR)

        last_improvement = evaluations

        while evaluations < self.budget:
            if evaluations - last_improvement > restart_threshold:
                best_ind = population[np.argmin(fitness)]
                population, fitness = local_restart(best_ind)
                F_values = np.full(len(population), F)
                CR_values = np.full(len(population), CR)
                last_improvement = evaluations

            sorted_indices = crowding_distance_sort(population, fitness)
            new_population = np.zeros_like(population)
            new_fitness = np.zeros(len(population))
            new_F_values = np.zeros(len(population))
            new_CR_values = np.zeros(len(population))

            for idx in range(len(population)):
                i = sorted_indices[idx]
                F_values, CR_values = adaptive_parameters(F_values, CR_values)

                mutation_strategy = select_mutation_strategy(fitness[i])
                mutant = mutation_strategy(population, i, F_values[i])

                # Archive-based mutation
                if np.random.rand() < 0.3:
                    mutant = archive_mutation(archive, mutant, F_values[i])

                # Crossover
                cross_points = np.random.rand(self.dim) < CR_values[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                evaluations += 1

                if f_trial < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial
                    new_F_values[i] = F_values[i]
                    new_CR_values[i] = CR_values[i]

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        last_improvement = evaluations

                    # Update Archive
                    archive.append(population[i])
                    if len(archive) > init_population_size:
                        archive.pop(0)
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]
                    new_F_values[i] = F_values[i]
                    new_CR_values[i] = CR_values[i]

                if evaluations >= self.budget:
                    break

            if evaluations - last_improvement > restart_threshold // 2:
                population_size = int(len(population) * 0.9)
            else:
                population_size = int(len(population) * 1.1)
            population_size = max(10, min(30, population_size))

            population, fitness = new_population[:population_size], new_fitness[:population_size]
            F_values, CR_values = new_F_values[:population_size], new_CR_values[:population_size]

        return self.f_opt, self.x_opt
