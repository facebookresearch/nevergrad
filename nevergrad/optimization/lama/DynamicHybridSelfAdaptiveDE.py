import numpy as np


class DynamicHybridSelfAdaptiveDE:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # fixed dimensionality

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        bounds = np.array([-5.0, 5.0])
        population_size = 20
        restart_threshold = 0.2 * self.budget  # Restart after 20% of budget if no improvement
        adaptive_interval = 50  # Adapt parameters every 50 evaluations

        def initialize_population():
            population = np.random.uniform(bounds[0], bounds[1], (population_size, self.dim))
            fitness = np.array([func(ind) for ind in population])
            return population, fitness

        def adaptive_parameters(F_values, CR_values):
            for i in range(population_size):
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

        def mutation_strategy_1(population, i, F, fitness):
            indices = list(range(population_size))
            indices.remove(i)
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            return np.clip(a + F * (b - c), bounds[0], bounds[1])

        def mutation_strategy_2(population, i, F, fitness):
            indices = list(range(population_size))
            indices.remove(i)
            a, b = population[np.random.choice(indices, 2, replace=False)]
            global_best = population[np.argmin(fitness)]
            return np.clip(a + F * (global_best - a) + F * (b - population[i]), bounds[0], bounds[1])

        def crowding_distance_selection(new_population, new_fitness, old_population, old_fitness):
            combined_population = np.vstack((new_population, old_population))
            combined_fitness = np.hstack((new_fitness, old_fitness))

            sorted_indices = np.argsort(combined_fitness)
            combined_population = combined_population[sorted_indices]
            combined_fitness = combined_fitness[sorted_indices]

            distance = np.zeros(len(combined_population))
            for i in range(self.dim):
                sorted_indices = np.argsort(combined_population[:, i])
                sorted_population = combined_population[sorted_indices]
                distance[sorted_indices[0]] = distance[sorted_indices[-1]] = np.inf
                for j in range(1, len(combined_population) - 1):
                    distance[sorted_indices[j]] += sorted_population[j + 1, i] - sorted_population[j - 1, i]

            selected_indices = np.argsort(distance)[-population_size:]
            return combined_population[selected_indices], combined_fitness[selected_indices]

        def select_mutation_strategy():
            return mutation_strategy_1 if np.random.rand() < 0.5 else mutation_strategy_2

        population, fitness = initialize_population()
        evaluations = population_size

        F_values = np.full(population_size, 0.8)
        CR_values = np.full(population_size, 0.9)

        last_improvement = evaluations

        while evaluations < self.budget:
            if evaluations - last_improvement > restart_threshold:
                best_ind = population[np.argmin(fitness)]
                population, fitness = local_restart(best_ind)
                F_values = np.full(population_size, 0.8)
                CR_values = np.full(population_size, 0.9)
                last_improvement = evaluations

            if evaluations % adaptive_interval == 0:
                F_values, CR_values = adaptive_parameters(F_values, CR_values)

            new_population = np.zeros_like(population)
            new_fitness = np.zeros(population_size)
            new_F_values = np.zeros(population_size)
            new_CR_values = np.zeros(population_size)

            for i in range(population_size):
                mutation_strategy = select_mutation_strategy()
                mutant = mutation_strategy(population, i, F_values[i], fitness)

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
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]
                    new_F_values[i] = F_values[i]
                    new_CR_values[i] = CR_values[i]

                if evaluations >= self.budget:
                    break

            # Crowding Distance Selection
            population, fitness = crowding_distance_selection(
                new_population, new_fitness, population, fitness
            )
            F_values, CR_values = new_F_values, new_CR_values

        return self.f_opt, self.x_opt
