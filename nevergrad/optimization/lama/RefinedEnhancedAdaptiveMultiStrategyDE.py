import numpy as np


class RefinedEnhancedAdaptiveMultiStrategyDE:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # fixed dimensionality

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        bounds = np.array([-5.0, 5.0])
        population_size = 20
        F_min, F_max = 0.5, 0.9
        CR_min, CR_max = 0.1, 1.0
        restart_threshold = 0.1 * self.budget  # Restart after 10% of budget if no improvement
        memory_size = 5  # Memory size for adaptive parameters
        memory_F = np.full(memory_size, 0.5)
        memory_CR = np.full(memory_size, 0.5)

        def initialize_population():
            population = np.random.uniform(bounds[0], bounds[1], (population_size, self.dim))
            fitness = np.array([func(ind) for ind in population])
            return population, fitness

        def adaptive_parameters(memory_F, memory_CR, k):
            idx = k % memory_size
            F = np.clip(np.random.normal(memory_F[idx], 0.1), F_min, F_max)
            CR = np.clip(np.random.normal(memory_CR[idx], 0.1), CR_min, CR_max)
            return F, CR

        def update_memory(memory_F, memory_CR, F_values, CR_values, delta_fitness):
            idx = np.argmax(delta_fitness)
            fidx = np.argmin(delta_fitness)
            memory_F[fidx % memory_size] = F_values[idx]
            memory_CR[fidx % memory_size] = CR_values[idx]

        def local_restart(best_ind):
            std_dev = np.std(population, axis=0)
            new_population = best_ind + np.random.normal(scale=std_dev, size=(population_size, self.dim))
            new_population = np.clip(new_population, bounds[0], bounds[1])
            new_fitness = np.array([func(ind) for ind in new_population])
            return new_population, new_fitness

        def mutation_strategy_1(population, i, F):
            indices = list(range(population_size))
            indices.remove(i)
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            return np.clip(a + F * (b - c), bounds[0], bounds[1])

        def mutation_strategy_2(population, i, F):
            indices = list(range(population_size))
            indices.remove(i)
            a, b = population[np.random.choice(indices, 2, replace=False)]
            global_best = population[np.argmin(fitness)]
            return np.clip(a + F * (global_best - a) + F * (b - population[i]), bounds[0], bounds[1])

        def select_mutation_strategy():
            return mutation_strategy_1 if np.random.rand() < 0.5 else mutation_strategy_2

        population, fitness = initialize_population()
        evaluations = population_size

        F_values = np.full(population_size, 0.8)
        CR_values = np.full(population_size, 0.9)

        last_improvement = evaluations
        k = 0

        while evaluations < self.budget:
            if evaluations - last_improvement > restart_threshold:
                best_ind = population[np.argmin(fitness)]
                population, fitness = local_restart(best_ind)
                F_values = np.full(population_size, 0.8)
                CR_values = np.full(population_size, 0.9)
                last_improvement = evaluations

            new_population = np.zeros_like(population)
            new_fitness = np.zeros(population_size)
            new_F_values = np.zeros(population_size)
            new_CR_values = np.zeros(population_size)
            delta_fitness = np.zeros(population_size)

            for i in range(population_size):
                F_values[i], CR_values[i] = adaptive_parameters(memory_F, memory_CR, k)
                mutation_strategy = select_mutation_strategy()
                mutant = mutation_strategy(population, i, F_values[i])

                # Crossover
                cross_points = np.random.rand(self.dim) < CR_values[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                evaluations += 1

                if f_trial < fitness[i]:
                    delta_fitness[i] = fitness[i] - f_trial

                    new_population[i] = trial
                    new_fitness[i] = f_trial
                    new_F_values[i] = F_values[i]
                    new_CR_values[i] = CR_values[i]

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        last_improvement = evaluations
                else:
                    delta_fitness[i] = 0.0

                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]
                    new_F_values[i] = F_values[i]
                    new_CR_values[i] = CR_values[i]

                if evaluations >= self.budget:
                    break

            update_memory(memory_F, memory_CR, F_values, CR_values, delta_fitness)
            population, fitness = new_population, new_fitness
            F_values, CR_values = new_F_values, new_CR_values
            k += 1

        return self.f_opt, self.x_opt
