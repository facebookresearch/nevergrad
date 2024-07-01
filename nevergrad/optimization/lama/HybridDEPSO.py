import numpy as np


class HybridDEPSO:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # fixed dimensionality

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        bounds = np.array([-5.0, 5.0])
        population_size = 20
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover probability
        restart_threshold = 0.2 * self.budget  # Restart after 20% of budget if no improvement

        def initialize_population():
            population = np.random.uniform(bounds[0], bounds[1], (population_size, self.dim))
            fitness = np.array([func(ind) for ind in population])
            velocities = np.zeros_like(population)
            return population, fitness, velocities

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
            new_velocities = np.zeros_like(new_population)
            return new_population, new_fitness, new_velocities

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

        def update_velocity(velocities, population, pbest, gbest, w=0.5, c1=1.5, c2=1.5):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            new_velocities = w * velocities + c1 * r1 * (pbest - population) + c2 * r2 * (gbest - population)
            return new_velocities

        def local_search(x):
            perturbation = np.random.normal(scale=0.1, size=x.shape)
            return np.clip(x + perturbation, bounds[0], bounds[1])

        population, fitness, velocities = initialize_population()
        evaluations = population_size

        F_values = np.full(population_size, F)
        CR_values = np.full(population_size, CR)

        pbest = population.copy()
        pbest_fitness = fitness.copy()
        gbest = population[np.argmin(fitness)]

        last_improvement = evaluations

        while evaluations < self.budget:
            if evaluations - last_improvement > restart_threshold:
                best_ind = population[np.argmin(fitness)]
                population, fitness, velocities = local_restart(best_ind)
                F_values = np.full(population_size, F)
                CR_values = np.full(population_size, CR)
                last_improvement = evaluations

            new_population = np.zeros_like(population)
            new_fitness = np.zeros(population_size)
            new_F_values = np.zeros(population_size)
            new_CR_values = np.zeros(population_size)
            new_velocities = np.zeros_like(velocities)

            for i in range(population_size):
                F_values, CR_values = adaptive_parameters(F_values, CR_values)

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
                    new_population[i] = trial
                    new_fitness[i] = f_trial
                    new_F_values[i] = F_values[i]
                    new_CR_values[i] = CR_values[i]
                    new_velocities[i] = velocities[i]

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        last_improvement = evaluations
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]
                    new_F_values[i] = F_values[i]
                    new_CR_values[i] = CR_values[i]
                    new_velocities[i] = velocities[i]

                if f_trial < pbest_fitness[i]:
                    pbest[i] = trial
                    pbest_fitness[i] = f_trial

                if f_trial < func(gbest):
                    gbest = trial

                if evaluations >= self.budget:
                    break

            velocities = update_velocity(velocities, population, pbest, gbest)
            population = new_population + velocities
            population = np.clip(population, bounds[0], bounds[1])
            fitness = np.array([func(ind) for ind in population])
            evaluations += population_size

            F_values, CR_values = new_F_values, new_CR_values

        return self.f_opt, self.x_opt
