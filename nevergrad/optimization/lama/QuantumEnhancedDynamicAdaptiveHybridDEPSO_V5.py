import numpy as np


class QuantumEnhancedDynamicAdaptiveHybridDEPSO_V5:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # fixed dimensionality

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        bounds = np.array([-5.0, 5.0])
        population_size = 30  # Maintain increased population size for better exploration
        w = 0.6  # Adjusted inertia weight for PSO
        c1 = 1.1  # Slightly reduced cognitive coefficient for PSO
        c2 = 1.2  # Slightly reduced social coefficient for PSO
        initial_F = 0.7  # Differential weight for DE
        initial_CR = 0.85  # Reduced crossover probability for DE
        restart_threshold = 0.1 * self.budget  # Dynamic restart threshold

        def initialize_population():
            population = np.random.uniform(bounds[0], bounds[1], (population_size, self.dim))
            velocity = np.random.uniform(-1, 1, (population_size, self.dim))
            fitness = np.array([func(ind) for ind in population])
            return population, velocity, fitness

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
            return mutation_strategy_1 if np.random.rand() < 0.6 else mutation_strategy_2

        def quantum_behavior(population, global_best, alpha=0.25, beta=0.75):
            for i in range(population_size):
                direction = global_best - population[i]
                step_size = alpha * np.random.normal(size=self.dim)
                diffusion = beta * np.random.normal(size=self.dim)
                population[i] = np.clip(
                    population[i] + direction * step_size + diffusion, bounds[0], bounds[1]
                )
            return population

        population, velocity, fitness = initialize_population()
        evaluations = population_size

        F_values = np.full(population_size, initial_F)
        CR_values = np.full(population_size, initial_CR)

        personal_best = np.copy(population)
        personal_best_fitness = np.copy(fitness)

        global_best = population[np.argmin(fitness)]
        global_best_fitness = np.min(fitness)

        last_improvement = evaluations

        while evaluations < self.budget:
            if evaluations - last_improvement > restart_threshold:
                best_ind = population[np.argmin(fitness)]
                population, fitness = local_restart(best_ind)
                F_values = np.full(population_size, initial_F)
                CR_values = np.full(population_size, initial_CR)
                last_improvement = evaluations

            new_population = np.zeros_like(population)
            new_fitness = np.zeros(population_size)
            new_F_values = np.zeros(population_size)
            new_CR_values = np.zeros(population_size)

            for i in range(population_size):
                F_values, CR_values = adaptive_parameters(F_values, CR_values)

                mutation_strategy = select_mutation_strategy()
                mutant = mutation_strategy(population, i, F_values[i])

                # Crossover
                cross_points = np.random.rand(self.dim) < CR_values[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

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

            # PSO update with quantum behavior
            r1, r2 = np.random.rand(2)
            velocity = (
                w * velocity + c1 * r1 * (personal_best - population) + c2 * r2 * (global_best - population)
            )
            population = np.clip(population + velocity, bounds[0], bounds[1])
            population = quantum_behavior(population, global_best)

            # Update personal bests
            for i in range(population_size):
                if new_fitness[i] < personal_best_fitness[i]:
                    personal_best[i] = new_population[i]
                    personal_best_fitness[i] = new_fitness[i]

            # Update global best
            if np.min(new_fitness) < global_best_fitness:
                global_best = new_population[np.argmin(new_fitness)]
                global_best_fitness = np.min(new_fitness)
                last_improvement = evaluations

            population, fitness = new_population, new_fitness
            F_values, CR_values = new_F_values, new_CR_values

            # Dynamic restart based on fitness stagnation
            if evaluations - last_improvement > restart_threshold:
                population, velocity, fitness = initialize_population()
                F_values = np.full(population_size, initial_F)
                CR_values = np.full(population_size, initial_CR)
                last_improvement = evaluations

        return self.f_opt, self.x_opt
