import numpy as np


class QuantumEnhancedAdaptiveDifferentialEvolution_v4:
    def __init__(self, budget=10000, initial_population_size=100, elite_size=10, epsilon=1e-8):
        self.budget = budget
        self.initial_population_size = initial_population_size
        self.elite_size = elite_size
        self.epsilon = epsilon

    def local_search(self, elite_individual, func):
        """Local search around elite individual for fine-tuning"""
        best_local = elite_individual.copy()
        best_fitness = func(elite_individual)
        for _ in range(5):  # small fixed number of local steps
            perturbation = np.random.uniform(-0.1, 0.1, len(elite_individual))
            candidate = elite_individual + perturbation
            candidate = np.clip(candidate, -5.0, 5.0)
            candidate_fitness = func(candidate)
            if candidate_fitness < best_fitness:
                best_local = candidate
                best_fitness = candidate_fitness
        return best_local, best_fitness

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        dim = 5  # given dimensionality
        bounds = [-5.0, 5.0]
        population_size = self.initial_population_size

        # Initialize population
        population = np.random.uniform(bounds[0], bounds[1], (population_size, dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        # Track best individual
        best_idx = np.argmin(fitness)
        self.x_opt = population[best_idx]
        self.f_opt = fitness[best_idx]

        # Differential Evolution parameters
        F = 0.8
        Cr = 0.9

        # Adaptive parameter ranges
        F_min, F_max = 0.5, 1.0
        Cr_min, Cr_max = 0.2, 0.9

        # History of improvements
        improvements = []

        while evaluations < self.budget:
            new_population = np.zeros_like(population)
            success_count = 0
            for i in range(population_size):
                # Select indices for mutation
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)

                # Enhanced Differential Mutation
                mutant = population[a] + F * (population[b] - population[c])
                mutant = np.clip(mutant, bounds[0], bounds[1])

                # Quantum-inspired jolt to escape local optima
                if np.random.rand() < 0.1:
                    jolt_intensity = 0.1 * (1 - success_count / population_size)
                    jolt = np.random.uniform(-jolt_intensity, jolt_intensity, dim)
                    mutant += jolt

                mutant = np.clip(mutant, bounds[0], bounds[1])

                # Crossover strategy
                cross_points = np.random.rand(dim) < Cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dim)] = True

                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    success_count += 1
                    improvements.append(evaluations)
                else:
                    new_population[i] = population[i]

                if trial_fitness < self.f_opt:
                    self.f_opt = trial_fitness
                    self.x_opt = trial

            # Adaptive Population Size
            if success_count / population_size > 0.2:
                population_size = min(population_size + 10, self.initial_population_size * 2)
            elif success_count / population_size < 0.1:
                population_size = max(population_size - 10, self.initial_population_size // 2)

            # Ensure the population size is within bounds
            population_size = np.clip(population_size, 10, self.initial_population_size * 2)

            # Resize population arrays if necessary
            if population_size > len(population):
                new_pop = np.random.uniform(bounds[0], bounds[1], (population_size - len(population), dim))
                population = np.vstack((population, new_pop))
                fitness = np.hstack((fitness, np.array([func(ind) for ind in new_pop])))
            elif population_size < len(population):
                population = population[:population_size]
                fitness = fitness[:population_size]

            # Perform local search on elite individuals
            elite_indices = np.argsort(fitness)[: self.elite_size]
            elite_population = population[elite_indices]
            elite_fitness = fitness[elite_indices]
            for j in range(self.elite_size):
                elite_population[j], elite_fitness[j] = self.local_search(elite_population[j], func)
                if elite_fitness[j] < self.f_opt:
                    self.f_opt = elite_fitness[j]
                    self.x_opt = elite_population[j]

            # Self-Adaptive Control Parameters
            success_rate = success_count / len(population)
            if success_rate > 0.2:
                F = min(F_max, F + 0.1 * (success_rate - 0.2))
                Cr = max(Cr_min, Cr - 0.1 * (0.2 - success_rate))
            else:
                F = max(F_min, F - 0.1 * (0.2 - success_rate))
                Cr = min(Cr_max, Cr + 0.1 * (success_rate - 0.2))

            if len(improvements) > 5:
                recent_improvements = evaluations - np.array(improvements[-5:])
                average_gap = np.mean(recent_improvements)
                if average_gap < 10:
                    F = min(F_max, F + 0.1)
                elif average_gap > 100:
                    F = max(F_min, F - 0.1)

        return self.f_opt, self.x_opt
