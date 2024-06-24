import numpy as np


class EnhancedQuantumMultiPhaseAdaptiveDE_v10:
    def __init__(self, budget=10000, population_size=100, elite_size=10, epsilon=1e-8):
        self.budget = budget
        self.dim = 5  # as given in the problem statement
        self.bounds = [-5.0, 5.0]
        self.population_size = population_size
        self.elite_size = elite_size
        self.epsilon = epsilon

    def local_search(self, elite_individual, func):
        """Local search around elite individual for fine-tuning"""
        best_local = elite_individual.copy()
        best_fitness = func(elite_individual)
        for _ in range(5):  # small fixed number of local steps
            perturbation = np.random.uniform(-0.1, 0.1, self.dim)
            candidate = elite_individual + perturbation
            candidate = np.clip(candidate, self.bounds[0], self.bounds[1])
            candidate_fitness = func(candidate)
            if candidate_fitness < best_fitness:
                best_local = candidate
                best_fitness = candidate_fitness
        return best_local, best_fitness

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        # Track best individual
        best_idx = np.argmin(fitness)
        self.x_opt = population[best_idx]
        self.f_opt = fitness[best_idx]

        # Differential Evolution parameters
        F = 0.8
        Cr = 0.9

        # Stagnation and success tracking
        stagnation_threshold = 50
        stagnation_counter = 0

        # Elite tracking
        elite_indices = np.argsort(fitness)[: self.elite_size]
        elite_population = population[elite_indices]
        elite_fitness = fitness[elite_indices]

        # Adaptive parameter ranges
        F_min, F_max = 0.5, 1.0
        Cr_min, Cr_max = 0.2, 0.9

        # History of improvements
        improvements = []

        while evaluations < self.budget:
            new_population = np.zeros_like(population)
            success_count = 0
            for i in range(self.population_size):
                # Select indices for mutation
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)

                # Dimension-wise adaptive mutation
                mutant = np.zeros(self.dim)
                for d in range(self.dim):
                    if np.random.rand() < 0.5:
                        elite_idx = np.random.choice(self.elite_size)
                        elite_ind = elite_population[elite_idx]
                        centroid = np.mean(population[[a, b, c], d])
                        mutant[d] = (
                            centroid
                            + F * (population[a, d] - population[b, d])
                            + 0.1 * (elite_ind[d] - population[i, d])
                        )
                    else:
                        mutant[d] = population[a, d] + F * (population[b, d] - population[c, d])

                # Quantum-inspired jolt to escape local optima
                if np.random.rand() < 0.1:
                    jolt_intensity = 0.1 * (1 - success_count / self.population_size)
                    jolt = np.random.uniform(-jolt_intensity, jolt_intensity, self.dim)
                    mutant += jolt

                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                # Crossover strategy
                cross_points = np.random.rand(self.dim) < Cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

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
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

            population = new_population

            elite_indices = np.argsort(fitness)[: self.elite_size]
            elite_population = population[elite_indices]
            elite_fitness = fitness[elite_indices]

            # Perform local search on elite individuals
            for j in range(self.elite_size):
                elite_population[j], elite_fitness[j] = self.local_search(elite_population[j], func)
                if elite_fitness[j] < self.f_opt:
                    self.f_opt = elite_fitness[j]
                    self.x_opt = elite_population[j]

            success_rate = success_count / self.population_size
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

            if stagnation_counter > stagnation_threshold:
                phase = (evaluations // stagnation_threshold) % 3
                diversity_factor = 0.5 if phase == 2 else 1.0

                if phase == 0:
                    reinit_indices = np.random.choice(
                        self.population_size, self.population_size // 3, replace=False
                    )
                elif phase == 1:
                    reinit_indices = np.random.choice(
                        self.population_size, self.population_size // 2, replace=False
                    )
                else:
                    reinit_indices = np.random.choice(
                        self.population_size, self.population_size // 2, replace=False
                    )

                population[reinit_indices] = (
                    np.random.uniform(self.bounds[0], self.bounds[1], (len(reinit_indices), self.dim))
                    * diversity_factor
                )
                fitness[reinit_indices] = np.array([func(ind) for ind in population[reinit_indices]])
                evaluations += len(reinit_indices)
                best_idx = np.argmin(fitness)
                self.x_opt = population[best_idx]
                self.f_opt = fitness[best_idx]
                stagnation_counter = 0

        return self.f_opt, self.x_opt
