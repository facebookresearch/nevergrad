import numpy as np


class QuantumEnhancedAdaptiveMultiPhaseDE:
    def __init__(self, budget=10000, population_size=100, elite_size=10, epsilon=1e-8):
        self.budget = budget
        self.population_size = population_size
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

    def differential_mutation(self, population, F, bounds):
        """Perform differential mutation"""
        pop_size, dim = population.shape
        idxs = np.random.choice(pop_size, 3, replace=False)
        a, b, c = population[idxs[0]], population[idxs[1]], population[idxs[2]]
        mutant = np.clip(a + F * (b - c), bounds[0], bounds[1])
        return mutant

    def quantum_jolt(self, individual, intensity, bounds):
        """Apply quantum-inspired jolt to an individual"""
        jolt = np.random.uniform(-intensity, intensity, len(individual))
        jolted_individual = np.clip(individual + jolt, bounds[0], bounds[1])
        return jolted_individual

    def cauchy_mutation(self, individual, scale):
        """Apply Cauchy mutation"""
        return individual + scale * np.tan(np.pi * (np.random.rand(len(individual)) - 0.5))

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        dim = 5  # given dimensionality
        bounds = [-5.0, 5.0]

        # Initialize population
        population = np.random.uniform(bounds[0], bounds[1], (self.population_size, dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

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
            for i in range(self.population_size):
                # Enhanced Differential Mutation
                mutant = self.differential_mutation(population, F, bounds)

                # Quantum-inspired jolt to escape local optima
                if np.random.rand() < 0.1:
                    jolt_intensity = 0.1 * (1 - success_count / self.population_size)
                    mutant = self.quantum_jolt(mutant, jolt_intensity, bounds)

                # Alternating with Cauchy mutation for exploration
                if np.random.rand() < 0.1:
                    mutant = self.cauchy_mutation(mutant, scale=0.1)

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

            population = new_population

            # Adaptive Population Size
            if success_count / self.population_size > 0.2:
                self.population_size = min(self.population_size + 10, self.population_size * 2)
            elif success_count / self.population_size < 0.1:
                self.population_size = max(self.population_size - 10, self.population_size // 2)

            # Ensure the population size is within bounds
            self.population_size = np.clip(self.population_size, 10, self.population_size * 2)

            # Resize population arrays if necessary
            if self.population_size > population.shape[0]:
                new_pop = np.random.uniform(
                    bounds[0], bounds[1], (self.population_size - population.shape[0], dim)
                )
                population = np.vstack((population, new_pop))
                fitness = np.hstack((fitness, np.array([func(ind) for ind in new_pop])))
            elif self.population_size < population.shape[0]:
                population = population[: self.population_size]
                fitness = fitness[: self.population_size]

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
