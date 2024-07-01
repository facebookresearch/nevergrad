import numpy as np


class RefinedAdaptiveQuantumEntropyDE:
    def __init__(
        self,
        budget=10000,
        population_size=100,
        elite_size=10,
        local_search_steps=100,
        F_min=0.5,
        F_max=1.0,
        Cr_min=0.2,
        Cr_max=0.9,
    ):
        self.budget = budget
        self.population_size = population_size
        self.elite_size = elite_size
        self.local_search_steps = local_search_steps
        self.dim = 5  # given dimensionality
        self.bounds = np.array([-5.0, 5.0])
        self.F_min = F_min
        self.F_max = F_max
        self.Cr_min = Cr_min
        self.Cr_max = Cr_max

    def local_search(self, individual, func):
        """Local search around an elite individual for fine-tuning using gradient estimation."""
        best_local = individual.copy()
        best_fitness = func(individual)
        step_size = 0.01
        for _ in range(self.local_search_steps):
            perturbation = np.random.uniform(-step_size, step_size, len(individual))
            candidate = individual + perturbation
            candidate = np.clip(candidate, self.bounds[0], self.bounds[1])
            candidate_fitness = func(candidate)
            if candidate_fitness < best_fitness:
                best_local = candidate
                best_fitness = candidate_fitness
        return best_local, best_fitness

    def differential_mutation(self, population, F):
        """Perform differential mutation."""
        pop_size, dim = population.shape
        idxs = np.random.choice(pop_size, 3, replace=False)
        a, b, c = population[idxs[0]], population[idxs[1]], population[idxs[2]]
        mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])
        return mutant

    def quantum_jolt(self, individual, intensity):
        """Apply quantum-inspired jolt to an individual."""
        jolt = np.random.uniform(-intensity, intensity, len(individual))
        jolted_individual = np.clip(individual + jolt, self.bounds[0], self.bounds[1])
        return jolted_individual

    def entropy_based_selection(self, population, fitness):
        """Select individuals based on entropy measure to maintain diversity."""
        probabilities = fitness / np.sum(fitness)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        if entropy < np.log(len(population)) / 2:
            selected_indices = np.argsort(fitness)[: self.elite_size]
        else:
            selected_indices = np.random.choice(len(population), self.elite_size, replace=False)
        return selected_indices

    def ensemble_optimization(self, func):
        """Ensemble of multiple strategies to enhance exploration and exploitation."""
        strategies = [self.differential_mutation, self.quantum_jolt]
        self.f_opt = np.Inf
        self.x_opt = None
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        best_idx = np.argmin(fitness)
        self.x_opt = population[best_idx]
        self.f_opt = fitness[best_idx]

        F, Cr = 0.8, 0.9

        while evaluations < self.budget:
            new_population = np.zeros_like(population)
            success_count = 0
            for i in range(self.population_size):
                strategy = np.random.choice(strategies)
                if strategy == self.differential_mutation:
                    mutant = self.differential_mutation(population, F)
                elif strategy == self.quantum_jolt:
                    intensity = 0.1 * (1 - success_count / self.population_size)
                    mutant = self.quantum_jolt(population[i], intensity)

                cross_points = np.random.rand(self.dim) < Cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    success_count += 1
                else:
                    new_population[i] = population[i]

                if trial_fitness < self.f_opt:
                    self.f_opt = trial_fitness
                    self.x_opt = trial

            population = new_population

            elite_indices = self.entropy_based_selection(population, fitness)
            elite_population = population[elite_indices]
            elite_fitness = fitness[elite_indices]
            for j in range(self.elite_size):
                elite_population[j], elite_fitness[j] = self.local_search(elite_population[j], func)
                if elite_fitness[j] < self.f_opt:
                    self.f_opt = elite_fitness[j]
                    self.x_opt = elite_population[j]

            worst_indices = np.argsort(fitness)[-self.elite_size :]
            for idx in worst_indices:
                if evaluations >= self.budget:
                    break
                elite_idx = np.random.choice(elite_indices)
                worst_idx = idx

                difference_vector = population[elite_idx] - population[worst_idx]
                new_candidate = population[worst_idx] + np.random.rand() * difference_vector
                new_candidate = np.clip(new_candidate, self.bounds[0], self.bounds[1])
                new_candidate_fitness = func(new_candidate)
                evaluations += 1

                if new_candidate_fitness < fitness[worst_idx]:
                    population[worst_idx] = new_candidate
                    fitness[worst_idx] = new_candidate_fitness
                    if new_candidate_fitness < self.f_opt:
                        self.f_opt = new_candidate_fitness
                        self.x_opt = new_candidate

            success_rate = success_count / len(population)
            if success_rate > 0.2:
                F = min(self.F_max, F + 0.1 * (success_rate - 0.2))
                Cr = max(self.Cr_min, Cr - 0.1 * (0.2 - success_rate))
            else:
                F = max(self.F_min, F - 0.1 * (0.2 - success_rate))
                Cr = min(self.Cr_max, Cr + 0.1 * (success_rate - 0.2))

        return self.f_opt, self.x_opt

    def __call__(self, func):
        return self.ensemble_optimization(func)
