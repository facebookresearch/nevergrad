import numpy as np


class EnhancedQuantumMultiStrategyOptimization_v2:
    def __init__(
        self,
        budget=10000,
        population_size=200,
        elite_size=30,
        local_search_steps=50,
        F_min=0.4,
        F_max=0.9,
        Cr_min=0.3,
        Cr_max=0.8,
        perturbation_intensity=0.02,
        perturbation_decay=0.98,
        alpha=0.5,
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
        self.perturbation_intensity = perturbation_intensity
        self.perturbation_decay = perturbation_decay
        self.alpha = alpha  # Weight for combining strategies

    def local_search(self, individual, func):
        best_local = individual.copy()
        best_fitness = func(individual)
        for _ in range(self.local_search_steps):
            perturbation = np.random.uniform(
                -self.perturbation_intensity, self.perturbation_intensity, len(individual)
            )
            candidate = individual + perturbation
            candidate = np.clip(candidate, self.bounds[0], self.bounds[1])
            candidate_fitness = func(candidate)
            if candidate_fitness < best_fitness:
                best_local = candidate
                best_fitness = candidate_fitness
        return best_local, best_fitness

    def differential_mutation(self, population, F, current_idx):
        """Perform differential mutation."""
        pop_size, dim = population.shape
        indices = [idx for idx in range(pop_size) if idx != current_idx]
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])
        return mutant

    def quantum_jolt(self, individual, intensity):
        """Apply quantum-inspired jolt to an individual."""
        jolt = np.random.uniform(-intensity, intensity, len(individual))
        jolted_individual = np.clip(individual + jolt, self.bounds[0], self.bounds[1])
        return jolted_individual

    def entropy_based_selection(self, fitness):
        """Select individuals based on entropy measure to maintain diversity."""
        probabilities = fitness / np.sum(fitness)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        if entropy < np.log(len(fitness)) / 2:
            selected_indices = np.argsort(fitness)[: self.elite_size]
        else:
            selected_indices = np.random.choice(len(fitness), self.elite_size, replace=False)
        return selected_indices

    def multi_population_management(self, populations, fitnesses):
        """Manage multiple sub-populations to enhance exploration and exploitation."""
        all_individuals = np.vstack(populations)
        all_fitnesses = np.hstack(fitnesses)
        best_idx = np.argmin(all_fitnesses)
        overall_best = all_individuals[best_idx]
        overall_best_fitness = all_fitnesses[best_idx]

        for i in range(len(populations)):
            if overall_best_fitness < np.min(fitnesses[i]):
                worst_idx = np.argmax(fitnesses[i])
                populations[i][worst_idx] = overall_best
                fitnesses[i][worst_idx] = overall_best_fitness

        return populations, fitnesses

    def ensemble_optimization(self, func):
        """Ensemble of multiple strategies to enhance exploration and exploitation."""
        strategies = [self.differential_mutation, self.quantum_jolt]
        self.f_opt = np.Inf
        self.x_opt = None

        sub_population_size = self.population_size // 2
        populations = [
            np.random.uniform(self.bounds[0], self.bounds[1], (sub_population_size, self.dim)),
            np.random.uniform(self.bounds[0], self.bounds[1], (sub_population_size, self.dim)),
        ]

        fitnesses = [np.array([func(ind) for ind in pop]) for pop in populations]
        evaluations = 2 * sub_population_size

        best_idx = np.argmin([np.min(fit) for fit in fitnesses])
        self.x_opt = populations[best_idx][np.argmin(fitnesses[best_idx])]
        self.f_opt = np.min([np.min(fit) for fit in fitnesses])

        F, Cr = 0.7, 0.8
        while evaluations < self.budget:
            for k in range(2):
                new_population = np.zeros_like(populations[k])
                success_count = 0
                for i in range(sub_population_size):
                    selected_strategy = np.random.choice(strategies)
                    if selected_strategy == self.differential_mutation:
                        mutant = self.differential_mutation(populations[k], F, i)
                    else:
                        intensity = self.perturbation_intensity * (1 - success_count / sub_population_size)
                        mutant = self.quantum_jolt(populations[k][i], intensity)

                    cross_points = np.random.rand(self.dim) < Cr
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True

                    trial = np.where(cross_points, mutant, populations[k][i])
                    trial_fitness = func(trial)
                    evaluations += 1

                    if trial_fitness < fitnesses[k][i]:
                        new_population[i] = trial
                        fitnesses[k][i] = trial_fitness
                        success_count += 1
                    else:
                        new_population[i] = populations[k][i]

                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial

                populations[k] = new_population

                elite_indices = self.entropy_based_selection(fitnesses[k])
                elite_population = populations[k][elite_indices]
                elite_fitness = fitnesses[k][elite_indices]

                # Performing local search on the elite population
                for j in range(self.elite_size):
                    elite_population[j], elite_fitness[j] = self.local_search(elite_population[j], func)
                    if elite_fitness[j] < self.f_opt:
                        self.f_opt = elite_fitness[j]
                        self.x_opt = elite_population[j]

                worst_indices = np.argsort(fitnesses[k])[-self.elite_size :]
                for idx in worst_indices:
                    if evaluations >= self.budget:
                        break
                    elite_idx = np.random.choice(elite_indices)
                    worst_idx = idx

                    difference_vector = populations[k][elite_idx] - populations[k][worst_idx]
                    new_candidate = populations[k][worst_idx] + np.random.rand() * difference_vector
                    new_candidate = np.clip(new_candidate, self.bounds[0], self.bounds[1])
                    new_candidate_fitness = func(new_candidate)
                    evaluations += 1

                    if new_candidate_fitness < fitnesses[k][worst_idx]:
                        populations[k][worst_idx] = new_candidate
                        fitnesses[k][worst_idx] = new_candidate_fitness
                        if new_candidate_fitness < self.f_opt:
                            self.f_opt = new_candidate_fitness
                            self.x_opt = new_candidate

                success_rate = success_count / sub_population_size
                if success_rate > 0.2:
                    F = min(self.F_max, F + 0.05 * (success_rate - 0.2))
                    Cr = max(self.Cr_min, Cr - 0.05 * (0.2 - success_rate))
                else:
                    F = max(self.F_min, F - 0.05 * (0.2 - success_rate))
                    Cr = min(self.Cr_max, Cr + 0.05 * (success_rate - 0.2))

            populations, fitnesses = self.multi_population_management(populations, fitnesses)
            self.perturbation_intensity *= self.perturbation_decay

        return self.f_opt, self.x_opt

    def __call__(self, func):
        return self.ensemble_optimization(func)
