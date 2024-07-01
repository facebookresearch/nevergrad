import numpy as np


class QuantumDifferentialEvolutionWithEliteGuidance:
    def __init__(self, budget=10000, population_size=30, elite_size=5):
        self.budget = budget
        self.dim = 5  # as given in the problem statement
        self.bounds = [-5.0, 5.0]
        self.population_size = population_size
        self.elite_size = elite_size

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
        base_F = 0.8
        base_Cr = 0.9
        F = base_F
        Cr = base_Cr

        stagnation_threshold = 50  # Number of generations to consider for stagnation
        stagnation_counter = 0

        # Elite tracking
        elite_indices = np.argsort(fitness)[: self.elite_size]
        elite_population = population[elite_indices]
        elite_fitness = fitness[elite_indices]

        while evaluations < self.budget:
            new_population = np.zeros_like(population)
            success_count = 0
            for i in range(self.population_size):
                # Select indices for mutation
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)

                # Quantum-inspired mutation strategy with elite guidance
                elite_idx = np.random.choice(self.elite_size)
                elite_ind = elite_population[elite_idx]

                centroid = np.mean(population[[a, b, c]], axis=0)
                mutant = centroid + F * (population[a] - population[b]) + 0.1 * (elite_ind - population[i])
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])  # Boundary handling

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
                else:
                    new_population[i] = population[i]

                # Update best solution found
                if trial_fitness < self.f_opt:
                    self.f_opt = trial_fitness
                    self.x_opt = trial
                    stagnation_counter = 0  # Reset stagnation counter
                else:
                    stagnation_counter += 1

            population = new_population

            # Update elite set
            elite_indices = np.argsort(fitness)[: self.elite_size]
            elite_population = population[elite_indices]
            elite_fitness = fitness[elite_indices]

            # Adaptive parameters based on success rate
            if success_count > self.population_size * 0.2:
                F = min(1.0, F + 0.1)
                Cr = max(0.1, Cr - 0.1)
            else:
                F = max(0.4, F - 0.1)
                Cr = min(1.0, Cr + 0.1)

            # Quantum-inspired restart mechanism
            if stagnation_counter > stagnation_threshold:
                # Re-initialize a portion of the population based on distance to the best solution
                distances = np.linalg.norm(population - self.x_opt, axis=1)
                reinit_indices = distances.argsort()[-int(self.population_size / 2) :]
                population[reinit_indices] = np.random.uniform(
                    self.bounds[0], self.bounds[1], (len(reinit_indices), self.dim)
                )
                fitness[reinit_indices] = np.array([func(ind) for ind in population[reinit_indices]])
                evaluations += len(reinit_indices)
                best_idx = np.argmin(fitness)
                self.x_opt = population[best_idx]
                self.f_opt = fitness[best_idx]
                stagnation_counter = 0

        return self.f_opt, self.x_opt
