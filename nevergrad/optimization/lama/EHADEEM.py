import numpy as np


class EHADEEM:
    def __init__(self, budget, population_size=60, F_base=0.6, CR_base=0.9):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Initial base for differential weight
        self.CR_base = CR_base  # Initial base for crossover probability
        self.dimension = 5
        self.bounds = (-5.0, 5.0)  # Search space bounds

    def __call__(self, func):
        # Initialize the population within the bounds
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        evaluations = self.population_size

        # Initialize adaptive parameters
        F = np.full(self.population_size, self.F_base)
        CR = np.full(self.population_size, self.CR_base)
        memory = np.zeros(self.population_size)  # Memory for adaptive adjustments
        success_count = np.zeros(self.population_size)  # Count successful generations for each individual

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Select mutation indices
                idxs = np.random.choice(
                    [idx for idx in range(self.population_size) if idx != i], 3, replace=False
                )
                a, b, c = population[idxs[0]], population[idxs[1]], population[idxs[2]]

                # Mutation: DE/best/1/bin scheme using best individual for faster convergence
                best = population[np.argmin(fitness)]
                mutant = np.clip(best + F[i] * (b - c), self.bounds[0], self.bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dimension) < CR[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dimension)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluation
                f_trial = func(trial)
                evaluations += 1

                # Selection and memory update
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    memory[i] += 1  # Increment success memory
                    success_count[i] += 1
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    memory[i] -= 1  # Decay memory on failure

                # Adaptive parameter tuning based on memory
                if memory[i] > 3:
                    F[i] = min(F[i] * 1.1, 1)
                    CR[i] = min(CR[i] * 1.1, 1)
                elif memory[i] < -3:
                    F[i] = max(F[i] * 0.9, 0.1)
                    CR[i] = max(CR[i] * 0.8, 0.1)

                # Reset memory if extremes are achieved
                if memory[i] > 8 or memory[i] < -8:
                    memory[i] = 0
                    F[i] = self.F_base
                    CR[i] = self.CR_base

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt
