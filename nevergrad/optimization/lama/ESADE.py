import numpy as np


class ESADE:
    def __init__(self, budget, population_size=50, F_base=0.5, CR_base=0.8):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Base differential weight
        self.CR_base = CR_base  # Base crossover probability
        self.dimension = 5
        self.bounds = (-5.0, 5.0)  # As problem specification

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])

        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        evaluations = self.population_size

        # Initialize adaptive parameters
        F = np.full(self.population_size, self.F_base)
        CR = np.full(self.population_size, self.CR_base)

        # Introduce memory for F and CR to enhance strategy adaptation
        F_memory = np.zeros(self.population_size)
        CR_memory = np.zeros(self.population_size)

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Adaptive mutation strategy based on fitness and memory
                if fitness[i] < np.median(fitness):
                    F[i] = min(F[i] * 1.1, 1)  # Limit F to a maximum of 1
                    F_memory[i] += 1
                else:
                    F[i] = max(F[i] * 0.9, 0.1)  # Ensure F does not fall below 0.1
                    F_memory[i] /= 1.1

                # Mutation and crossover
                idxs = [idx for idx in range(self.population_size) if idx != i]
                best_idx = np.argmin(fitness[idxs])
                a, b, c = (
                    population[idxs[best_idx]],
                    population[np.random.choice(idxs)],
                    population[np.random.choice(idxs)],
                )
                mutant = np.clip(a + F[i] * (b - c), self.bounds[0], self.bounds[1])

                cross_points = np.random.rand(self.dimension) < CR[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dimension)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate trial solution
                f_trial = func(trial)
                evaluations += 1

                # Selection: Accept the trial if it is better
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    CR[i] = min(CR[i] * 1.05, 1)  # Limit CR to a maximum of 1
                    CR_memory[i] += 1

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    CR[i] = max(CR[i] * 0.95, 0.1)  # Ensure CR does not drop below 0.1
                    CR_memory[i] /= 1.1

                # Dynamically adjust strategy based on memory
                if F_memory[i] > 3 or CR_memory[i] > 3:
                    F[i] = self.F_base
                    CR[i] = self.CR_base
                    F_memory[i] = 0
                    CR_memory[i] = 0

                # Check if budget is exhausted
                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt
