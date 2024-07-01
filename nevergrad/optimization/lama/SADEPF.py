import numpy as np


class SADEPF:
    def __init__(self, budget, population_size=50, F_init=0.5, CR_init=0.9):
        self.budget = budget
        self.CR_init = CR_init
        self.F_init = F_init
        self.population_size = population_size
        self.dimension = 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dimension)
        )
        fitness = np.array([func(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        F = self.F_init
        CR = self.CR_init
        success_memory = []

        evaluations = self.population_size
        while evaluations < self.budget:
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = population[a] + F * (population[b] - population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                cross_points = np.random.rand(self.dimension) < CR
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness
                    success_memory.append(1)
                else:
                    success_memory.append(0)

                if evaluations >= self.budget:
                    break

            # Adaptive Feedback Mechanism
            if len(success_memory) > 20:  # Using the last 20 steps to calculate success rate
                success_rate = np.mean(success_memory[-20:])
                F = np.clip(F + 0.1 * (success_rate - 0.5), 0.1, 1.0)
                CR = np.clip(CR + 0.1 * (success_rate - 0.5), 0.1, 1.0)

        return best_fitness, best_solution
