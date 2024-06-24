import numpy as np


class ADEDLR:
    def __init__(self, budget, population_size=40, CR=0.9, F=0.8):
        self.budget = budget
        self.CR = CR  # Crossover probability
        self.F = F  # Differential weight
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

        evaluations = self.population_size
        while evaluations < self.budget:
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                crossover = np.random.rand(self.dimension) < self.CR
                trial = np.where(crossover, mutant, population[i])

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                # Adaptive parameter adjustment
                if evaluations % 10 == 0:
                    improvement_rate = np.mean(fitness) - best_fitness
                    if improvement_rate < 1e-5:
                        self.F = min(self.F * 1.1, 1.0)
                        self.CR = min(self.CR * 0.9, 1.0)

                if evaluations >= self.budget:
                    break

            # Local search phase
            local_best = best_solution.copy()
            for j in range(10):
                local_trial = local_best + np.random.normal(0, 0.1, self.dimension)
                local_trial = np.clip(local_trial, self.lower_bound, self.upper_bound)
                local_fitness = func(local_trial)
                evaluations += 1

                if local_fitness < best_fitness:
                    best_solution = local_trial
                    best_fitness = local_fitness
                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
