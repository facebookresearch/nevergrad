import numpy as np


class ADEGE:
    def __init__(
        self, budget, dimension=5, lower_bound=-5.0, upper_bound=5.0, population_size=40, F=0.8, CR=0.9
    ):
        self.budget = budget
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.F = F
        self.CR = CR

    def __call__(self, func):
        # Initialize population uniformly
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dimension)
        )
        fitness = np.array([func(individual) for individual in population])
        evaluations = self.population_size

        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]

        # Strategy adaptation coefficients
        adaptation_frequency = max(1, int(0.1 * self.budget))
        success_memory = []

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Mutation using "best/2" strategy
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c, d = np.random.choice(idxs, 4, replace=False)
                mutant = best_individual + self.F * (
                    population[a] + population[b] - population[c] - population[d]
                )
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                j_rand = np.random.randint(self.dimension)
                trial = np.array(
                    [
                        mutant[j] if np.random.rand() < self.CR or j == j_rand else population[i][j]
                        for j in range(self.dimension)
                    ]
                )

                # Selection
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    success_memory.append(1)

                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()
                else:
                    success_memory.append(0)

                # Adapt strategy parameters if enough trials have been made
                if len(success_memory) >= adaptation_frequency:
                    success_rate = np.mean(success_memory)
                    self.F = self.F * (0.85 if success_rate < 0.15 else 1.15)
                    self.CR = self.CR * (0.85 if success_rate > 0.15 else 1.15)
                    self.F = max(0.5, min(self.F, 0.95))
                    self.CR = max(0.5, min(self.CR, 0.95))
                    success_memory = []

        return best_fitness, best_individual
