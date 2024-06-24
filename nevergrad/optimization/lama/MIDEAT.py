import numpy as np


class MIDEAT:
    def __init__(self, budget, population_size=30, CR=0.9, F=0.8, momentum_factor=0.5):
        self.budget = budget
        self.CR = CR  # Crossover probability
        self.F = F  # Differential weight
        self.population_size = population_size
        self.dimension = 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.momentum_factor = momentum_factor

    def __call__(self, func):
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dimension)
        )
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        velocity = np.zeros((self.population_size, self.dimension))

        evaluations = self.population_size
        while evaluations < self.budget:
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                crossover = np.random.rand(self.dimension) < self.CR
                trial = np.where(crossover, mutant, population[i])

                velocity[i] = self.momentum_factor * velocity[i] + (1 - self.momentum_factor) * (
                    trial - population[i]
                )
                trial = np.clip(population[i] + velocity[i], self.lower_bound, self.upper_bound)

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
