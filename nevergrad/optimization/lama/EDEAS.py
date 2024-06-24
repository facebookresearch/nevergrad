import numpy as np


class EDEAS:
    def __init__(self, budget, population_size=60, F_min=0.5, F_max=0.8, CR=0.9):
        self.budget = budget
        self.population_size = population_size
        self.F_min = F_min  # Minimum scale factor
        self.F_max = F_max  # Maximum scale factor
        self.CR = CR  # Crossover probability
        self.dimension = 5
        self.bounds = (-5.0, 5.0)  # Search space bounds

    def __call__(self, func):
        # Initialize population randomly within bounds
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        evaluations = self.population_size

        # Adaptive F based on linear scaling
        F_scale = np.linspace(self.F_min, self.F_max, self.population_size)

        while evaluations < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(
                    [j for j in range(self.population_size) if j != i], 3, replace=False
                )
                x1, x2, x3 = population[indices]

                # Mutant vector creation with adaptive F
                F_i = F_scale[np.argsort(fitness)[i]]  # Scale F based on fitness rank
                mutant = x1 + F_i * (x2 - x3)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                # Crossover
                trial = np.where(np.random.rand(self.dimension) < self.CR, mutant, population[i])

                # Selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt
