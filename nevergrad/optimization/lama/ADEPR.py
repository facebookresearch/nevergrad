import numpy as np


class ADEPR:
    def __init__(self, budget, population_size=100, F_base=0.5, CR_base=0.8):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Differential weight
        self.CR_base = CR_base  # Crossover probability
        self.dimension = 5
        self.bounds = (-5.0, 5.0)  # Search space bounds

    def __call__(self, func):
        # Initialize the population randomly within the bounds
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        evaluations = self.population_size

        # Main optimization loop
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation: DE/rand/1 scheme
                indices = np.random.choice(
                    [j for j in range(self.population_size) if j != i], 3, replace=False
                )
                x1, x2, x3 = population[indices]
                mutant = np.clip(x1 + self.F_base * (x2 - x3), self.bounds[0], self.bounds[1])

                # Crossover: Binomial with adaptive CR
                trial = np.copy(population[i])
                cr = self.CR_base if np.random.rand() < 0.1 else np.random.normal(self.CR_base, 0.1)
                cr = np.clip(cr, 0, 1)
                crossover = np.random.rand(self.dimension) < cr
                trial[crossover] = mutant[crossover]

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
