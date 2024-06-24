import numpy as np


class ADEGM:
    def __init__(self, budget, population_size=50, F_init=0.5, F_end=0.8, CR=0.9):
        self.budget = budget
        self.population_size = population_size
        self.F_init = F_init  # Initial differential weight
        self.F_end = F_end  # Final differential weight for linear adaptation
        self.CR = CR  # Crossover probability
        self.dimension = 5
        self.bounds = (-5.0, 5.0)  # Search space bounds

    def __call__(self, func):
        # Initialize the population within the bounds
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        evaluations = self.population_size

        while evaluations < self.budget:
            # Adaptive F scaling based on the linear progression from initial to end value
            F_current = self.F_init + (self.F_end - self.F_init) * (evaluations / self.budget)

            for i in range(self.population_size):
                # Select three random distinct indices
                indices = np.random.choice(
                    [j for j in range(self.population_size) if j != i], 3, replace=False
                )
                x1, x2, x3 = population[indices]

                # Guided mutation: DE/current-to-best/1
                best = population[np.argmin(fitness)]
                mutant = x1 + F_current * (best - x1 + x2 - x3)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                # Binomial crossover
                trial = np.where(np.random.rand(self.dimension) < self.CR, mutant, population[i])

                # Evaluate the new candidate
                f_trial = func(trial)
                evaluations += 1

                # Selection
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt
