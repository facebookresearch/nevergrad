import numpy as np


class APDE:
    def __init__(
        self, budget, population_size=50, F_base=0.5, CR_base=0.9, adapt_rate=0.1, precision_factor=0.05
    ):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.F_base = F_base  # Base scaling factor for differential evolution
        self.CR_base = CR_base  # Base crossover rate
        self.adapt_rate = adapt_rate  # Rate of adaptation for F and CR
        self.precision_factor = precision_factor  # Adaptively adjusts mutation precision over iterations

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        num_evals = self.population_size

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        # Main loop
        while num_evals < self.budget:
            # Adapt F, CR, and precision adaptively
            Fs = np.clip(
                np.random.normal(
                    self.F_base, self.adapt_rate * np.exp(-num_evals / self.budget), self.population_size
                ),
                0.1,
                1.0,
            )
            CRs = np.clip(np.random.normal(self.CR_base, self.adapt_rate, self.population_size), 0.0, 1.0)
            precisions = self.precision_factor / np.log2(2 + num_evals / self.budget)

            # Mutation, Crossover, and Selection
            for i in range(self.population_size):
                if num_evals >= self.budget:
                    break

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + Fs[i] * (population[b] - population[c]) * precisions
                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover
                trial = np.where(np.random.rand(self.dimension) < CRs[i], mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                num_evals += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update best solution found
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()

        return best_fitness, best_individual
