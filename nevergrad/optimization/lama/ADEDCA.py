import numpy as np


class ADEDCA:
    def __init__(self, budget, population_size=150, F_base=0.8, CR_init=0.5, adapt_F=0.1, adapt_CR=0.05):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.F_base = F_base  # Initial base differential weight
        self.CR_init = CR_init  # Initial crossover probability
        self.adapt_F = adapt_F  # Rate of adaptation for F
        self.adapt_CR = adapt_CR  # Rate of adaptation for CR

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        num_evals = self.population_size

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        # Main evolutionary loop
        while num_evals < self.budget:
            # Update F and CR for each generation to adapt to landscape
            Fs = np.clip(np.random.normal(self.F_base, self.adapt_F, self.population_size), 0.4, 1.2)
            CRs = np.clip(np.random.normal(self.CR_init, self.adapt_CR, self.population_size), 0, 1)

            for i in range(self.population_size):
                # Mutation using current F values
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + Fs[i] * (population[b] - population[c])
                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover using current CR values
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

                if num_evals >= self.budget:
                    break

            # Dynamic adaptation of CR and F
            self.CR_init = (
                np.mean(CRs[fitness < np.array([func(ind) for ind in population])])
                if len(CRs[fitness < fitness]) > 0
                else self.CR_init
            )
            self.F_base = (
                np.mean(Fs[fitness < np.array([func(ind) for ind in population])])
                if len(Fs[fitness < fitness]) > 0
                else self.F_base
            )

        return best_fitness, best_individual
