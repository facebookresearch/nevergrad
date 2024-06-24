import numpy as np


class DADERC:
    def __init__(self, budget, population_size=100, F_base=0.5, CR_base=0.9, adapt_rate=0.1):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.F_base = F_base  # Base differential weight
        self.CR_base = CR_base  # Base crossover probability
        self.adapt_rate = adapt_rate  # Rate of adaptation for parameters

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
            # Adaptive mutation strategy controlled by a dual mechanism
            Fs = np.random.normal(self.F_base, 0.1, self.population_size)
            CRs = np.random.normal(self.CR_base, 0.1, self.population_size)

            for i in range(self.population_size):
                # Ensure parameter bounds
                F = np.clip(Fs[i], 0.1, 2)
                CR = np.clip(CRs[i], 0, 1)

                # Mutation
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + F * (population[b] - population[c])
                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover
                trial = np.where(np.random.rand(self.dimension) < CR, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                num_evals += 1
                if num_evals >= self.budget:
                    break

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update best solution found
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()

            # Adapt F_base and CR_base using feedback from the population improvement
            successful_Fs = Fs[fitness < np.array([func(ind) for ind in population])]
            successful_CRs = CRs[fitness < np.array([func(ind) for ind in population])]
            if len(successful_Fs) > 0:
                self.F_base += self.adapt_rate * (np.mean(successful_Fs) - self.F_base)
            if len(successful_CRs) > 0:
                self.CR_base += self.adapt_rate * (np.mean(successful_CRs) - self.CR_base)

        return best_fitness, best_individual
