import numpy as np


class AVDE:
    def __init__(self, budget, population_size=100, F_base=0.5, CR_init=0.7, adapt_F=0.02, adapt_CR=0.01):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.F_base = F_base  # Base differential weight
        self.CR_init = CR_init  # Initial crossover probability
        self.adapt_F = adapt_F  # Rate of adaptation for F
        self.adapt_CR = adapt_CR  # Rate of adaptation for CR

    def __call__(self, func):
        # Initialize population and velocities
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        velocities = np.zeros_like(population)
        fitness = np.array([func(ind) for ind in population])
        num_evals = self.population_size

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        # Main evolutionary loop
        while num_evals < self.budget:
            # Update differential weight (F) and crossover probability (CR) dynamically
            Fs = np.clip(np.random.normal(self.F_base, self.adapt_F, self.population_size), 0.1, 1.0)
            CRs = np.clip(np.random.normal(self.CR_init, self.adapt_CR, self.population_size), 0.1, 1.0)

            for i in range(self.population_size):
                # Mutation
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + Fs[i] * (population[b] - population[c])
                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover
                trial = np.where(np.random.rand(self.dimension) < CRs[i], mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                num_evals += 1
                if trial_fitness < fitness[i]:
                    velocities[i] = trial - population[i]
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update best solution found
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()

                if num_evals >= self.budget:
                    break

            # Adjust F_base and CR_init based on successful strategies
            successful_Fs = Fs[fitness < np.array([func(ind) for ind in population])]
            successful_CRs = CRs[fitness < np.array([func(ind) for ind in population])]
            if successful_Fs.size > 0:
                self.F_base = np.mean(successful_Fs)
            if successful_CRs.size > 0:
                self.CR_init = np.mean(successful_CRs)

        return best_fitness, best_individual
