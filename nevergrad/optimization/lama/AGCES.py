import numpy as np


class AGCES:
    def __init__(
        self, budget, population_size=100, F_base=0.5, CR_base=0.9, adapt_rate=0.1, gradient_weight=0.05
    ):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.F_base = F_base  # Base scaling factor for differential evolution
        self.CR_base = CR_base  # Base crossover rate
        self.adapt_rate = adapt_rate  # Rate of adaptation for F and CR
        self.gradient_weight = gradient_weight  # Weighting for gradient influence in mutation

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
            # Adapt F and CR adaptively
            Fs = np.clip(np.random.normal(self.F_base, self.adapt_rate, self.population_size), 0.1, 1.0)
            CRs = np.clip(np.random.normal(self.CR_base, self.adapt_rate, self.population_size), 0.1, 1.0)

            # Compute numerical gradients for population
            gradients = np.zeros_like(population)
            for i in range(self.population_size):
                for d in range(self.dimension):
                    original = population[i][d]
                    increment = 0.01 * (self.ub - self.lb)

                    population[i][d] += increment
                    f_plus = func(population[i])
                    population[i][d] = original

                    population[i][d] -= increment
                    f_minus = func(population[i])
                    population[i][d] = original

                    gradients[i][d] = (f_plus - f_minus) / (2 * increment)
                    num_evals += 2
                    if num_evals >= self.budget:
                        return best_fitness, best_individual

            # Mutation, Crossover and Selection
            for i in range(self.population_size):
                if num_evals >= self.budget:
                    break

                # Mutation with gradient guidance
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b = np.random.choice(indices, 2, replace=False)
                mutant = population[i] + Fs[i] * (
                    best_individual
                    - population[i]
                    + population[a]
                    - population[b]
                    - self.gradient_weight * gradients[i]
                )
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
