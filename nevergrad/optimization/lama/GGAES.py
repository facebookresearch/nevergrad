import numpy as np


class GGAES:
    def __init__(self, budget, population_size=150, F_base=0.8, CR_base=0.7, adapt_rate=0.2):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.F_base = F_base  # Base differential weight
        self.CR_base = CR_base  # Base crossover probability
        self.adapt_rate = adapt_rate  # Adaptation rate for parameters

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
            # Adaptive F and CR
            Fs = np.clip(np.random.normal(self.F_base, self.adapt_rate, self.population_size), 0.2, 1.0)
            CRs = np.clip(np.random.normal(self.CR_base, self.adapt_rate, self.population_size), 0.2, 1.0)

            gradients = np.zeros_like(population)
            gradient_step = 0.1 * (self.ub - self.lb)

            # Estimate gradients for population
            for i in range(self.population_size):
                for d in range(self.dimension):
                    plus = population[i].copy()
                    minus = population[i].copy()
                    plus[d] += gradient_step
                    minus[d] -= gradient_step
                    grad_fitness_plus = func(plus)
                    grad_fitness_minus = func(minus)
                    gradients[i, d] = (grad_fitness_plus - grad_fitness_minus) / (2 * gradient_step)
                    num_evals += 2
                    if num_evals >= self.budget:
                        return best_fitness, best_individual

            # Evolutionary operations
            for i in range(self.population_size):
                if num_evals >= self.budget:
                    break
                # Mutation with gradient guidance
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b = np.random.choice(indices, 2, replace=False)
                mutant = population[i] + Fs[i] * (
                    best_individual - population[i] + population[a] - population[b] - gradients[i]
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
