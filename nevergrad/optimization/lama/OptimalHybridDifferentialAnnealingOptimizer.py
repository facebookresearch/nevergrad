import numpy as np


class OptimalHybridDifferentialAnnealingOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = np.full(self.dim, -5.0)  # Lower bound as per the problem description
        self.ub = np.full(self.dim, 5.0)  # Upper bound as per the problem description

    def __call__(self, func):
        # Initial temperature tailored for better control early and late in the search
        T = 1.0
        T_min = 0.0001  # Lower minimum temperature for finer control at late stages
        alpha = 0.98  # Very slow cooling rate to allow more extensive exploration

        # Mutation factor and crossover probability
        F = 0.5  # Mutation factor for balanced search intensity
        CR = 0.95  # Very high crossover probability to maintain diversity

        # Population size increased for more diverse initial solutions
        population_size = 100
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Evolutionary operations with annealing acceptance
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = a + F * (b - c)
                mutant = np.clip(mutant, self.lb, self.ub)
                trial = np.where(np.random.rand(self.dim) < CR, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1

                # Acceptance condition based on simulated annealing
                if trial_fitness < fitness[i] or np.random.rand() < np.exp(-(trial_fitness - fitness[i]) / T):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Adaptive cooling based on progress and temperature
            T *= alpha ** (1 + 0.2 * (evaluation_count / self.budget))

        return f_opt, x_opt
