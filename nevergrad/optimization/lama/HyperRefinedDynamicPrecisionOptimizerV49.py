import numpy as np


class HyperRefinedDynamicPrecisionOptimizerV49:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5
        self.lb = -5.0  # Lower bound
        self.ub = 5.0  # Upper bound

    def __call__(self, func):
        # Initialize temperature and cooling parameters
        T = 1.1  # Initial temperature adjusted for balance between exploration and exploitation
        T_min = 0.0003  # Lower minimum temperature to allow fine-grained adjustments late in the search
        alpha = 0.95  # Reduced cooling rate to extend the exploitation phase

        # Mutation and crossover parameters for enhanced search dynamics
        F = 0.8  # Increased mutation factor to encourage diversity in the population
        CR = 0.9  # Increased crossover rate to better mix beneficial traits

        population_size = 85  # Optimal population size after tuning
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Introduce a dynamic mutation approach with a sigmoid-based adaptation
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                dynamic_F = F * (0.9 + 0.1 * np.tanh(5 * (evaluation_count / self.budget - 0.5)))
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                if trial_fitness < fitness[i]:  # Direct acceptance of better solutions
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Adaptive cooling strategy with logistic decay for temperature
            adaptive_cooling = alpha - 0.005 * (
                1 / (1 + np.exp(-10 * (evaluation_count / self.budget - 0.5)))
            )
            T *= adaptive_cooling

        return f_opt, x_opt
