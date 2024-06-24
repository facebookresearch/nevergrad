import numpy as np


class AdvancedOptimalHybridDifferentialAnnealingOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # The problem dimensionality is fixed at 5 as per the description
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Fine-tuned temperature parameters for simulated annealing
        T = 1.0
        T_min = 0.001  # Lower minimum temperature for precise late-stage optimization
        alpha = 0.95  # Cooling rate, slower to allow deeper exploration

        # Parameters for differential evolution
        F = 0.8  # Mutation factor adjusted for more aggressive mutations
        CR = 0.85  # Crossover probability to ensure good mixing of attributes

        # Population size adjusted for a balanced exploration-exploitation trade-off
        population_size = 75
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                # Selecting indices for mutation
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                # Dynamic mutation factor based on temperature to adjust aggressiveness
                dynamic_F = F * (1 - 0.1 * np.tanh(T))
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Acceptance based on simulated annealing principle with a temperature-dependent probability
                if delta_fitness < 0 or np.random.rand() < np.exp(-delta_fitness / T):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Adaptive cooling rate based on progress to ensure adequate time for local exploitation
            adaptive_cooling = alpha ** (1 - 0.3 * (evaluation_count / self.budget))
            T *= adaptive_cooling

        return f_opt, x_opt
