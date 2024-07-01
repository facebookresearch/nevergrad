import numpy as np


class HyperRefinedAdaptiveDynamicPrecisionOptimizerV52:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality fixed at 5
        self.lb = -5.0  # Lower bound of the search space
        self.ub = 5.0  # Upper bound of the search space

    def __call__(self, func):
        # Initialize refined temperature and adaptive cooling parameters
        T = 1.1  # Optimized initial temperature for broader exploration
        T_min = 0.0003  # Reduced minimal temperature for extended fine-tuning in late optimization stages
        alpha = 0.93  # Slightly adjusted cooling rate to enhance prolonged search effectiveness

        # Mutation and crossover parameters fine-tuned for diversity and convergence balance
        F = 0.8  # Moderately high mutation factor to encourage exploratory mutations
        CR = 0.88  # High crossover probability to ensure effective information exchange

        population_size = 85  # Population size optimized based on prior performance reviews
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Introducing a more aggressive dynamic mutation based on exploration phase
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Dynamic mutation factor adjusted with exponential decay and sigmoid modulation
                dynamic_F = (
                    F * np.exp(-0.05 * T) * (0.6 + 0.4 * np.tanh(5 * (evaluation_count / self.budget - 0.6)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Adaptive cooling strategy enhanced with sinusoidal modulation for phase-based cooling
            adaptive_cooling = alpha - 0.009 * np.sin(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
