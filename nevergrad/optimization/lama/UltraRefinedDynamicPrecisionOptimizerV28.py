import numpy as np


class UltraRefinedDynamicPrecisionOptimizerV28:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5
        self.lb = -5.0  # Lower bound of the search space
        self.ub = 5.0  # Upper bound of the search space

    def __call__(self, func):
        # Adjusted temperature and cooling parameters for enhanced exploration and exploitation
        T = 1.2  # Starting temperature adjusted for broader initial exploration
        T_min = 0.0003  # Lower minimum temperature to enable thorough late-stage search
        alpha = 0.91  # Fine-tuned cooling rate

        # Mutation and crossover parameters optimized for diversity and convergence
        F = 0.77  # Mutation factor adjusted for a balanced exploration-exploitation
        CR = 0.89  # Crossover probability fine-tuned for maintaining genetic diversity

        population_size = 85  # Population size adjusted for effective search
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Dynamic mutation strategy with exponential decay and sigmoid-based modulation
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                dynamic_F = (
                    F * np.exp(-0.1 * T) * (0.85 + 0.15 * np.tanh(5 * (evaluation_count / self.budget - 0.5)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                if trial_fitness < fitness[i] or np.random.rand() < np.exp(
                    -(trial_fitness - fitness[i]) / (T * (1 + 0.05 * np.abs(trial_fitness - fitness[i])))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Adaptive cooling strategy adjusted with sinusoidal modulation for agile response
            adaptive_cooling = alpha - 0.007 * np.sin(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
