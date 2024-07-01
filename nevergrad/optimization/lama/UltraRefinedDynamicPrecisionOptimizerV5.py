import numpy as np


class UltraRefinedDynamicPrecisionOptimizerV5:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initialize temperature and adaptive cooling parameters for enhanced control
        T = 1.2  # Slightly increased initial temperature for broader exploration at start
        T_min = 0.0003  # Lowered minimum temperature for prolonged fine exploration
        alpha = 0.93  # Slower cooling rate to retain effective search capability longer

        # Finely tuned mutation and crossover parameters for optimal diversity and convergence
        F = 0.77  # Adjusted mutation factor to balance between global and local search
        CR = 0.88  # Increased crossover probability to ensure robust mixing of solutions

        population_size = 85  # Slightly larger population for better sampling of the search space
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Dynamic mutation using a refined sigmoid adaptation strategy
                dynamic_F = (
                    F
                    * np.exp(-0.06 * T)
                    * (0.75 + 0.25 * np.tanh(3.5 * (evaluation_count / self.budget - 0.45)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Advanced acceptance criteria incorporating a more responsive temperature function
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.07 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Further enhanced cooling strategy with sinusoidal amplitude modulation
            adaptive_cooling = alpha - 0.009 * np.cos(2.6 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
