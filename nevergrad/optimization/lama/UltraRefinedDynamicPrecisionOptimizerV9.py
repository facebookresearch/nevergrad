import numpy as np


class UltraRefinedDynamicPrecisionOptimizerV9:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initialize temperature and adaptive cooling parameters for refined control
        T = 1.2  # Increased initial temperature for broader initial exploration
        T_min = 0.0003  # Lower minimum temperature for deeper late-stage exploration
        alpha = 0.94  # Slower cooling rate to maintain search capability longer

        # Fine-tuned mutation and crossover parameters for optimal exploration and convergence
        F = 0.78  # Adjusted mutation factor to balance between diversification and intensification
        CR = 0.89  # High crossover probability to ensure robust solution mixing

        population_size = 90  # Increased population for better sampling and diversity
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Dynamic mutation using a refined exponential decay and tanh function for precision control
                dynamic_F = (
                    F * np.exp(-0.05 * T) * (0.8 + 0.2 * np.tanh(4 * (evaluation_count / self.budget - 0.4)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Advanced acceptance criteria incorporate a more responsive temperature function
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.08 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Enhanced adaptive cooling strategy with sinusoidal amplitude modulation
            adaptive_cooling = alpha - 0.007 * np.cos(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
