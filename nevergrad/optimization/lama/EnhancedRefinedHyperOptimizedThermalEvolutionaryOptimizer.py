import numpy as np


class EnhancedRefinedHyperOptimizedThermalEvolutionaryOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initialize temperature and cooling parameters
        T = 1.2  # Slightly higher starting temperature for improved initial exploration
        T_min = 0.0003  # Reduced minimum temperature for more pronounced fine-tuning
        alpha = 0.92  # Adjusted cooling rate to balance exploration and convergence

        # Mutation and crossover parameters refined further
        F_base = 0.75  # Adjusted mutation factor for improved balance
        CR = 0.91  # Modified crossover probability to enhance good trait retention

        population_size = 90  # Slightly increased population size for better coverage
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Advanced mutation dynamics with temperature and progress-dependent scaling
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Adjusted dynamic mutation factor to reflect both temperature and proportional progress
                dynamic_F = (
                    F_base * np.exp(-0.1 * T) * (0.5 + 0.5 * np.sin(np.pi * evaluation_count / self.budget))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Enhanced acceptance criterion leveraging delta fitness, temperature, and a fine-tuned function
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.04 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Non-linear adaptive cooling strategy with periodic modulation adjustments
            adaptive_cooling = alpha - 0.012 * np.cos(1.5 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
