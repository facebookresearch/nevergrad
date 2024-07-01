import numpy as np


class HyperRefinedAdaptivePrecisionOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initialize temperature and cooling parameters
        T = 1.2  # Starting temperature slightly higher for initial global exploration
        T_min = 0.0005  # Lower minimum temperature for deep exploitation
        alpha = 0.92  # Adjusted cooling rate for a more nuanced temperature drop

        # Optimized mutation and crossover parameters for a dynamic balance
        F_base = 0.8  # Base mutation factor for initial broad mutations
        CR_base = 0.85  # Base crossover probability for maintaining solution diversity

        population_size = 80  # Adjusted population size for better coverage
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Evolution loop with dynamically adapting mutation and crossover
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Mutation factor dynamically adjusted by temperature and exploration depth
                dynamic_F = (
                    F_base
                    * np.exp(-0.12 * T)
                    * (0.75 + 0.25 * np.sin(1.5 * np.pi * evaluation_count / self.budget))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                CR_dynamic = CR_base + 0.1 * np.cos(2 * np.pi * evaluation_count / self.budget)
                cross_points = np.random.rand(self.dim) < CR_dynamic
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Enhanced adaptive acceptance criterion based on delta fitness and temperature
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.1 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Progressive cooling strategy with adaptive modulation based on exploration depth
            adaptive_cooling = alpha - 0.015 * np.cos(2 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
