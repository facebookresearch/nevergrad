import numpy as np


class UltraRefinedDynamicPrecisionOptimizerV55:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initialize temperature and cooling parameters
        T = 1.2  # Starting temperature, adjusted for aggressive early exploration
        T_min = 0.0001  # Lower minimum temperature for deep late-stage exploration
        alpha = 0.95  # Slower cooling rate to extend the effective search phase

        # Mutation and crossover parameters are finely tuned
        F = 0.82  # Adjusted Mutation factor for a balance between exploration and exploitation
        CR = 0.84  # Modified Crossover probability to ensure diverse genetic mixing

        population_size = 85  # Optimized population size to balance diversity and convergence
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Introduce a dynamic mutation approach with sigmoid-based adaptation
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Mutation factor dynamically adapts using a logistic function refined for control
                dynamic_F = F / (1 + np.exp(-10 * (evaluation_count / self.budget - 0.5))) * (b - c)
                mutant = np.clip(a + dynamic_F, self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Refined acceptance criteria incorporate a temperature function adjusted for sensitivity
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.05 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Enhanced adaptive cooling strategy with a sinusoidal modulation for better control
            adaptive_cooling = alpha - 0.005 * np.sin(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
