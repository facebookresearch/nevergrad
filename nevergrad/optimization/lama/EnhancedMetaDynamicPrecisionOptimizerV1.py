import numpy as np


class EnhancedMetaDynamicPrecisionOptimizerV1:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Initialize temperature and annealing parameters
        T = 1.2  # Optimized initial temperature to encourage exploration
        T_min = 0.0001  # Very low minimum temperature for deep exploration in late stages
        alpha = 0.93  # Optimal cooling rate to maintain a balance between exploration and exploitation

        # Mutation and crossover parameters adjusted for optimal search dynamics
        F = 0.78  # Mutation factor adjusted for aggressive exploration
        CR = 0.88  # High crossover probability to ensure diversity in solutions

        population_size = 85  # Optimized population size for effective coverage
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Integrate adaptive mutation strategy with enhanced dynamic control
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Dynamic mutation factor adapts based on the sigmoid function for refined control
                dynamic_F = (
                    F * np.exp(-0.07 * T) * (0.8 + 0.2 * np.tanh(3 * (evaluation_count / self.budget - 0.5)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Improved acceptance criteria incorporating a dynamic temperature-dependent function
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.08 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Adaptive cooling strategy enhanced with sinusoidal modulation for longer effective search
            adaptive_cooling = alpha - 0.005 * np.sin(3.5 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
