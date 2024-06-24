import numpy as np


class HyperAdvancedDynamicPrecisionOptimizerV41:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is set as per the problem description
        self.lb = -5.0  # Lower bound of the search space
        self.ub = 5.0  # Upper bound of the search space

    def __call__(self, func):
        # Enhanced temperature dynamics and control parameters
        T = 1.2  # Higher initial temperature for more explorative early phases
        T_min = 0.0003  # Lower minimum temperature to allow finer exploration at late stages
        alpha = 0.95  # Slower cooling rate to extend effective search time

        # Refined mutation and crossover parameters for robust exploration and exploitation
        F = 0.78  # Slightly increased Mutation factor for more aggressive explorative behavior
        CR = 0.85  # Reduced Crossover probability to promote more individual trait preservation

        population_size = 84  # Slightly tweaked population size to enhance population dynamics
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Dynamic mutation using sigmoid and hyperbolic tangent functions for precise control
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                dynamic_F = F * (0.65 + 0.35 * np.tanh(4 * (evaluation_count / self.budget - 0.5)))
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Incorporate a dynamic acceptance criteria with enhanced sensitivity to fitness changes
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.07 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Adaptive cooling schedule with enhanced modulation
            T *= alpha - 0.009 * np.sin(3.5 * np.pi * evaluation_count / self.budget)

        return f_opt, x_opt
