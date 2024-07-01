import numpy as np


class UltraOptimizedDynamicPrecisionOptimizerV18:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality remains fixed at 5
        self.lb = -5.0  # Lower bound of the search space
        self.ub = 5.0  # Upper bound of the search space

    def __call__(self, func):
        # Initialize the temperature and cooling strategy with refined parameters based on previous findings
        T = 1.2  # Slightly higher initial temperature for a more aggressive global search at the beginning
        T_min = 0.0003  # Lower minimum temperature to allow very precise exploration in the later stages
        alpha = 0.95  # Slower cooling rate to sustain the search process over a longer period

        # Mutation and crossover parameters further refined for enhanced performance
        F = 0.77  # Mutation factor adjusted for an optimal balance of exploration and exploitation
        CR = 0.89  # Crossover probability finely tuned to encourage better integration of good traits

        population_size = 90  # Adjusted population size to optimize computational resources and diversity
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Implementing an enhanced dynamic mutation strategy with adaptive sigmoid function
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                # Dynamic mutation rate adapting with a sophisticated sigmoid-based model
                dynamic_F = (
                    F * np.exp(-0.05 * T) * (0.8 + 0.2 * np.tanh(4 * (evaluation_count / self.budget - 0.5)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Advanced acceptance criteria with temperature-sensitive decision making
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.05 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Advanced cooling strategy integrating a periodic modulation for more nuanced temperature control
            adaptive_cooling = alpha - 0.007 * np.cos(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
