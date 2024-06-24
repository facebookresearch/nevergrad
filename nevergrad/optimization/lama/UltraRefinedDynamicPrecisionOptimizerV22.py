import numpy as np


class UltraRefinedDynamicPrecisionOptimizerV22:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality as per the problem description
        self.lb = -5.0  # Lower bound of the search space
        self.ub = 5.0  # Upper bound of the search space

    def __call__(self, func):
        # Advanced temperature and cooling dynamics
        T = 1.17  # Slightly increased starting temperature for enhanced initial exploration
        T_min = 0.00045  # Lower minimum temperature for extended fine-tuning in the late stages
        alpha = 0.91  # Slower cooling rate for prolonged exploration and exploitation period

        # Mutation and crossover parameters finely tuned for dynamic environment
        F = 0.77  # Mutation factor adjusted for a better balance of explorative and exploitative moves
        CR = 0.89  # Crossover probability optimized to sustain diversity and allow better convergence

        population_size = 82  # Adjusted population size for optimal performance
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Implementing a dynamic mutation strategy with sigmoid-based adaptation
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Dynamic mutation rate, adjusting with a sigmoid function for more refined control
                sigmoid_adjustment = 0.8 + 0.2 * np.tanh(4 * (evaluation_count / self.budget - 0.5))
                dynamic_F = F * np.exp(-0.065 * T) * sigmoid_adjustment
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Enhanced sensitivity in acceptance criteria with a refined temperature function
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.08 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Sinusoidal modulation in the cooling strategy for finer temperature control
            adaptive_cooling = alpha - 0.01 * np.sin(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
