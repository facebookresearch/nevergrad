import numpy as np


class HyperOptimizedDynamicPrecisionOptimizerV57:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Initialize temperature and cooling parameters
        T = 1.10  # Initial temperature for substantial exploration upfront
        T_min = 0.0001  # Lower minimum temperature to allow for fine-tuned exploitation
        alpha = 0.93  # Cooling rate slightly reduced to extend the effective search phase

        # Mutation and crossover parameters refined
        F = 0.79  # Mutation factor adjusted for a balance between exploration and exploitation
        CR = 0.88  # Crossover probability adjusted for maintaining genetic diversity

        population_size = 100  # Increased population size for better coverage
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Dynamic mutation strategy with sigmoid-based adaptive control
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                dynamic_F = (
                    F * np.exp(-0.06 * T) * (0.8 + 0.2 * np.tanh(4 * (evaluation_count / self.budget - 0.5)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Improved acceptance criteria with a more refined temperature function
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.05 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Enhanced adaptive cooling strategy with a more nuanced modulation
            adaptive_cooling = alpha - 0.007 * np.sin(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
