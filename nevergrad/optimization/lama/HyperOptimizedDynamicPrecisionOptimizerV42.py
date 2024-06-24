import numpy as np


class HyperOptimizedDynamicPrecisionOptimizerV42:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is set as per the problem description
        self.lb = -5.0  # Lower bound of the search space
        self.ub = 5.0  # Upper bound of the search space

    def __call__(self, func):
        # Initialize temperature and cooling parameters for enhanced dynamic control
        T = 1.2  # Increased starting temperature for extensive initial exploration
        T_min = 0.0004  # Lower minimum temperature for deeper exploration in later stages
        alpha = 0.93  # Optimized cooling rate to balance exploration and convergence

        # Refined mutation and crossover parameters for better performance
        F = 0.77  # Mutation factor adjusted for balanced exploration and exploitation
        CR = 0.88  # Fine-tuned crossover probability to maintain diversity and solution quality

        population_size = 85  # Adjusted population size for optimal exploration and exploitation balance
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Introduce a dynamic mutation strategy with enhanced control precision
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                # Dynamic mutation factor adapting with a sigmoid function for precise control
                dynamic_F = F * (0.72 + 0.28 * np.tanh(4 * (evaluation_count / self.budget - 0.5)))
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Enhanced acceptance criteria with a temperature-dependent function
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.065 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Adaptive cooling schedule enhanced with sinusoidal modulation
            T *= alpha - 0.01 * np.sin(3 * np.pi * evaluation_count / self.budget)

        return f_opt, x_opt
