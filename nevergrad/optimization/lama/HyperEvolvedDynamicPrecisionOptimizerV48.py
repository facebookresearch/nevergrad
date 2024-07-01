import numpy as np


class HyperEvolvedDynamicPrecisionOptimizerV48:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initialize temperature and cooling parameters
        T = 1.20  # Higher starting temperature for more aggressive initial exploration
        T_min = 0.0003  # Lower minimum temperature for prolonged fine-tuning
        alpha = 0.93  # Slower cooling rate to maximize exploitation during the cooling phase

        # Mutation and crossover parameters are refined further
        F = 0.79  # Slightly increased mutation factor to enhance exploratory mutations
        CR = 0.85  # Adjusted crossover probability to maintain diversity while encouraging strong features

        population_size = 90  # Increased population size to improve sampling diversity
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Initiate a dynamic mutation approach with a sigmoid-based adaptation
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Mutation factor dynamically adapts with a sigmoid function for refined control
                dynamic_F = (
                    F * np.exp(-0.05 * T) * (0.8 + 0.2 * np.tanh(4 * (evaluation_count / self.budget - 0.5)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Refined acceptance criteria incorporate a more sensitive temperature function
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.05 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Enhanced adaptive cooling strategy with sinusoidal modulation
            adaptive_cooling = alpha - 0.007 * np.cos(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
