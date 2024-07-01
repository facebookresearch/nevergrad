import numpy as np


class UltraRefinedDynamicPrecisionOptimizerV35:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initialize temperature and advanced cooling parameters
        T = 1.2  # Higher starting temperature to encourage extensive initial exploration
        T_min = 0.0002  # Even lower minimum temperature for deeper exploration in the late stages
        alpha = 0.95  # Slower cooling rate to extend the search duration and improve convergence

        # Adjust mutation and crossover parameters for dynamic adaptability
        F = 0.8  # Increased Mutation factor to enhance exploratory capabilities
        CR = 0.9  # High Crossover probability to ensure better gene mixing and diversity

        population_size = 100  # Increased population size to improve coverage of the search space
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Dynamic mutation approach with advanced adaptive control
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Enhanced dynamic mutation factor with exponential decay and sigmoid modulation
                dynamic_F = (
                    F
                    * np.exp(-0.05 * T)
                    * (0.65 + 0.35 * np.tanh(5 * (evaluation_count / self.budget - 0.5)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1

                # Refined acceptance criteria incorporating temperature-based probabilistic acceptance
                if trial_fitness < fitness[i] or np.random.rand() < np.exp(-(trial_fitness - fitness[i]) / T):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Advanced adaptive cooling strategy using a sinusoidal modulation pattern
            adaptive_cooling = alpha - 0.005 * np.sin(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
