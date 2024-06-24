import numpy as np


class UltraRefinedDynamicPrecisionOptimizerV37:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initialize temperature and cooling parameters
        T = 1.2  # Adjusted starting temperature to enhance initial exploratory capability
        T_min = 0.0001  # Reduced minimum temperature for deeper late-stage optimization
        alpha = 0.91  # Slower cooling rate to support prolonged search dynamics

        # Mutation and crossover parameters refined for adaptive balance
        F = 0.8  # Slightly increased Mutation factor to boost exploratory ventures
        CR = 0.88  # Increased Crossover probability to promote genetic diversity

        population_size = 90  # Increased population size for more robust sampling
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Implement dynamic mutation with combined exponential and logistic decay
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Mutation dynamically adjusted with temperature and logistic function for precise control
                dynamic_F = (
                    F
                    * np.exp(-0.05 * T)
                    * (0.65 + 0.35 * (1 / (1 + np.exp(-10 * (evaluation_count / self.budget - 0.5)))))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Enhanced acceptance criteria using a dynamic temperature function
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.05 * np.sin(np.pi * evaluation_count / self.budget)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Cooling strategy incorporating adaptive rate with sinusoidal influence
            adaptive_cooling = alpha - 0.009 * np.sin(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
