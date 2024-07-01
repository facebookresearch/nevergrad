import numpy as np


class UltraPreciseDynamicOptimizerV26:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Given dimensionality of the problem
        self.lb = -5.0  # Lower bound of the search space
        self.ub = 5.0  # Upper bound of the search space

    def __call__(self, func):
        # Initialize temperature and cooling parameters
        T = 1.15  # Starting temperature, adjusted for broader initial exploration
        T_min = 0.0002  # Lower minimum temperature for more thorough late-stage exploration
        alpha = 0.90  # Cooling rate, optimized for extended search phases

        # Mutation and crossover parameters are finely tuned
        F = 0.78  # Mutation factor, adjusted for optimal exploration-exploitation balance
        CR = 0.88  # Crossover probability, modified for increased genetic diversity

        population_size = 85  # Adjusted population size for efficient search space scanning
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Implement a dynamic mutation approach with an adaptive exponential modulation
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Mutation factor dynamically adapts using an exponential decay and hyperbolic tangent for refined control
                dynamic_F = (
                    F
                    * (1 - np.exp(-0.1 * T))
                    * (0.65 + 0.35 * np.tanh(3 * (evaluation_count / self.budget - 0.5)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Adapted acceptance criteria including a more aggressive temperature-dependent probability
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.07 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Enhanced adaptive cooling strategy with a sinusoidal modulation
            adaptive_cooling = alpha - 0.009 * np.sin(2.5 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
