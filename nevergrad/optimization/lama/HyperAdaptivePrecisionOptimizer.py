import numpy as np


class HyperAdaptivePrecisionOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimension set as per problem description
        self.lb = -5.0  # Lower boundary of the search space
        self.ub = 5.0  # Upper boundary of the search space

    def __call__(self, func):
        # Initialize temperature and cooling parameters
        T = 1.2  # Starting temperature for vigorous initial exploration
        T_min = 0.0001  # Minimum threshold temperature for precision exploitation
        alpha = 0.95  # Cooling rate for a strong balance exploration-exploitation

        # Mutation and crossover parameters optimized further
        F_base = 0.8  # Base mutation factor
        CR = 0.95  # High crossover probability to maintain strong diversity

        population_size = 90  # Adjusted population size for a more thorough search
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Dynamic mutation influenced by temperature and time
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Modify mutation factor dynamically based on temperature and remaining budget
                dynamic_F = (
                    F_base * np.exp(-0.15 * T) * (0.7 + 0.3 * np.cos(np.pi * evaluation_count / self.budget))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Enhanced acceptance criterion incorporating delta_fitness and temperature
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.1 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Progressive adaptive cooling strategy with non-linear modulation
            adaptive_cooling = alpha - 0.02 * np.sin(2.5 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
