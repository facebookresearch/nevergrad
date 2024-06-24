import numpy as np


class OptimalPrecisionEvolutionaryThermalOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Modified initial temperature and advanced cooling rate for better exploration-exploitation balance
        T = 1.0  # Reduced initial temperature for more controlled exploration
        T_min = 0.002  # Lower minimum temperature for extended fine-tuning at late stages
        alpha = 0.92  # Reduced cooling rate to extend the exploration phase

        # Refined mutation and crossover parameters based on prior performance insights
        F = 0.8  # Increased mutation factor for bolder search moves
        CR = 0.9  # Higher crossover probability to enhance solution variability

        population_size = 100  # Increased population size for broader initial coverage
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Integrate dynamic mutational strategies and adaptive temperature-based acceptance conditions
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                dynamic_F = (
                    F * np.cos(np.pi * T / 2) * (0.7 + 0.3 * np.cos(np.pi * evaluation_count / self.budget))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Advanced acceptance criterion incorporating thermal influence and relative fitness improvement
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.05 * np.log(1 + np.abs(delta_fitness))))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Implement a progressive cooling strategy that adapts more precisely based on optimization depth
            adaptive_cooling = alpha - 0.01 * np.tanh(evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
