import numpy as np


class PrecisionEvolutionaryThermalOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality set from the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initialize advanced parameters for exploration and exploitation
        T = 1.5  # Higher initial temperature for broader exploration at the start
        T_min = 0.001  # Lower minimum temperature for extended fine-tuning
        alpha = 0.98  # Slower cooling to allow more thorough examination at each temperature level

        # Updated mutation and crossover parameters for higher diversity
        F = 0.8  # Increased mutation factor to encourage more pronounced variations
        CR = 0.9  # Higher crossover probability to increase gene mixing

        population_size = 100  # Increased population size for better initial search space coverage
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Dynamic mutation and adaptive simulated annealing acceptance
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Mutation factor dynamically adjusted based on temperature and progression
                dynamic_F = (
                    F * (0.5 + 0.5 * np.cos(np.pi * T)) * (0.5 + 0.5 * (evaluation_count / self.budget))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Simulated annealing acceptance criterion adjusted to account for temperature and fitness improvement
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.1 * np.log(1 + np.abs(delta_fitness))))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Cooling rate adaptation considering optimization stage
            adaptive_cooling = alpha - 0.01 * (evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
