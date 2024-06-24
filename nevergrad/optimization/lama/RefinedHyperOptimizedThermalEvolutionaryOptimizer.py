import numpy as np


class RefinedHyperOptimizedThermalEvolutionaryOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initialize temperature and cooling parameters for refined control
        T = 1.1  # Optimized starting temperature for better initial exploration
        T_min = 0.0005  # Lower minimum temperature for extended fine-tuning phase
        alpha = 0.95  # Slower cooling rate to enhance exploration over iterations

        # Mutation and crossover parameters further optimized
        F_base = 0.7  # Base mutation factor adjusted for a balance between exploration and exploitation
        CR = 0.92  # Crossover probability finely tuned for better solution diversity and quality

        population_size = 85  # Adjusted population size for optimal usage of the budget
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Advanced mutation dynamics with temperature-dependent mutation scaling
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Dynamic mutation factor influences by both temperature and evaluation progress
                dynamic_F = (
                    F_base * np.exp(-0.1 * T) * (0.5 + 0.5 * np.tanh(2 * evaluation_count / self.budget))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Enhanced acceptance criterion incorporating both temperature and delta fitness
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.06 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Non-linear adaptive cooling strategy with periodic modulation to prevent stagnation
            adaptive_cooling = alpha - 0.01 * np.cos(2 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
