import numpy as np


class UltraOptimizedPrecisionAdaptiveEvolutionaryOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimension set as per problem description
        self.lb = -5.0  # Lower boundary of the search space
        self.ub = 5.0  # Upper boundary of the search space

    def __call__(self, func):
        # Initialize thermal and evolutionary parameters
        T = 1.2  # Starting temperature slightly higher for vigorous initial exploration
        T_min = 0.001  # Lower threshold of temperature for fine-grained exploration at later stages
        alpha = 0.91  # Cooling rate selected for a balanced exploration-exploitation trade-off

        # Mutation and crossover parameters refined for optimal performance
        F_base = 0.75  # Mutation factor for controlling differential variation
        CR = 0.92  # High crossover probability to maintain diversity

        population_size = 80  # Optimized population size for the budget
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Advanced mutation dynamics with temperature and progress-dependent adaptation
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Adaptive mutation factor influenced by both temperature and iteration progress
                dynamic_F = F_base * np.exp(-0.12 * T) * (0.5 + 0.5 * np.tanh(evaluation_count / self.budget))
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Enhanced acceptance criterion that considers both delta_fitness and temperature
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.045 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Adaptive cooling strategy enhanced with a non-linear modulation (sinusoidal adjustments)
            adaptive_cooling = alpha - 0.015 * np.sin(1.8 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
