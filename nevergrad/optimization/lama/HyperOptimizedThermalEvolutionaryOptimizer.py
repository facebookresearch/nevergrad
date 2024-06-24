import numpy as np


class HyperOptimizedThermalEvolutionaryOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initialize temperature and enhanced cooling parameters
        T = 1.2  # Starting temperature slightly higher for initial global exploration
        T_min = 0.0003  # Lower minimum temperature for prolonged fine-tuning
        alpha = 0.92  # Slower cooling rate to extend exploration

        # Optimized mutation and crossover parameters
        F_base = 0.75  # Base mutation factor for robust exploration and exploitation balance
        CR = 0.88  # Crossover probability finely tuned for better offspring quality

        population_size = 80  # Optimal population size for this budget and problem complexity
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
                # Dynamic mutation factor influenced by temperature and evaluation progress
                dynamic_F = F_base * np.exp(-0.15 * T) * (0.5 + 0.5 * np.tanh(evaluation_count / self.budget))
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Enhanced acceptance criterion incorporating both temperature and delta fitness
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.07 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Non-linear adaptive cooling strategy with periodic modulation for stagnation avoidance
            adaptive_cooling = alpha - 0.015 * np.cos(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
