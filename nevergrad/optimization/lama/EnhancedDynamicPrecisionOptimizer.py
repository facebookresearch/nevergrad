import numpy as np


class EnhancedDynamicPrecisionOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initialize temperature and cooling parameters
        T = 1.2  # Starting temperature, slightly higher for more initial exploration
        T_min = 0.0003  # Lower minimum temperature threshold for longer fine-tuning
        alpha = 0.95  # Slower cooling rate, providing more opportunity for search

        # Mutation and crossover parameters further refined
        F = 0.8  # Increased mutation factor for more aggressive explorative moves
        CR = 0.87  # Adjusted crossover probability for balanced exploration and exploitation

        population_size = 90  # Increased population size for more diverse solutions
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Incorporating more dynamic mutation and adaptive temperature schedule
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                # More dynamic mutation adapting with both temperature and normalized progress
                dynamic_F = F * (
                    np.exp(-0.2 * T) + (0.5 - 0.5 * np.cos(2 * np.pi * evaluation_count / self.budget))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Advanced acceptance criterion considering the annealing process and dynamic fitness improvement
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.03 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Gradual and adaptive cooling with an additional modulation to accommodate search stages
            adaptive_cooling = alpha - 0.02 * np.sin(1.5 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
