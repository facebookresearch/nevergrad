import numpy as np


class UltraRefinedDynamicPrecisionOptimizerV38:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initialize temperature and cooling parameters
        T = 1.18  # Slightly increased starting temperature for more aggressive initial exploration
        T_min = 0.0003  # Further reduced minimum temperature for extended deep exploration
        alpha = 0.93  # Adjusted slower cooling rate to extend effective search duration

        # Refined mutation and crossover parameters
        F = 0.77  # Adjusted Mutation factor to encourage robust exploration and prevent premature convergence
        CR = 0.89  # Increased Crossover probability to promote diversity

        population_size = (
            85  # Carefully balanced population size for effective exploration and evaluation efficiency
        )
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Implement dynamic mutation with sigmoid-based adaptation for precise control
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Dynamic mutation factor incorporates a sigmoid function with refined tuning
                dynamic_F = (
                    F
                    * np.exp(-0.06 * T)
                    * (0.72 + 0.28 * np.tanh(4 * (evaluation_count / self.budget - 0.5)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1

                # Enhanced acceptance criteria using dynamically adjusted temperature function
                if trial_fitness < fitness[i] or np.random.rand() < np.exp(
                    -(trial_fitness - fitness[i])
                    / (T * (1 + 0.07 * np.sin(np.pi * evaluation_count / self.budget)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Adaptive cooling strategy incorporating a dynamic rate with sinusoidal influence
            adaptive_cooling = alpha - 0.007 * np.sin(3.5 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
