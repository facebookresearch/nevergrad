import numpy as np


class UltraOptimizedDynamicPrecisionOptimizerV53:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality fixed at 5 as specified
        self.lb = -5.0  # Lower bound of the search space
        self.ub = 5.0  # Upper bound of the search space

    def __call__(self, func):
        # Further optimize temperature and cooling parameters for adaptive exploration
        T = 1.18  # Starting temperature adjusted for a more aggressive initial exploration
        T_min = 0.0003  # Lower minimum temperature to enable finer-grained late-stage optimization
        alpha = 0.90  # Cooling rate optimized for an extended and flexible search

        # Mutation and crossover parameters finely tuned for diversity and convergence
        F = 0.77  # Mutation factor adjusted to promote a better exploration-exploitation balance
        CR = 0.88  # Crossover probability adjusted to optimize genetic mixing

        population_size = 85  # Population size tuned for a balanced search diversity and efficiency
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Introduce a dynamic mutation approach with adaptive mutation factor and cooling modulation
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Dynamically adapt mutation factor using a more responsive exponential decay
                dynamic_F = (
                    F * np.exp(-0.05 * T) * (0.75 + 0.25 * np.cos(2 * np.pi * evaluation_count / self.budget))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Incorporate an improved acceptance criteria with a dynamically adjusted temperature function
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.05 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Enhanced adaptive cooling strategy with a new modulation curve for temperature adjustments
            adaptive_cooling = alpha - 0.006 * np.sin(3.5 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
