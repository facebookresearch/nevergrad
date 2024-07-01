import numpy as np


class MetaDynamicPrecisionOptimizerV1:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Initialize temperature and annealing parameters
        T = 1.20  # Higher initial temperature to encourage initial exploration
        T_min = 0.0001  # Very low minimum temperature to allow thorough late-stage exploitation
        alpha = 0.95  # Slower cooling rate to extend the effective search phase

        # Refined mutation and crossover parameters for dynamic adaptability
        F = 0.8  # Slightly increased mutation factor for robust exploration
        CR = 0.85  # Crossover probability fine-tuned for better genetic diversity

        population_size = 90  # Optimized population size for more effective search
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Adaptive mutation strategy with enhanced dynamic control
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                dynamic_F = (
                    F * np.exp(-0.05 * T) * (0.75 + 0.25 * np.sin(5 * (evaluation_count / self.budget - 0.5)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Improved acceptance criteria with a finer temperature function
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.04 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Advanced cooling strategy including sinusoidal modulation
            adaptive_cooling = alpha - 0.006 * np.sin(3.5 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
