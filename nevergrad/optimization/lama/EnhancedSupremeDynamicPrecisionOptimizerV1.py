import numpy as np


class EnhancedSupremeDynamicPrecisionOptimizerV1:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Initialize temperature and adaptive cooling settings
        T = 1.2  # Higher initial temperature for more aggressive initial exploration
        T_min = 0.0001  # Lower minimum temperature to enable finer search in later stages
        alpha = 0.90  # Adjusted cooling rate for an extended search period with finer transitions

        # Mutation and crossover parameters optimized for dynamic adjustments
        F = 0.78  # Mutation factor with moderate aggressiveness
        CR = 0.88  # High crossover probability to ensure substantial solution mixing

        population_size = (
            85  # Fine-tuned population size to maintain an efficient exploration-exploitation trade-off
        )
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Integrate a dynamic mutation factor influenced by both temperature and progression
                dynamic_F = (
                    F
                    * np.exp(-0.06 * T)
                    * (0.75 + 0.25 * np.tanh(3.5 * (evaluation_count / self.budget - 0.5)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Enhance acceptance criteria with a more dynamic temperature-dependent probability
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.05 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Advanced adaptive cooling with a modulation that adapts to the search phase
            adaptive_cooling = alpha - 0.007 * np.sin(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
