import numpy as np


class EnhancedDynamicAdaptiveOptimizerV8:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initialize temperature and cooling parameters
        T = 1.1  # Optimized starting temperature for better initial exploration
        T_min = 0.0005  # Fine-tuned minimum temperature for sustained exploration at later stages
        alpha = 0.91  # Optimized cooling rate to maintain effective search duration

        # Mutation and crossover parameters are finely tuned for balance
        F = 0.76  # Fine-tuned mutation factor to optimize explorative capabilities
        CR = 0.86  # Optimized crossover probability to ensure diversity within population

        population_size = 82  # Optimally adjusted population size for effective search space sampling
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Implementing a dynamic mutation approach with sigmoid-based control for adaptation
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Adaptively tuning the mutation factor with a sigmoid function for refined control
                dynamic_F = (
                    F
                    * np.exp(-0.08 * T)
                    * (0.7 + 0.3 * np.tanh(3.3 * (evaluation_count / self.budget - 0.5)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Incorporating a more sensitive acceptance criterion that takes the current temperature into account
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.05 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Advanced adaptive cooling strategy with sinusoidal modulation for precise temperature control
            adaptive_cooling = alpha - 0.007 * np.cos(2.7 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
