import numpy as np


class HyperOptimizedDynamicPrecisionOptimizerV43:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem specification
        self.lb = -5.0  # Lower bound as per the problem specification
        self.ub = 5.0  # Upper bound as per the problem specification

    def __call__(self, func):
        # Initialize temperature and progressively adaptive cooling parameters
        T = 1.2  # Increased initial temperature for aggressive initial exploration
        T_min = 0.0003  # Lower minimum temperature to enhance fine-grained exploration in later stages
        alpha = 0.91  # Optimized cooling rate to extend exploration duration

        # Mutant and crossover parameters fine-tuned for enhanced performance
        F = 0.78  # Slightly increased Mutation factor to explore more diverse solutions
        CR = 0.89  # Increased Crossover probability to ensure better gene mixing

        population_size = 90  # Increased population size to enhance diversity
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Implement a dynamic mutation strategy with refined control
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                # Dynamic mutation factor adapts using a sigmoid function for precise control
                dynamic_F = F * (0.75 + 0.25 * np.tanh(5 * (evaluation_count / self.budget - 0.5)))
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Enhanced acceptance criteria with a temperature-dependent function
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.07 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Adaptive cooling schedule with refined sinusoidal modulation
            T *= alpha - 0.012 * np.sin(3.5 * np.pi * evaluation_count / self.budget)

        return f_opt, x_opt
