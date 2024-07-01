import numpy as np


class EliteTranscendentalEvolutionaryOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality of the problem
        self.lb = -5.0  # Lower bound of the search space
        self.ub = 5.0  # Upper bound of the search space

    def __call__(self, func):
        # Initiation of advanced temperature management settings
        T = 1.5  # Higher initial temperature to allow more exploration initially
        T_min = 0.0005  # Lower threshold to extend the search phase
        alpha = 0.95  # Slower cooling rate to explore more thoroughly

        # Enhanced mutation strategy parameters
        F_initial = 0.8  # Initial mutation factor
        F_final = 0.5  # Final mutation factor
        CR = 0.8  # Adjusted crossover probability

        population_size = 100  # Increased population size for better sampling
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Incorporating a mutation factor schedule and an adaptive acceptance mechanism
        while evaluation_count < self.budget and T > T_min:
            F = F_initial + (F_final - F_initial) * (
                evaluation_count / self.budget
            )  # Linear schedule for mutation factor
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                if trial_fitness < fitness[i] or np.random.rand() < np.exp(-(trial_fitness - fitness[i]) / T):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            T *= alpha  # Cooling happens after each generation

        return f_opt, x_opt
