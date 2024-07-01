import numpy as np


class HybridEvolutionaryAnnealingOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimension is set to 5 as per problem statement
        self.lb = -5.0  # Lower bound of the search space
        self.ub = 5.0  # Upper bound of the search space

    def __call__(self, func):
        # Initialize parameters
        T_initial = 1.0  # Initial temperature for simulated annealing
        T = T_initial
        T_min = 0.01  # Minimum temperature to keep annealing active
        alpha = 0.98  # Cooling rate for temperature
        CR = 0.8  # Crossover probability
        F = 0.5  # Differential weight
        population_size = 40  # Population size

        # Initialize population
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                # Mutation and Crossover
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                # Simulated annealing acceptance
                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                if delta_fitness < 0 or np.random.rand() < np.exp(-delta_fitness / T):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Adaptive parameter updates
            T *= alpha
            CR = max(0.1, CR * 0.99)  # Gradually decrease CR
            F = min(1.0, F * 1.02)  # Gradually increase F to enhance exploration

        return f_opt, x_opt
