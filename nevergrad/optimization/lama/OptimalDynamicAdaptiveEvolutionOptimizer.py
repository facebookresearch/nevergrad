import numpy as np


class OptimalDynamicAdaptiveEvolutionOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality as per the problem description
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Initial temperature for simulated annealing and evolutionary strategies
        T = 1.2
        T_min = 0.0005
        alpha = 0.92  # Cooling rate

        # Evolutionary strategy parameters
        F = 0.75  # Base mutation factor
        CR = 0.88  # Crossover probability optimized for diversity and convergence

        population_size = 80  # Population size adjusted for optimal search space exploration
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Main optimization loop
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                # Dynamic mutation influenced by temperature and normalized progress
                dynamic_F = F * (
                    np.exp(-0.2 * T) * (0.5 + 0.5 * np.cos(np.pi * evaluation_count / self.budget))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Temperature and fitness-delta based acceptance
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.1 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Adaptively modify the cooling rate based on the search state
            adaptive_cooling = alpha - 0.015 * np.sin(1.5 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
