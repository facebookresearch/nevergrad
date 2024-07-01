import numpy as np


class EnhancedDifferentialSimulatedAnnealingOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initialize parameters
        T = 1.0  # Initial temperature for simulated annealing
        T_min = 0.001  # Minimum temperature to stop annealing
        alpha = 0.95  # Cooling rate for annealing, increased for slower cooling
        mutation_factor = 0.85  # Enhanced mutation factor for better exploration
        crossover_probability = 0.7  # Increased crossover probability

        # Initialize the population
        population_size = 20  # Increased population size for better diversity
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))

        # Evaluate the initial population
        fitness = np.array([func(ind) for ind in population])
        f_opt = np.min(fitness)
        x_opt = population[np.argmin(fitness)]

        # Main optimization loop
        evaluation_count = population_size
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                # Differential mutation and crossover
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + mutation_factor * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < crossover_probability
                trial = np.where(cross_points, mutant, population[i])

                # Simulated annealing acceptance criterion
                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]
                if delta_fitness < 0 or np.random.rand() < np.exp(-delta_fitness / T):
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Cool down the temperature
            T *= alpha

        return f_opt, x_opt
