import numpy as np


class AdvancedRefinedHybridEvolutionaryAnnealingOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Adjusted initial temperature and more gradual cooling
        T = 2.0  # Start with a higher initial temperature for wider early exploration
        T_min = 0.0005  # Further decreased minimum temperature for extended fine-tuning
        alpha = 0.95  # More gradual cooling to enhance thorough exploration

        # Mutation and recombination parameters fine-tuned
        F = 0.8  # Increased mutation factor
        CR = 0.85  # Adjusted crossover probability

        # Population size slightly increased
        population_size = 70
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Iteration with enhanced mutation strategy and dynamic components adjustment
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Adaptive mutation factor influenced by both temperature and performance
                dynamic_F = F * (1 + 0.15 * np.log(1 + T) * (f_opt / fitness[i]))
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Dynamic adjustment of cooling based on both temperature and progress
            adaptive_cooling = alpha - 0.02 * (T / T_min) * (evaluation_count / self.budget)
            T = max(T * adaptive_cooling, T_min)

        return f_opt, x_opt
