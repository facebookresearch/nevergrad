import numpy as np


class EnhancedPrecisionBoostedDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 100  # Maintaining high population for diversity
        self.F_base = 0.5  # Reduced mutation factor for better local search
        self.CR_base = 0.9  # Increased crossover probability for higher genetic exchange
        self.F_min = 0.1  # Lower minimum mutation factor to prevent excessive perturbation
        self.CR_min = 0.6  # Higher minimum crossover to ensure continuous variability
        self.adaptive_increment = 0.05  # Increment factor for adaptive mutation strategy
        self.loss_threshold = 1e-6  # Threshold for loss stabilization

    def __call__(self, func):
        # Initialize population uniformly within search space bounds
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        # Identify the best individual
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx]

        # Evolutionary process given the budget constraints
        n_iterations = int(self.budget / self.pop_size)
        F = self.F_base
        CR = self.CR_base
        prev_best_fitness = best_fitness

        for iteration in range(n_iterations):
            # Check if fitness improvements have stabilized
            if abs(prev_best_fitness - best_fitness) < self.loss_threshold:
                F += self.adaptive_increment  # Increase mutation factor adaptively

            prev_best_fitness = best_fitness

            # Mutate and recombine
            for i in range(self.pop_size):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = a + F * (b - c)
                mutant = np.clip(mutant, -5.0, 5.0)
                trial = np.array([mutant[j] if np.random.rand() < CR else pop[i][j] for j in range(self.dim)])

                # Evaluate and select
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

            # Decay CR to maintain exploitation ability
            CR = max(self.CR_min, CR - (self.CR_base - self.CR_min) / n_iterations)

        return best_fitness, best_ind
