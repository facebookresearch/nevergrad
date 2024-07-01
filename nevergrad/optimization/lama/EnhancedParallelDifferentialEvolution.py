import numpy as np


class EnhancedParallelDifferentialEvolution:
    def __init__(self, budget=10000, population_size=100, F=0.8, CR=0.9, strategy="best"):
        self.budget = budget
        self.population_size = population_size
        self.F = F  # Increased Differential weight to encourage more aggressive diversification
        self.CR = CR  # Increased Crossover probability to allow more mixing
        self.strategy = strategy  # Strategy for the selection of base vector in mutation
        self.dim = 5  # Dimensionality of the problem
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Initialize population randomly
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        # Main loop
        evaluations = self.population_size
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation based on strategy
                if self.strategy == "best":
                    best_idx = np.argmin(fitness)
                    base = population[best_idx]
                else:  # "random" strategy
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    base = population[np.random.choice(idxs)]

                # Generate mutant
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(base + self.F * (a - b), self.lb, self.ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

                # Check if budget exhausted
                if evaluations >= self.budget:
                    break

        # Find the best solution
        best_idx = np.argmin(fitness)
        return fitness[best_idx], population[best_idx]


# Example usage:
# optimizer = EnhancedParallelDifferentialEvolution(budget=10000)
# best_f, best_x = optimizer(your_black_box_function)
