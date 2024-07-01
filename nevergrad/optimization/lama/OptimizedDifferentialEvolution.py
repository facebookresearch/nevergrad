import numpy as np


class OptimizedDifferentialEvolution:
    def __init__(self, budget=10000, population_size=100, F=0.5, CR=0.7):
        self.budget = budget
        self.population_size = population_size
        self.F = F  # Differential weight, slightly reduced for stability
        self.CR = CR  # Crossover probability, lowered to increase exploration
        self.dim = 5  # Hardcoded as per problem specification
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
                # Mutation
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lb, self.ub)

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
# optimizer = OptimizedDifferentialEvolution(budget=10000)
# best_f, best_x = optimizer(your_black_box_function)
