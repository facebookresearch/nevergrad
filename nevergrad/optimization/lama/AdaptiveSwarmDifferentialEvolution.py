import numpy as np


class AdaptiveSwarmDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 150  # Population size
        self.F_base = 0.5  # Base mutation factor
        self.CR = 0.9  # Crossover probability
        self.adapt_rate = 0.1  # Rate at which F adapts
        self.swarm_factor = np.linspace(0.1, 0.9, self.pop_size)  # Swarm interaction factor

    def __call__(self, func):
        # Initialize population within search space bounds
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        # Find the best initial solution
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Main evolutionary loop
        for i in range(int(self.budget / self.pop_size)):
            F_adapted = self.F_base + self.adapt_rate * np.sin(2 * np.pi * i / (self.budget / self.pop_size))

            for j in range(self.pop_size):
                # Mutation: DE/rand/1/bin with swarm interaction
                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = pop[j] + self.swarm_factor[j] * (best_ind - pop[j]) + F_adapted * (b - c)
                mutant = np.clip(mutant, -5.0, 5.0)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[j])

                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[j]:
                    pop[j] = trial
                    fitness[j] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

        return best_fitness, best_ind
