import numpy as np


class QuantumAdaptiveHybridStrategyV4:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 300  # Slightly increased population size
        self.sigma_initial = 0.1  # Initial mutation spread
        self.elitism_factor = 3  # Reduced elite size for more diversity
        self.sigma_decay = 0.98  # Steeper decay for sigma
        self.CR_base = 0.95  # Higher initial crossover probability
        self.CR_decay = 0.99  # Slower decay rate for crossover probability
        self.q_impact = 0.1  # Lower quantum impact
        self.q_impact_increase = 0.05  # Increase quantum impact dynamically
        self.q_impact_limit = 0.95  # Maximum limit for quantum impact

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(x) for x in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        sigma = self.sigma_initial
        CR = self.CR_base
        elite_size = int(self.elitism_factor * self.pop_size / 100)
        q_impact = self.q_impact

        # Evolutionary loop
        for iteration in range(self.budget // self.pop_size):
            for i in range(self.pop_size):
                if i < elite_size:  # Elite members are carried forward
                    continue

                # Mutation: DE-like strategy with quantum effects
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = best_ind + sigma * (a - b + c) + q_impact * np.random.standard_cauchy(self.dim)
                mutant = np.clip(mutant, -5.0, 5.0)

                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

            # Adaptive parameter updates
            sigma *= self.sigma_decay
            CR *= self.CR_decay
            if iteration % (self.budget // (10 * self.pop_size)) == 0 and q_impact < self.q_impact_limit:
                q_impact += self.q_impact_increase  # Dynamically increase quantum impact

        return best_fitness, best_ind
