import numpy as np


class QuantumHybridAdaptiveStrategyV2:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 200  # Population size
        self.sigma_initial = 0.1  # Initial mutation spread, reduced to tighten exploration
        self.elitism_factor = 10  # Increased elite size to preserve good candidates
        self.sigma_decay = 0.98  # Slower decay for mutation spread
        self.CR_base = 0.8  # Lower initial crossover probability to allow more mutation effects
        self.CR_decay = 0.99  # Slower decay rate for crossover probability
        self.q_impact = 0.5  # Increased quantum impact factor on mutation vector for more diversity
        self.adaptation_rate = 0.05  # Rate at which the quantum impact factor adapulates

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

        # Evolutionary loop
        for _ in range(self.budget // self.pop_size):
            for i in range(self.pop_size):
                if i < elite_size:  # Elite members are carried forward
                    continue

                # Mutation using DE-like strategy with quantum effects
                idxs = [idx for idx in range(self.pop_size) if idx != i and idx >= elite_size]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = best_ind + sigma * (a - b + c) + self.q_impact * np.random.standard_cauchy(self.dim)
                mutant = np.clip(mutant, -5.0, 5.0)

                # Adaptive Crossover
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

            # Adaptive updates to parameters
            sigma *= self.sigma_decay
            CR *= self.CR_decay
            self.q_impact *= 1 + self.adaptation_rate if np.random.rand() < 0.5 else 1 - self.adaptation_rate

        return best_fitness, best_ind
