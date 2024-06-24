import numpy as np


class QuantumHybridAdaptiveStrategy:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 200  # Population size
        self.sigma_initial = 0.5  # Initial mutation spread
        self.elitism_factor = 5  # Percentage of elite individuals to carry forward without mutation
        self.sigma_decay = 0.99  # Decay factor for mutation spread
        self.CR_base = 0.9  # Initial crossover probability
        self.CR_decay = 0.995  # Decay rate for crossover probability
        self.q_impact = 0.3  # Quantum impact factor on mutation vector
        self.momentum = 0.8  # Momentum for the update mechanism

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
        velocities = np.zeros_like(pop)  # Initialize velocities for momentum-based updates

        # Evolutionary loop
        for _ in range(self.budget // self.pop_size):
            for i in range(self.pop_size):
                if i < elite_size:  # Elite members are carried forward
                    continue

                # Mutation using DE-like strategy with added momentum and quantum effects
                idxs = [idx for idx in range(self.pop_size) if idx != i and idx >= elite_size]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = best_ind + sigma * (a - b + c) + self.q_impact * np.random.standard_cauchy(self.dim)
                mutant = np.clip(mutant, -5.0, 5.0)

                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Update with momentum
                velocities[i] = self.momentum * velocities[i] + (1 - self.momentum) * (trial - pop[i])
                trial = pop[i] + velocities[i]
                trial = np.clip(trial, -5.0, 5.0)

                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

            # Update sigma and CR
            sigma *= self.sigma_decay
            CR *= self.CR_decay

        return best_fitness, best_ind
