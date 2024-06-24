import numpy as np


class RefinedQuantumHybridAdaptiveStrategyV3:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 250  # Increased Population size for broader search
        self.sigma_initial = 0.05  # Further reduced mutation spread
        self.elitism_factor = 5  # Reduced elite size to increase diversity
        self.sigma_decay = 0.99  # Reduced decay for slower convergence
        self.CR_base = 0.9  # Increased crossover probability for more exploration
        self.CR_decay = 0.995  # Slower decay rate for crossover probability
        self.q_impact = 0.8  # Increased base quantum impact
        self.q_impact_decay = 0.99  # Added decay to quantum impact to stabilize late convergence
        self.adaptation_rate = 0.1  # Increased adaptation rate for dynamic response

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
        for _ in range(self.budget // self.pop_size):
            for i in range(self.pop_size):
                if i < elite_size:  # Elite members are carried forward
                    continue

                # Mutation using a DE-like strategy with enhanced quantum effects
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

            # Adaptive updates to parameters
            sigma *= self.sigma_decay
            CR *= self.CR_decay
            q_impact *= self.q_impact_decay  # Decaying quantum impact based on adaptation rate
            if np.random.rand() < 0.5:
                q_impact += self.adaptation_rate * q_impact
            else:
                q_impact -= self.adaptation_rate * q_impact

        return best_fitness, best_ind
