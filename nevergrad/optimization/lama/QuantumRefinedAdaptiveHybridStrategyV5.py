import numpy as np


class QuantumRefinedAdaptiveHybridStrategyV5:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 400  # Further increased population size for better exploration
        self.sigma_initial = 0.2  # Initial mutation spread increased for wider searches early on
        self.elitism_factor = 2  # More limited elite size to encourage diversity
        self.sigma_decay = 0.97  # Faster decay for sigma to stabilize mutations as convergence approaches
        self.CR_base = 0.9  # Starting crossover probability slightly reduced to improve exploration
        self.CR_decay = 0.995  # More gradual crossover decay for sustained exploratory capability
        self.q_impact = 0.15  # Initial quantum impact increased
        self.q_impact_increase = 0.1  # Faster increase rate for quantum impact
        self.q_impact_limit = 1.0  # Higher limit for quantum impact to maximize the non-classical effects

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

                # Mutation: Adjusted strategy with quantum effects
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
            if iteration % (self.budget // (5 * self.pop_size)) == 0 and q_impact < self.q_impact_limit:
                q_impact += self.q_impact_increase  # Dynamically increase quantum impact more frequently

        return best_fitness, best_ind
