import numpy as np


class AdaptiveQuantumEvolutionStrategy:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality of the problem
        self.pop_size = 200  # Adjusted population size for enhanced exploration
        self.sigma_initial = 0.5  # Adjusted initial standard deviation for mutation
        self.learning_rate = 0.1  # Learning rate for adaptive quantum impact
        self.CR = 0.8  # Adjusted crossover probability for robustness
        self.q_impact_initial = 0.05  # Initial quantum impact in mutation
        self.q_impact_decay = 0.995  # Decay rate for quantum impact
        self.sigma_decay = 0.995  # Adjusted decay rate for sigma

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        sigma = self.sigma_initial
        q_impact = self.q_impact_initial

        # Evolution loop
        for iteration in range(int(self.budget / self.pop_size)):
            # Adapt sigma and quantum impact
            sigma *= self.sigma_decay
            q_impact *= self.q_impact_decay

            # Generate new trial vectors
            for i in range(self.pop_size):
                # Mutation using differential evolution strategy and quantum impact
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = (
                    best_ind + sigma * (a - b) + q_impact * np.random.standard_cauchy(self.dim)
                )  # Quantum influenced mutation
                mutant = np.clip(mutant, -5.0, 5.0)

                # Adaptive Crossover
                CRi = self.CR + self.learning_rate * (np.random.randn())
                cross_points = np.random.rand(self.dim) < CRi
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Evaluate
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

        return best_fitness, best_ind
