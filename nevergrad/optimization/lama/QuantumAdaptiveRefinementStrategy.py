import numpy as np


class QuantumAdaptiveRefinementStrategy:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 300  # Further increased population size for broader exploration
        self.sigma_initial = 1.5  # Start with a wider spread in the population
        self.learning_rate = 0.025  # Finer learning rate adjustment
        self.CR_base = 0.7  # Higher initial crossover probability for vigorous exploration
        self.q_impact_initial = 0.8  # Increased initial quantum impact for stronger exploration
        self.q_impact_decay = 0.98  # Slower decay rate for the quantum impact
        self.sigma_decay = 0.98  # Slower decay for sigma to enhance exploration period
        self.elitism_factor = 15  # Increased elitism factor to stabilize top performers

    def __call__(self, func):
        # Initialize population within bounds
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Setup for elite solutions
        elite_size = max(1, int(self.elitism_factor * self.pop_size / 100))
        elites = np.argsort(fitness)[:elite_size]

        sigma = self.sigma_initial
        q_impact = self.q_impact_initial

        # Evolutionary loop
        for iteration in range(int(self.budget / self.pop_size)):
            sigma *= self.sigma_decay
            q_impact *= self.q_impact_decay
            current_CR = (
                self.CR_base - (iteration / (self.budget / self.pop_size)) * 0.2
            )  # Gradually decrease CR

            for i in range(self.pop_size):
                if i in elites:  # Elite members are kept unchanged
                    continue

                idxs = [idx for idx in range(self.pop_size) if idx != i and idx not in elites]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                quantum_term = q_impact * np.random.standard_cauchy(self.dim)
                mutant = best_ind + sigma * (a - b + c + quantum_term)
                mutant = np.clip(mutant, -5.0, 5.0)

                CRi = np.clip(current_CR + self.learning_rate * np.random.randn(), 0, 1)
                cross_points = np.random.rand(self.dim) < CRi
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

            # Update elites regularly
            elites = np.argsort(fitness)[:elite_size]

        return best_fitness, best_ind
