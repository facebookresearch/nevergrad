import numpy as np


class QuantumStateRefinedHybridStrategy:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality
        self.pop_size = 300  # Population size
        self.sigma_initial = 0.3  # Initial standard deviation for exploration
        self.learning_rate = 0.08  # Learning rate for adaptiveness in crossover
        self.CR = 0.9  # Crossover probability
        self.q_impact_initial = 0.2  # Initial quantum impact for exploration
        self.q_impact_decay = 0.99  # Decay rate for quantum impact
        self.sigma_decay = 0.99  # Decay rate for sigma
        self.elitism_factor = 10  # Increased elitism factor

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Maintain a set of elite solutions
        elite_size = max(1, int(self.elitism_factor * self.pop_size / 100))
        elites = np.argsort(fitness)[:elite_size]

        sigma = self.sigma_initial
        q_impact = self.q_impact_initial

        # Evolution loop
        for iteration in range(int(self.budget / self.pop_size)):
            sigma *= self.sigma_decay
            q_impact *= self.q_impact_decay

            for i in range(self.pop_size):
                if i in elites:  # Skip mutation for elites
                    continue

                idxs = [idx for idx in range(self.pop_size) if idx != i and idx not in elites]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                quantum_term = q_impact * np.random.standard_cauchy(self.dim)
                mutant = best_ind + sigma * (a - b) + quantum_term
                mutant = np.clip(mutant, -5.0, 5.0)

                CRi = self.CR + self.learning_rate * (np.random.randn())
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

            # Update elites
            elites = np.argsort(fitness)[:elite_size]

        return best_fitness, best_ind
