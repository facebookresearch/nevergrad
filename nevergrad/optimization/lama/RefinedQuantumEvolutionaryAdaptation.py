import numpy as np


class RefinedQuantumEvolutionaryAdaptation:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality of the problem
        self.pop_size = 250  # Increased population size for better exploration
        self.sigma_initial = 0.3  # Reduced initial standard deviation for more precise mutations
        self.learning_rate = 0.05  # Lower learning rate for gradual adaptive changes
        self.CR = 0.9  # Increased crossover probability for stronger gene mixing
        self.q_impact_initial = 0.1  # Increased initial quantum impact for robust global search
        self.q_impact_decay = 0.99  # Slower decay rate for quantum impact to sustain influence
        self.sigma_decay = 0.99  # Slower decay rate for sigma to maintain a valuable exploration range longer

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
            # Adapt sigma and quantum impact with refined rates
            sigma *= self.sigma_decay
            q_impact *= self.q_impact_decay

            # Generate new trial vectors
            for i in range(self.pop_size):
                # Mutation using differential evolution strategy with enhanced quantum impact
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                quantum_term = q_impact * np.random.standard_cauchy(self.dim)
                mutant = best_ind + sigma * (a - b) + quantum_term
                mutant = np.clip(mutant, -5.0, 5.0)

                # Crossover using an adaptively modified rate
                CRi = self.CR + self.learning_rate * (np.random.randn())
                cross_points = np.random.rand(self.dim) < CRi
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Fitness evaluation and selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

        return best_fitness, best_ind
