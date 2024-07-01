import numpy as np


class QuantumHarmonicAdaptationStrategy:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality
        self.pop_size = 150  # Adjusted population size for targeted exploration
        self.sigma_initial = 0.7  # Initial standard deviation for enhanced exploration
        self.learning_rate = 0.08  # Fine-tuned learning rate for mutation influence
        self.CR_base = 0.5  # Base crossover probability, enabling more diversity
        self.q_impact_initial = 0.3  # Initial quantum impact for enhanced exploration
        self.q_impact_decay = 0.99  # Slower decay rate for quantum impact
        self.sigma_decay = 0.99  # Slower sigma decay rate
        self.elitism_factor = 2  # Lowered elitism factor to maintain strong candidates
        self.CR_adaptive_increment = 0.005  # Increment for adaptive crossover

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Preliminary setup for elite solutions
        elite_size = max(1, int(self.elitism_factor * self.pop_size / 100))
        elites = np.argsort(fitness)[:elite_size]

        sigma = self.sigma_initial
        q_impact = self.q_impact_initial

        # Evolution loop
        for iteration in range(int(self.budget / self.pop_size)):
            sigma *= self.sigma_decay
            q_impact *= self.q_impact_decay
            current_CR = self.CR_base + self.CR_adaptive_increment * iteration

            for i in range(self.pop_size):
                if i in elites:  # Skip mutation for elite members
                    continue

                idxs = [idx for idx in range(self.pop_size) if idx != i and idx not in elites]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                quantum_term = q_impact * np.random.standard_cauchy(self.dim)
                mutant = best_ind + sigma * (a - b + c) + quantum_term
                mutant = np.clip(mutant, -5.0, 5.0)

                CRi = current_CR + self.learning_rate * (np.random.rand() - 0.5)
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

            # Constantly update elites
            elites = np.argsort(fitness)[:elite_size]

        return best_fitness, best_ind
