import numpy as np


class EnhancedQuantumHarmonicAdaptationStrategy:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 180  # Slightly larger population size for broader search
        self.sigma_initial = 1.0  # Increased initial standard deviation for global exploration
        self.learning_rate = 0.07  # Slightly reduced learning rate for more stable adaptation
        self.CR_base = 0.6  # Increased base crossover probability for higher diversity in offspring
        self.q_impact_initial = 0.4  # Increased initial quantum impact to boost early exploration
        self.q_impact_decay = 0.98  # More aggressive decay rate for the quantum impact
        self.sigma_decay = 0.98  # More aggressive decay for sigma to focus on local areas faster
        self.elitism_factor = 5  # Increased elitism factor to ensure survival of top individuals
        self.CR_adaptive_increment = 0.004  # More fine-tuned control over adaptive crossover increment

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
            current_CR = self.CR_base + self.CR_adaptive_increment * iteration

            for i in range(self.pop_size):
                if i in elites:  # Avoid disturbing elite members
                    continue

                idxs = [idx for idx in range(self.pop_size) if idx != i and idx not in elites]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                quantum_term = q_impact * np.random.standard_cauchy(self.dim)
                mutant = best_ind + sigma * (a - b + c + quantum_term)
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

            # Update elites regularly
            elites = np.argsort(fitness)[:elite_size]

        return best_fitness, best_ind
