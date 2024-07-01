import numpy as np


class QuantumSpectralAdaptiveOptimizerV2:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality
        self.pop_size = 800  # Population size adjusted for exploration vs. computation trade-off
        self.sigma_initial = 0.3  # Wider initial mutation spread
        self.sigma_final = 0.005  # Final reduced mutation spread
        self.elitism_factor = 0.1  # Reduced elitism factor for increased competition
        self.CR_initial = 0.95  # High initial crossover probability
        self.CR_final = 0.6  # Higher final crossover rate to allow for meaningful late convergence
        self.q_impact_initial = 0.005  # Lower initial quantum impact
        self.q_impact_final = 0.05  # Lesser final quantum impact
        self.q_impact_increase_rate = 0.0005  # Slower rate of quantum impact increase

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(x) for x in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        sigma = self.sigma_initial
        CR = self.CR_initial
        q_impact = self.q_impact_initial

        # Dynamic adjustment of parameters
        for iteration in range(self.budget // self.pop_size):
            elite_size = int(self.elitism_factor * self.pop_size)

            for i in range(self.pop_size):
                if i < elite_size:  # Elite members skip mutation and crossover
                    continue

                # Quantum-inspired trigonometric mutation with differential mutation components
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = (
                    best_ind + sigma * (a - b + np.sin(c)) + q_impact * np.random.standard_cauchy(self.dim)
                )
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
            sigma = sigma * (self.sigma_final / self.sigma_initial) ** (1 / (self.budget / self.pop_size))
            CR = max(self.CR_final, CR - (self.CR_initial - self.CR_final) / (self.budget / self.pop_size))
            q_impact = min(self.q_impact_final, q_impact + self.q_impact_increase_rate)

        return best_fitness, best_ind
