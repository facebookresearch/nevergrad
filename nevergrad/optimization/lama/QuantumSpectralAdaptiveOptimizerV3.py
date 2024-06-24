import numpy as np


class QuantumSpectralAdaptiveOptimizerV3:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality
        self.pop_size = 500  # Population size adjusted for further exploration
        self.sigma_initial = 0.5  # Wider initial mutation spread
        self.sigma_final = 0.001  # Further reduced mutation spread at the end
        self.elitism_factor = 0.05  # Further reduced elitism factor
        self.CR_initial = 0.9  # Initial crossover probability
        self.CR_final = 0.7  # Higher final crossover rate
        self.q_impact_initial = 0.01  # Increased initial quantum impact
        self.q_impact_final = 0.1  # Increased final quantum impact
        self.q_impact_increase_rate = 0.001  # Increased rate of quantum impact increase

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

        # Evolutionary loop
        for iteration in range(self.budget // self.pop_size):
            elite_size = int(self.elitism_factor * self.pop_size)

            for i in range(self.pop_size):
                if i < elite_size:  # Elite members skip mutation and crossover
                    continue

                # Quantum-inspired trigonometric mutation with differential mutation components
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = (
                    best_ind
                    + sigma * (a - b + np.cos(c))
                    + q_impact * np.tan(np.random.standard_cauchy(self.dim))
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
