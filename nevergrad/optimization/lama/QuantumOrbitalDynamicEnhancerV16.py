import numpy as np


class QuantumOrbitalDynamicEnhancerV16:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5
        self.pop_size = 250  # Further increased population size for better exploration
        self.sigma_initial = 3.0  # Increased initial mutation spread to explore more aggressively
        self.sigma_final = 0.0005  # Reduced final mutation spread for precise fine-tuning
        self.CR_initial = 0.95  # Higher initial crossover probability for promoting early diversity
        self.CR_final = 0.01  # Lower final crossover probability to maintain important features
        self.elitism_factor = 0.01  # Reducing elitism to increase diversity and avoid local minima
        self.q_impact_initial = 0.01  # Reduced initial quantum impact to ensure gentle start
        self.q_impact_final = 0.99  # Increased maximum quantum impact for stronger exploitation later
        self.q_impact_increase_rate = 0.01  # Increased rate for a smoother transition to exploitation
        self.harmonic_impulse_frequency = 0.05  # Reduced frequency for less frequent but impactful shifts
        self.impulse_amplitude = 1.0  # Increased amplitude for more significant dynamic shifts

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        sigma = self.sigma_initial
        CR = self.CR_initial
        q_impact = self.q_impact_initial

        # Evolution loop
        for iteration in range(self.budget // self.pop_size):
            elite_size = int(self.elitism_factor * self.pop_size)

            for i in range(self.pop_size):
                if i < elite_size:  # Elite individuals undergo smaller changes
                    continue

                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                impulse = self.impulse_amplitude * np.sin(
                    2 * np.pi * self.harmonic_impulse_frequency * iteration
                )
                mutant = a + sigma * (b - c) + q_impact * np.sin(c + impulse) * (b - a)
                mutant = np.clip(mutant, -5.0, 5.0)

                # Crossover process
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

            # Adaptive parameter updating
            sigma = sigma * (self.sigma_final / self.sigma_initial) ** (1 / (self.budget / self.pop_size))
            CR = max(self.CR_final, CR - (self.CR_initial - self.CR_final) / (self.budget / self.pop_size))
            q_impact = min(self.q_impact_final, q_impact + self.q_impact_increase_rate)

        return best_fitness, best_ind
