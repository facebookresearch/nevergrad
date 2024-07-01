import numpy as np


class QuantumOrbitalDynamicOptimizerV11:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality
        self.pop_size = 300  # Reduced population size for more iterations per individual
        self.sigma_initial = 3.0  # Increased initial mutation spread
        self.sigma_final = 0.001  # Further reduced final mutation spread
        self.CR_initial = 0.9  # High initial crossover probability
        self.CR_final = 0.2  # Increased minimum final crossover probability
        self.elitism_factor = 0.1  # Increased elitism factor
        self.q_impact_initial = 0.05  # Lower initial quantum impact
        self.q_impact_final = 0.9  # Increased maximum quantum impact
        self.q_impact_increase_rate = 0.005  # Slower increase rate for quantum impact
        self.harmonic_impulse_frequency = 0.1  # Adjusted frequency of harmonic impulse modulation
        self.impulse_amplitude = 0.6  # Increased amplitude of the harmonic impulse

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

        # Evolutionary loop
        for iteration in range(self.budget // self.pop_size):
            elite_size = int(self.elitism_factor * self.pop_size)

            for i in range(self.pop_size):
                if i < elite_size:  # Elite members skip mutation and crossover
                    continue

                idxs = [j for j in range(self.pop_size) if j != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                impulse = self.impulse_amplitude * np.cos(
                    2 * np.pi * self.harmonic_impulse_frequency * iteration
                )
                mutant = a + sigma * (b - c + q_impact * np.sin(c + impulse))
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

            # Adaptively update parameters
            sigma = sigma * (self.sigma_final / self.sigma_initial) ** (1 / (self.budget / self.pop_size))
            CR = max(self.CR_final, CR - (self.CR_initial - self.CR_final) / (self.budget / self.pop_size))
            q_impact = min(self.q_impact_final, q_impact + self.q_impact_increase_rate)

        return best_fitness, best_ind
