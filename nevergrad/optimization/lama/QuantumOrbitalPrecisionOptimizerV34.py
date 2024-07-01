import numpy as np


class QuantumOrbitalPrecisionOptimizerV34:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5
        self.pop_size = 150  # Further reduced population size for quicker evaluations
        self.sigma_initial = 0.3  # Starting mutation spread
        self.sigma_final = 0.0005  # Final mutation spread for precise tweaking
        self.CR_initial = 0.9  # High initial crossover probability for diverse exploration
        self.CR_final = 0.1  # Ending crossover probability for maintaining elite traits

        # Impulse strategy parameters
        self.harmonic_impulse_frequency = 0.1  # Increase in frequency for dynamic adaptation
        self.impulse_amplitude_initial = 2.0  # Larger initial amplitude for strong early global search
        self.impulse_amplitude_final = 0.005  # Minimized final amplitude for local area exploitation

    def __call__(self, func):
        # Initialize population within the bounds and evaluate fitness
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Evolutionary loop
        for iteration in range(self.budget // self.pop_size):
            impulse_amplitude = self.impulse_amplitude_final + (
                self.impulse_amplitude_initial - self.impulse_amplitude_final
            ) * (1 - iteration / (self.budget / self.pop_size))

            for i in range(self.pop_size):
                # Mutation with dynamic impulse
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                normal_vector = np.random.normal(0, 1, self.dim)
                mutant = (
                    a
                    + impulse_amplitude * (b - c)
                    + impulse_amplitude
                    * np.cos(2 * np.pi * self.harmonic_impulse_frequency * iteration)
                    * normal_vector
                )
                mutant = np.clip(mutant, -5.0, 5.0)

                # Crossover with dynamic rate adjustment
                cr_current = self.CR_initial - (self.CR_initial - self.CR_final) * (
                    iteration / (self.budget / self.pop_size)
                )
                cross_points = np.random.rand(self.dim) < cr_current
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Selection step
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

        return best_fitness, best_ind
