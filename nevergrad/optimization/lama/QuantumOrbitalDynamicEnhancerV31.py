import numpy as np


class QuantumOrbitalDynamicEnhancerV31:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 500  # Increased population size for more exploration potential
        self.sigma_initial = 0.9  # Initial mutation spread
        self.sigma_final = 0.001  # Final mutation spread for deeper exploitation
        self.CR_initial = 0.95  # Initial crossover probability
        self.CR_final = 0.1  # Final crossover rate to maintain a balance between exploration and exploitation
        self.harmonic_impulse_frequency = 0.025  # Adjust frequency based on previous results
        self.impulse_amplitude_initial = 1.2  # Start with a higher impulse amplitude
        self.impulse_amplitude_final = 0.05  # Reduce the amplitude over iterations
        self.adaptive_impulse = True  # Enable adaptive impulse adjustments based on performance

    def __call__(self, func):
        # Initialize population and fitness
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Evolution loop
        for iteration in range(self.budget // self.pop_size):
            impulse_amplitude = self.impulse_amplitude_final + (
                self.impulse_amplitude_initial - self.impulse_amplitude_final
            ) * (1 - iteration / (self.budget / self.pop_size))

            for i in range(self.pop_size):
                # Mutation: DE/rand/1/bin with enhanced harmonic impulse
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                normal_vector = np.random.normal(0, 1, self.dim)  # Adding randomness
                mutant = (
                    a
                    + impulse_amplitude * (b - c)
                    + impulse_amplitude
                    * np.cos(2 * np.pi * self.harmonic_impulse_frequency * iteration)
                    * normal_vector
                )
                mutant = np.clip(mutant, -5.0, 5.0)

                # Crossover: Binomial with adaptive crossover rate
                cr_current = self.CR_initial - (self.CR_initial - self.CR_final) * (
                    iteration / (self.budget / self.pop_size)
                )
                cross_points = np.random.rand(self.dim) < cr_current
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

        return best_fitness, best_ind
