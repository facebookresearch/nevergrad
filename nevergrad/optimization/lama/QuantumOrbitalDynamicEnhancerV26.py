import numpy as np


class QuantumOrbitalDynamicEnhancerV26:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 150  # Further increased population size for enhanced initial exploration
        self.sigma_initial = 0.7  # Increasing the initial mutation spread for broader initial search
        self.sigma_final = 0.001  # Further reduced final mutation spread for finer exploitation
        self.CR_initial = 0.9  # Initial crossover probability
        self.CR_final = 0.01  # Minimal final crossover rate for increased exploitation in later stages
        self.harmonic_impulse_frequency = 0.1  # Adjusted frequency for harmonic impulse
        self.impulse_amplitude = 0.95  # Slightly adjusted impulse amplitude

    def __call__(self, func):
        # Initialize population and fitness
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Adaptive parameters
        sigma = self.sigma_initial
        CR = self.CR_initial

        # Evolution loop
        for iteration in range(self.budget // self.pop_size):
            for i in range(self.pop_size):
                # Mutation: DE/rand/1/bin
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = (
                    a
                    + sigma * (b - c)
                    + self.impulse_amplitude
                    * np.sin(2 * np.pi * self.harmonic_impulse_frequency * iteration)
                    * (b - a)
                )
                mutant = np.clip(mutant, -5.0, 5.0)

                # Crossover: Binomial
                cross_points = np.random.rand(self.dim) < CR
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

            # Adaptive parameter updates
            sigma = self.sigma_final + (self.sigma_initial - self.sigma_final) * (
                1 - iteration / (self.budget / self.pop_size)
            )
            CR = self.CR_final + (self.CR_initial - self.CR_final) * (
                1 - iteration / (self.budget / self.pop_size)
            )

        return best_fitness, best_ind
