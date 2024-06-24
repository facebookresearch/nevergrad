import numpy as np


class QuantumOrbitalDynamicEnhancerV29:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 300  # Increased population size for further exploration
        self.sigma_initial = 1.5  # Increased initial mutation spread
        self.sigma_final = 0.00001  # Smaller final mutation spread for finer exploitation
        self.CR_initial = 0.95  # High initial crossover probability
        self.CR_final = 0.01  # Higher final crossover rate for better exploitation
        self.harmonic_impulse_frequency = 0.01  # Lower frequency for significant impacts at fewer intervals
        self.impulse_amplitude = 1.5  # Increased amplitude for stronger perturbations
        self.adaptive_impulse = True  # Enable adaptive impulse adjustments based on performance stagnation

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
        stagnation_counter = 0  # Track performance stagnation

        # Evolution loop
        for iteration in range(self.budget // self.pop_size):
            for i in range(self.pop_size):
                # Mutation: DE/rand/1/bin
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                normal_vector = np.random.normal(0, 1, self.dim)  # Adding randomness
                mutant = (
                    a
                    + sigma * (b - c)
                    + self.impulse_amplitude
                    * np.sin(2 * np.pi * self.harmonic_impulse_frequency * iteration)
                    * normal_vector
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
                        stagnation_counter = 0  # Reset counter on improvement
                    else:
                        stagnation_counter += 1

            # Adaptive impulse based on performance stagnation
            if self.adaptive_impulse and stagnation_counter > self.pop_size * 10:
                self.impulse_amplitude += 0.5
                stagnation_counter = 0  # Reset after adaptation

            # Adaptive parameter updates
            sigma = self.sigma_final + (self.sigma_initial - self.sigma_final) * (
                1 - iteration / (self.budget / self.pop_size)
            )
            CR = self.CR_final + (self.CR_initial - self.CR_final) * (
                1 - iteration / (self.budget / self.pop_size)
            )

        return best_fitness, best_ind
