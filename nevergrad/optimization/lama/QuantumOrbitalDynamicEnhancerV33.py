import numpy as np


class QuantumOrbitalDynamicEnhancerV33:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the optimization problem
        self.pop_size = 500  # Adjusted population size for better balance
        self.sigma_initial = 0.9  # Slightly reduced mutation spread for improved control
        self.sigma_final = 0.005  # Finer final mutation spread for detailed exploitation
        self.CR_initial = 0.9  # Adjusted initial crossover probability
        self.CR_final = 0.1  # Slightly increased final crossover rate for better genetic mixing
        self.harmonic_impulse_frequency = 0.02  # Moderately set frequency for impulse
        self.impulse_amplitude_initial = 1.2  # Moderately high initial amplitude for exploration
        self.impulse_amplitude_final = (
            0.02  # Increased final amplitude for more effective late-stage optimization
        )

    def __call__(self, func):
        # Initialize population and fitness array
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Evolution process
        for iteration in range(self.budget // self.pop_size):
            impulse_amplitude = self.impulse_amplitude_final + (
                self.impulse_amplitude_initial - self.impulse_amplitude_final
            ) * (1 - iteration / (self.budget / self.pop_size))

            for i in range(self.pop_size):
                # Mutation strategy with impulse modification
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                normal_vector = np.random.normal(0, 1, self.dim)
                mutant = (
                    a
                    + impulse_amplitude * (b - c)
                    + impulse_amplitude
                    * np.sin(2 * np.pi * self.harmonic_impulse_frequency * iteration)
                    * normal_vector
                )
                mutant = np.clip(mutant, -5.0, 5.0)

                # Crossover strategy with dynamic rate
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
