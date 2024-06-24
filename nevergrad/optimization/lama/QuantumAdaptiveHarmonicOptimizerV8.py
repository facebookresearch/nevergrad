import numpy as np


class QuantumAdaptiveHarmonicOptimizerV8:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 1500  # Increased population size for greater coverage
        self.sigma_initial = 1.5  # Initial mutation spread
        self.sigma_final = 0.005  # Finer final mutation spread for precision
        self.elitism_factor = 0.02  # Reduced elitism to increase diversity
        self.CR_initial = 0.95  # High initial crossover probability
        self.CR_final = 0.1  # Reduced final crossover probability
        self.q_impact_initial = 0.02  # Higher initial quantum impact
        self.q_impact_final = 0.6  # Increased final quantum impact for deep exploitation
        self.q_impact_increase_rate = 0.002  # Gradual increase in quantum impact
        self.harmonic_scale = 0.3  # Scaling factor for harmonic modulation

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

                # Mutation with quantum harmonic adjustments
                idxs = [j for j in range(self.pop_size) if j != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                harmonic_term = self.harmonic_scale * np.sin(
                    2 * np.pi * iteration / (self.budget / self.pop_size)
                )
                mutant = a + sigma * (b - c + q_impact * np.sin(c + harmonic_term))
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
