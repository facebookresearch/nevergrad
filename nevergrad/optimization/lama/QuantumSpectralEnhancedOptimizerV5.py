import numpy as np


class QuantumSpectralEnhancedOptimizerV5:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality of the problem
        self.pop_size = 500  # Increase population size for better exploration
        self.sigma_initial = 0.8  # Adjust initial mutation spread
        self.sigma_final = 0.0005  # Tighter control at the end of the search
        self.elitism_factor = 0.2  # Higher elitism to ensure retention of best found solutions
        self.CR_initial = 0.9  # Starting crossover probability
        self.CR_final = 0.5  # Lower final crossover probability to allow detailed local search
        self.q_impact_initial = 0.1  # Adjusted initial quantum impact for stronger early exploration
        self.q_impact_final = 0.2  # Higher final quantum impact for intensive exploitation
        self.q_impact_increase_rate = 0.003  # Faster increase in quantum impact through generations

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

                # Hybrid trigonometric mutation strategy enhanced with quantum theory influence
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = (
                    best_ind
                    + sigma * (a - b + np.cos(c))
                    + q_impact * np.cos(np.pi * np.random.standard_cauchy(self.dim))
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

            # Adaptively update parameters
            sigma = sigma * (self.sigma_final / self.sigma_initial) ** (1 / (self.budget / self.pop_size))
            CR = max(self.CR_final, CR - (self.CR_initial - self.CR_final) / (self.budget / self.pop_size))
            q_impact = min(self.q_impact_final, q_impact + self.q_impact_increase_rate)

        return best_fitness, best_ind
