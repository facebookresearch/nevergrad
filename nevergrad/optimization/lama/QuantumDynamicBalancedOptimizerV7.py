import numpy as np


class QuantumDynamicBalancedOptimizerV7:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 1000  # Increased population size for greater exploration
        self.sigma_initial = 1.0  # Initial mutation spread
        self.sigma_final = 0.01  # Final mutation spread for fine-tuning
        self.elitism_factor = 0.05  # Reduced elitism to prevent premature convergence
        self.CR_initial = 0.9  # Initial crossover probability
        self.CR_final = 0.2  # Final crossover probability
        self.q_impact_initial = 0.01  # Initial quantum impact
        self.q_impact_final = 0.5  # Increased final quantum impact for intensive exploitation
        self.q_impact_increase_rate = 0.001  # Gradual increase in quantum impact

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

                # Mutation with quantum dynamic adjustment
                idxs = [j for j in range(self.pop_size) if j != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = (
                    a
                    + sigma * (b - c + q_impact * np.sin(c))
                    + q_impact * np.sin(np.pi * np.random.normal(size=self.dim))
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
