import numpy as np


class QuantumInspiredSpiralOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality set as constant

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = np.zeros(self.dim)

        # Initialize population
        population_size = 30  # Reduced population size for more focused search
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        # Quantum-inspired parameters
        beta_min = 0.1  # Minimum beta value for quantum behavior
        beta_max = 1.0  # Maximum beta value for classical behavior
        beta_decay = 0.995  # Decay rate for beta to transition from quantum to classical

        mutation_factor = 0.75  # Mutation factor for differential evolution
        crossover_probability = 0.7  # Crossover probability

        # Adaptive step sizes for local search
        initial_step_size = 0.1
        step_decay = 0.98

        evaluations_left = self.budget - population_size
        beta = beta_max

        while evaluations_left > 0:
            # Update beta towards more classical behavior
            beta = max(beta_min, beta * beta_decay)

            for i in range(population_size):
                # Differential evolution strategy
                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + mutation_factor * (b - c), -5.0, 5.0)

                # Crossover
                cross_points = np.random.rand(self.dim) < crossover_probability
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Quantum-inspired perturbation
                quantum_jitter = np.random.normal(0, beta, self.dim)
                trial += quantum_jitter
                trial = np.clip(trial, -5.0, 5.0)

                # Evaluate trial solution
                f_trial = func(trial)
                evaluations_left -= 1
                if evaluations_left <= 0:
                    break

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                # Adaptive local search with diminishing step size
                step_size = initial_step_size * step_decay ** (self.budget - evaluations_left)
                for _ in range(5):  # Limited number of local search steps
                    new_trial = trial + np.random.normal(scale=step_size, size=self.dim)
                    new_trial = np.clip(new_trial, -5.0, 5.0)
                    f_new_trial = func(new_trial)
                    evaluations_left -= 1
                    if evaluations_left <= 0:
                        break
                    if f_new_trial < fitness[i]:
                        trial = new_trial
                        fitness[i] = f_new_trial
                        if f_new_trial < self.f_opt:
                            self.f_opt = f_new_trial
                            self.x_opt = trial

        return self.f_opt, self.x_opt
