import numpy as np


class ProgressiveParticleSwarmOptimization:
    def __init__(self, budget=10000, population_size=40, omega=0.5, phi_p=0.2, phi_g=0.5):
        self.budget = budget
        self.population_size = population_size
        self.omega = omega  # Inertia weight
        self.phi_p = phi_p  # Personal learning coefficient
        self.phi_g = phi_g  # Global learning coefficient
        self.dim = 5  # Dimension of the problem

    def __call__(self, func):
        lb, ub = -5.0, 5.0  # Search space bounds
        # Initialize population
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.zeros((self.population_size, self.dim))
        personal_best = pop.copy()
        fitness = np.array([func(ind) for ind in pop])
        personal_best_fitness = fitness.copy()

        # Initial global best
        global_best_idx = np.argmin(fitness)
        global_best = pop[global_best_idx]

        evaluations = self.population_size

        # Main loop
        while evaluations < self.budget:
            r_p = np.random.uniform(0, 1, (self.population_size, self.dim))
            r_g = np.random.uniform(0, 1, (self.population_size, self.dim))

            # Update velocity and positions
            velocity = (
                self.omega * velocity
                + self.phi_p * r_p * (personal_best - pop)
                + self.phi_g * r_g * (global_best - pop)
            )
            pop = np.clip(pop + velocity, lb, ub)

            # Evaluate new positions
            for i in range(self.population_size):
                current_fitness = func(pop[i])
                evaluations += 1

                if current_fitness < personal_best_fitness[i]:
                    personal_best[i] = pop[i]
                    personal_best_fitness[i] = current_fitness

                    if current_fitness < fitness[global_best_idx]:
                        global_best_idx = i
                        global_best = pop[i]

            # Adaptive inertia weight
            if evaluations % (self.budget // 5) == 0:
                progress = evaluations / self.budget
                self.omega = max(0.4, self.omega * (1 - progress))
                self.phi_p = min(0.3, self.phi_p + progress * 0.1)
                self.phi_g = max(0.3, self.phi_g - progress * 0.1)

        return fitness[global_best_idx], global_best
