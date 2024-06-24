import numpy as np


class QuantumEvolutionaryOptimization:
    def __init__(self, budget=1000, num_particles=10, num_iterations=100):
        self.budget = budget
        self.num_particles = num_particles
        self.num_iterations = num_iterations

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        dimensions = 5
        bounds = func.bounds
        particles = np.random.uniform(bounds.lb, bounds.ub, size=(self.num_particles, dimensions))
        velocities = np.zeros((self.num_particles, dimensions))
        pbest_positions = particles.copy()
        pbest_values = np.array([func(p) for p in pbest_positions])
        gbest_position = pbest_positions[np.argmin(pbest_values)]
        gbest_value = np.min(pbest_values)

        for _ in range(self.num_iterations):
            for i in range(self.num_particles):
                r1, r2 = np.random.uniform(0, 1, size=2)
                velocities[i] = (
                    0.5 * velocities[i]
                    + 2 * r1 * (pbest_positions[i] - particles[i])
                    + 2 * r2 * (gbest_position - particles[i])
                )
                particles[i] = np.clip(particles[i] + velocities[i], bounds.lb, bounds.ub)
                f = func(particles[i])

                if f < pbest_values[i]:
                    pbest_positions[i] = particles[i]
                    pbest_values[i] = f

                    if f < gbest_value:
                        gbest_position = particles[i]
                        gbest_value = f

        self.f_opt = gbest_value
        self.x_opt = gbest_position

        return self.f_opt, self.x_opt
