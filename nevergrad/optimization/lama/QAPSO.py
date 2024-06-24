import numpy as np


class QAPSO:
    def __init__(
        self,
        budget=1000,
        num_particles=30,
        inertia_weight=0.5,
        cognitive_weight=1.5,
        social_weight=2.0,
        acceleration_coeff=1.1,
    ):
        self.budget = budget
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.acceleration_coeff = acceleration_coeff

    def __call__(self, func):
        self.dim = 5
        self.f_opt = np.Inf
        self.x_opt = None

        # Initialize particles positions and velocities
        particles_pos = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        particles_vel = np.zeros((self.num_particles, self.dim))
        personal_best_pos = np.copy(particles_pos)
        personal_best_val = np.array([func(x) for x in particles_pos])
        global_best_idx = np.argmin(personal_best_val)
        global_best_pos = np.copy(personal_best_pos[global_best_idx])

        for _ in range(self.budget):
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                r3 = np.random.rand()
                particles_vel[i] = (
                    self.inertia_weight * particles_vel[i]
                    + self.cognitive_weight * r1 * (personal_best_pos[i] - particles_pos[i])
                    + self.social_weight * r2 * (global_best_pos - particles_pos[i])
                )

                # Acceleration towards global best
                accel = self.acceleration_coeff * r3 * (global_best_pos - particles_pos[i])
                particles_vel[i] += accel
                particles_pos[i] += particles_vel[i]

                # Boundary enforcement
                particles_pos[i] = np.clip(particles_pos[i], -5.0, 5.0)

                # Update personal best
                f_val = func(particles_pos[i])
                if f_val < personal_best_val[i]:
                    personal_best_val[i] = f_val
                    personal_best_pos[i] = np.copy(particles_pos[i])

                    if f_val < self.f_opt:
                        self.f_opt = f_val
                        self.x_opt = np.copy(particles_pos[i])

            global_best_idx = np.argmin(personal_best_val)
            global_best_pos = np.copy(personal_best_pos[global_best_idx])

        return self.f_opt, self.x_opt
