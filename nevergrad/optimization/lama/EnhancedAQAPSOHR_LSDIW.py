import numpy as np


class EnhancedAQAPSOHR_LSDIW:
    def __init__(
        self,
        budget=1000,
        num_particles=30,
        cognitive_weight=1.5,
        social_weight=2.0,
        acceleration_coeff=1.1,
        restart_threshold=50,
        restart_prob=0.1,
        initial_velocity_clamp=0.5,
        local_search_radius=0.05,
        local_search_samples=20,
        inertia_weight=0.5,
    ):
        self.budget = budget
        self.num_particles = num_particles
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.acceleration_coeff = acceleration_coeff
        self.restart_threshold = restart_threshold
        self.restart_prob = restart_prob
        self.initial_velocity_clamp = initial_velocity_clamp
        self.local_search_radius = local_search_radius
        self.local_search_samples = local_search_samples
        self.inertia_weight = inertia_weight

    def random_restart(self):
        return np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))

    def local_search(self, x, func):
        best_x = x
        best_f = func(x)

        for _ in range(self.local_search_samples):
            x_new = x + np.random.uniform(-self.local_search_radius, self.local_search_radius, size=self.dim)
            x_new = np.clip(x_new, -5.0, 5.0)
            f_val = func(x_new)

            if f_val < best_f:
                best_f = f_val
                best_x = x_new

        return best_x, best_f

    def update_velocity_clamp(self, t):
        return max(0.1, self.initial_velocity_clamp - 0.3 * t / self.budget)

    def update_inertia_weight(self, t):
        return self.inertia_weight + 0.5 * (1 - t / self.budget)

    def __call__(self, func):
        self.dim = 5
        self.f_opt = np.Inf
        self.x_opt = None

        particles_pos = self.random_restart()
        particles_vel = np.zeros((self.num_particles, self.dim))
        personal_best_pos = np.copy(particles_pos)
        personal_best_val = np.array([func(x) for x in particles_pos])
        global_best_idx = np.argmin(personal_best_val)
        global_best_pos = np.copy(personal_best_pos[global_best_idx])

        for t in range(1, self.budget + 1):
            inertia_weight = self.update_inertia_weight(t)
            velocity_clamp = self.update_velocity_clamp(t)

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                r3 = np.random.rand()

                particles_vel[i] = (
                    inertia_weight * particles_vel[i]
                    + self.cognitive_weight * r1 * (personal_best_pos[i] - particles_pos[i])
                    + self.social_weight * r2 * (global_best_pos - particles_pos[i])
                )

                accel = self.acceleration_coeff * r3 * (global_best_pos - particles_pos[i])
                particles_vel[i] += accel

                # Velocity clamping
                particles_vel[i] = np.clip(particles_vel[i], -velocity_clamp, velocity_clamp)

                particles_pos[i] += particles_vel[i]
                particles_pos[i] = np.clip(particles_pos[i], -5.0, 5.0)

                f_val = func(particles_pos[i])

                if f_val < personal_best_val[i]:
                    personal_best_val[i] = f_val
                    personal_best_pos[i] = np.copy(particles_pos[i])

                    if f_val < self.f_opt:
                        self.f_opt = f_val
                        self.x_opt = np.copy(particles_pos[i])

            global_best_idx = np.argmin(personal_best_val)
            global_best_pos = np.copy(personal_best_pos[global_best_idx])

            if np.random.rand() < self.restart_prob:  # Random restart with probability restart_prob
                particles_pos = self.random_restart()
                particles_vel = np.zeros((self.num_particles, self.dim))

            # Integrate local search
            for i in range(self.num_particles):
                particles_pos[i], _ = self.local_search(particles_pos[i], func)

        return self.f_opt, self.x_opt
