import numpy as np


class EnhancedAQAPSO_LS_DIW_AP_Ultimate_Redefined_Refined:
    def __init__(self, budget=1000, num_particles=30):
        self.budget = budget
        self.num_particles = num_particles
        self.dim = 5

    def random_restart(self):
        return np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))

    def local_search(self, x, func):
        best_x = x
        best_f = func(x)

        for _ in range(500):  # Keep the same local search iterations
            x_new = x + 0.1 * np.random.randn(self.dim)  # Adjusted the local search step size
            x_new = np.clip(x_new, -5.0, 5.0)
            f_val = func(x_new)

            if f_val < best_f:
                best_f = f_val
                best_x = x_new

        return best_x, best_f

    def update_inertia_weight(self, t):
        return 0.9 - 0.8 * t / self.budget  # Keep the same inertia weight update

    def update_parameters(self, t):
        return (
            1.8 - 1.2 * t / self.budget,
            2.2 - 1.2 * t / self.budget,
        )  # Keep the same cognitive and social weights

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        particles_pos = self.random_restart()
        particles_vel = np.zeros((self.num_particles, self.dim))
        personal_best_pos = np.copy(particles_pos)
        personal_best_val = np.array([func(x) for x in particles_pos])
        global_best_idx = np.argmin(personal_best_val)
        global_best_pos = np.copy(personal_best_pos[global_best_idx])

        for t in range(1, self.budget + 1):
            inertia_weight = self.update_inertia_weight(t)
            cognitive_weight, social_weight = self.update_parameters(t)

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                r3 = np.random.rand()

                particles_vel[i] = (
                    inertia_weight * particles_vel[i]
                    + cognitive_weight * r1 * (personal_best_pos[i] - particles_pos[i])
                    + social_weight * r2 * (global_best_pos - particles_pos[i])
                )

                accel = 1.8 * r3 * (global_best_pos - particles_pos[i])  # Adjusted acceleration coefficient
                particles_vel[i] += accel

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

            # Integrate local search
            for i in range(self.num_particles):
                particles_pos[i], _ = self.local_search(particles_pos[i], func)

        return self.f_opt, self.x_opt
