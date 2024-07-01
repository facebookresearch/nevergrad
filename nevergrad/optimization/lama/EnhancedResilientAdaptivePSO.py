import numpy as np


class EnhancedResilientAdaptivePSO:
    def __init__(
        self,
        budget=10000,
        population_size=100,
        omega=0.7,
        phi_p=0.15,
        phi_g=0.3,
        precision_decay=0.98,
        adaptive_phi=True,
    ):
        self.budget = budget
        self.population_size = population_size
        self.omega = omega  # Inertia coefficient
        self.phi_p = phi_p  # Coefficient of personal best
        self.phi_g = phi_g  # Coefficient of global best
        self.dim = 5  # Dimension of the problem
        self.lb, self.ub = -5.0, 5.0  # Search space bounds
        self.precision_decay = precision_decay
        self.adaptive_phi = adaptive_phi

    def __call__(self, func):
        particles = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([func(p) for p in particles])
        global_best_position = particles[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        evaluation_counter = self.population_size
        remaining_budget = self.budget - evaluation_counter

        while evaluation_counter < self.budget:
            self.omega *= self.precision_decay  # Gradually reduce inertia

            # Dynamically adjust phi parameters if adaptive_phi is True
            if self.adaptive_phi:
                self.phi_p = 0.1 + (0.5 - 0.1) * np.exp(-0.01 * evaluation_counter)
                self.phi_g = 0.1 + (0.5 - 0.1) * np.exp(-0.01 * evaluation_counter)

            for i in range(self.population_size):
                r_p = np.random.rand(self.dim)
                r_g = np.random.rand(self.dim)

                velocities[i] = (
                    self.omega * velocities[i]
                    + self.phi_p * r_p * (personal_best_positions[i] - particles[i])
                    + self.phi_g * r_g * (global_best_position - particles[i])
                )

                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.lb, self.ub)

                current_score = func(particles[i])
                evaluation_counter += 1

                if current_score < personal_best_scores[i]:
                    personal_best_positions[i] = particles[i]
                    personal_best_scores[i] = current_score

                    if current_score < global_best_score:
                        global_best_position = particles[i]
                        global_best_score = current_score

                if evaluation_counter >= self.budget:
                    break

        return global_best_score, global_best_position
