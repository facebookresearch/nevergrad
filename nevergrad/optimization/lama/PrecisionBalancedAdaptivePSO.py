import numpy as np


class PrecisionBalancedAdaptivePSO:
    def __init__(
        self,
        budget=10000,
        population_size=100,
        omega_initial=0.9,
        omega_final=0.4,
        phi_p=0.2,
        phi_g=0.4,
        adaptive_precision=True,
    ):
        self.budget = budget
        self.population_size = population_size
        self.omega_initial = omega_initial  # Initial inertia coefficient
        self.omega_final = omega_final  # Final inertia coefficient
        self.phi_p = phi_p  # Personal best influence factor
        self.phi_g = phi_g  # Global best influence factor
        self.dim = 5  # Problem dimensionality
        self.lb, self.ub = -5.0, 5.0  # Bounds of the search space
        self.adaptive_precision = adaptive_precision  # Flag to enable adaptive precision

    def __call__(self, func):
        particles = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([func(p) for p in particles])
        global_best_position = particles[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        evaluation_counter = self.population_size
        while evaluation_counter < self.budget:
            # Update inertia over time to balance exploration and exploitation
            omega = self.omega_initial - (
                (self.omega_initial - self.omega_final) * (evaluation_counter / self.budget)
            )

            for i in range(self.population_size):
                r_p = np.random.rand(self.dim)
                r_g = np.random.rand(self.dim)

                velocities[i] = (
                    omega * velocities[i]
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
