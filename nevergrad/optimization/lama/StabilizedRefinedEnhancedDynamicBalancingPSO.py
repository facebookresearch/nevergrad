import numpy as np


class StabilizedRefinedEnhancedDynamicBalancingPSO:
    def __init__(
        self, budget=10000, population_size=200, omega=0.5, phi_p=0.2, phi_g=0.3, adaptive_threshold=0.1
    ):
        self.budget = budget
        self.population_size = population_size
        self.omega = omega  # Inertia coefficient
        self.phi_p = phi_p  # Coefficient of personal best
        self.phi_g = phi_g  # Coefficient of global best
        self.adaptive_threshold = adaptive_threshold
        self.dim = 5  # Dimension of the problem

    def __call__(self, func):
        lb, ub = -5.0, 5.0  # Search space bounds
        particles = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([func(p) for p in particles])
        global_best_position = particles[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                r_p = np.random.rand(self.dim)
                r_g = np.random.rand(self.dim)

                velocities[i] = (
                    self.omega * velocities[i]
                    + self.phi_p * r_p * (personal_best_positions[i] - particles[i])
                    + self.phi_g * r_g * (global_best_position - particles[i])
                )

                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)

                current_score = func(particles[i])
                evaluations += 1

                if current_score < personal_best_scores[i]:
                    personal_best_positions[i] = particles[i]
                    personal_best_scores[i] = current_score

                    if current_score < global_best_score:
                        global_best_position = particles[i]
                        global_best_score = current_score

                if evaluations >= self.budget:
                    break

            # Adaptive diversity control
            diversity = np.std(particles)
            if diversity < self.adaptive_threshold:
                self.phi_p += 0.01  # encourage exploration
                self.phi_g -= 0.01  # reduce exploitation
            else:
                self.phi_p -= 0.01  # reduce exploration
                self.phi_g += 0.01  # encourage exploitation

        return global_best_score, global_best_position
