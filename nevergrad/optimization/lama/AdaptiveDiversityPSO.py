import numpy as np


class AdaptiveDiversityPSO:
    def __init__(
        self,
        budget=10000,
        population_size=100,
        omega_start=0.9,
        omega_end=0.4,
        phi_p=0.1,
        phi_g=0.1,
        beta=0.2,
    ):
        self.budget = budget
        self.population_size = population_size
        # Inertia weight decreases linearly from omega_start to omega_end
        self.omega_start = omega_start
        self.omega_end = omega_end
        # Personal and global acceleration coefficients
        self.phi_p = phi_p
        self.phi_g = phi_g
        # Diversity control parameter
        self.beta = beta
        self.dim = 5  # Dimension of the problem

    def __call__(self, func):
        lb, ub = -5.0, 5.0  # Search space bounds
        particles = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best = particles.copy()
        personal_best_scores = np.array([func(p) for p in particles])
        global_best = particles[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        evaluations = self.population_size

        # Optimization loop
        while evaluations < self.budget:
            omega = self.omega_start - ((self.omega_start - self.omega_end) * evaluations / self.budget)
            mean_position = np.mean(particles, axis=0)
            diversity = np.mean(np.linalg.norm(particles - mean_position, axis=1))

            for i in range(self.population_size):
                r_p = np.random.random(self.dim)
                r_g = np.random.random(self.dim)
                r_b = np.random.random(self.dim)

                # Update velocities considering diversity
                velocities[i] = (
                    omega * velocities[i]
                    + self.phi_p * r_p * (personal_best[i] - particles[i])
                    + self.phi_g * r_g * (global_best - particles[i])
                    + self.beta * r_b * (mean_position - particles[i])
                )

                # Update positions
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)

                # Evaluate new solutions
                current_score = func(particles[i])
                evaluations += 1

                if evaluations >= self.budget:
                    break

                # Update personal and global bests
                if current_score < personal_best_scores[i]:
                    personal_best[i] = particles[i]
                    personal_best_scores[i] = current_score

                    if current_score < global_best_score:
                        global_best = particles[i]
                        global_best_score = current_score

        return global_best_score, global_best
