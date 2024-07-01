import numpy as np


class DynamicBalancingPSO:
    def __init__(
        self,
        budget=10000,
        population_size=100,
        omega_start=0.9,
        omega_end=0.4,
        phi_p=0.14,
        phi_g=0.16,
        adaptive_diversity=True,
    ):
        self.budget = budget
        self.population_size = population_size
        self.omega_start = omega_start
        self.omega_end = omega_end
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.adaptive_diversity = adaptive_diversity
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
        diversity = np.std(particles)

        # Optimization loop
        while evaluations < self.budget:
            omega = self.omega_start - ((self.omega_start - self.omega_end) * evaluations / self.budget)
            phi_total = self.phi_p + self.phi_g
            phi_ratio = self.phi_p / phi_total

            for i in range(self.population_size):
                r_p = np.random.random(self.dim)
                r_g = np.random.random(self.dim)

                # Update velocities
                velocities[i] = (
                    omega * velocities[i]
                    + phi_ratio * self.phi_p * r_p * (personal_best[i] - particles[i])
                    + (1 - phi_ratio) * self.phi_g * r_g * (global_best - particles[i])
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

            if self.adaptive_diversity:
                current_diversity = np.std(particles)
                if current_diversity < diversity:
                    phi_ratio += 0.05  # Encourage exploration
                else:
                    phi_ratio -= 0.05  # Encourage exploitation
                diversity = current_diversity

        return global_best_score, global_best
