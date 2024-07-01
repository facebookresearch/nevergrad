import numpy as np


class EnhancedHybridGradientPSO:
    def __init__(
        self,
        budget=10000,
        population_size=50,
        omega_start=0.9,
        omega_end=0.4,
        phi_p=0.5,
        phi_g=0.5,
        learning_rate=0.05,
    ):
        self.budget = budget
        self.population_size = population_size
        self.omega_start = omega_start
        self.omega_end = omega_end
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.learning_rate = learning_rate
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
            for i in range(self.population_size):
                r_p = np.random.random(self.dim)
                r_g = np.random.random(self.dim)

                # Update velocities
                velocities[i] = (
                    omega * velocities[i]
                    + self.phi_p * r_p * (personal_best[i] - particles[i])
                    + self.phi_g * r_g * (global_best - particles[i])
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

    def estimate_gradient(self, x, objective, epsilon=1e-5):
        gradient = np.zeros(self.dim)
        for j in range(self.dim):
            x1 = np.array(x)
            x2 = np.array(x)
            x1[j] += epsilon
            x2[j] -= epsilon
            gradient[j] = (objective(x1) - objective(x2)) / (2 * epsilon)
        return gradient
