import numpy as np


class PrecisionAdaptivePSO:
    def __init__(
        self,
        budget=10000,
        population_size=150,
        inertia_strategy="nonlinear",
        phi_p=0.1,
        phi_g=0.9,
        min_omega=0.1,
        max_omega=0.9,
    ):
        self.budget = budget
        self.population_size = population_size
        self.inertia_strategy = inertia_strategy
        self.phi_p = phi_p  # Personal attraction coefficient
        self.phi_g = phi_g  # Global attraction coefficient
        self.min_omega = min_omega
        self.max_omega = max_omega
        self.dim = 5  # Problem dimensionality
        self.lb, self.ub = -5.0, 5.0  # Boundary limits of the search space

    def __call__(self, func):
        particles = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([func(p) for p in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        evaluation_counter = self.population_size

        while evaluation_counter < self.budget:
            omega = self.compute_inertia(evaluation_counter)

            for i in range(self.population_size):
                r_p = np.random.rand(self.dim)
                r_g = np.random.rand(self.dim)

                # Update velocities and positions with dynamic inertia
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

    def compute_inertia(self, current_step):
        if self.inertia_strategy == "nonlinear":
            return self.max_omega - (self.max_omega - self.min_omega) * (current_step**2 / self.budget**2)
        else:
            return self.max_omega - (self.max_omega - self.min_omega) * (current_step / self.budget)
