import numpy as np


class AdaptiveEnhancedGradientGuidedHybridPSO:
    def __init__(
        self,
        budget=10000,
        population_size=50,
        initial_inertia=0.9,
        final_inertia=0.4,
        cognitive_weight=2.0,
        social_weight=1.8,
    ):
        self.budget = budget
        self.population_size = population_size
        self.inertia_weight = initial_inertia
        self.final_inertia = final_inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.dim = 5  # Problem dimensionality
        self.lb, self.ub = -5.0, 5.0  # Boundary limits
        self.evolution_rate = (initial_inertia - final_inertia) / budget

    def __call__(self, func):
        particles = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.zeros_like(particles)
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([func(p) for p in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        evaluation_counter = self.population_size
        while evaluation_counter < self.budget:
            self.inertia_weight = max(
                self.inertia_weight - self.evolution_rate, self.final_inertia
            )  # Adaptive inertia weight
            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                # Adaptive gradient-guided component
                gradient_guided_component = (
                    0.1
                    * (global_best_position - particles[i])
                    / (1 + np.sqrt(np.sum((global_best_position - particles[i]) ** 2)))
                )
                personal_component = r1 * self.cognitive_weight * (personal_best_positions[i] - particles[i])
                social_component = r2 * self.social_weight * (global_best_position - particles[i])
                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + personal_component
                    + social_component
                    + gradient_guided_component
                )  # Adaptive hybridization with gradient direction
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
