import numpy as np


class RefinedPrecisionAdaptivePSO:
    def __init__(
        self, budget=10000, population_size=200, inertia_weight=0.9, cognitive_weight=0.5, social_weight=0.5
    ):
        self.budget = budget
        self.population_size = population_size
        self.inertia_weight = inertia_weight  # How much to weigh the previous velocity
        self.cognitive_weight = cognitive_weight  # How much to consider personal best
        self.social_weight = social_weight  # How much to consider global best
        self.dim = 5  # Problem dimensionality
        self.lb, self.ub = -5.0, 5.0  # Boundary limits of the search space

    def __call__(self, func):
        particles = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.zeros_like(particles)
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([func(p) for p in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        evaluation_counter = self.population_size

        while evaluation_counter < self.budget:
            for i in range(self.population_size):
                personal_component = np.random.rand(self.dim) * (personal_best_positions[i] - particles[i])
                social_component = np.random.rand(self.dim) * (global_best_position - particles[i])
                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_weight * personal_component
                    + self.social_weight * social_component
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
