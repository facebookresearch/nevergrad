import numpy as np


class RefinedGradientBalancedExplorationPSO:
    def __init__(
        self,
        budget=10000,
        population_size=50,
        initial_inertia=0.95,
        final_inertia=0.5,
        cognitive_weight=2.0,
        social_weight=1.8,
        exploration_factor=0.1,
    ):
        self.budget = budget
        self.population_size = population_size
        self.initial_inertia = initial_inertia
        self.final_inertia = final_inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.exploration_factor = exploration_factor
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
                self.initial_inertia - (self.evolution_rate * evaluation_counter), self.final_inertia
            )
            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                personal_component = r1 * self.cognitive_weight * (personal_best_positions[i] - particles[i])
                social_component = r2 * self.social_weight * (global_best_position - particles[i])
                random_exploration = self.exploration_factor * np.random.normal(
                    0, 1, self.dim
                )  # Random exploration factor

                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + personal_component
                    + social_component
                    + random_exploration
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
