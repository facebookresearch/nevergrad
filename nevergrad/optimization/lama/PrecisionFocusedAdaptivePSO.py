import numpy as np


class PrecisionFocusedAdaptivePSO:
    def __init__(
        self,
        budget=10000,
        population_size=250,
        initial_inertia=0.9,
        final_inertia=0.4,
        cognitive_weight=2.2,
        social_weight=2.0,
        elite_ratio=0.1,
        mutation_intensity=0.1,
    ):
        self.budget = budget
        self.population_size = population_size
        self.initial_inertia = initial_inertia
        self.final_inertia = final_inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.elite_ratio = elite_ratio
        self.mutation_intensity = mutation_intensity
        self.dim = 5
        self.lb, self.ub = -5.0, 5.0
        self.evolution_rate = (self.initial_inertia - self.final_inertia) / budget

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
            elite_size = int(self.population_size * self.elite_ratio)
            elite_indices = np.argsort(personal_best_scores)[:elite_size]

            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                personal_component = r1 * self.cognitive_weight * (personal_best_positions[i] - particles[i])
                social_component = r2 * self.social_weight * (global_best_position - particles[i])
                velocities[i] = self.inertia_weight * velocities[i] + personal_component + social_component
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.lb, self.ub)

                if i in elite_indices:
                    # Elite particles undergo a focused mutation
                    mutation_indices = np.random.choice(
                        self.dim, size=int(np.ceil(self.dim * 0.5)), replace=False
                    )
                    particles[i][mutation_indices] += np.random.normal(
                        0, self.mutation_intensity, size=len(mutation_indices)
                    )

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
