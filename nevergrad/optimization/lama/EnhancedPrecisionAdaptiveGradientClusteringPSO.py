import numpy as np


class EnhancedPrecisionAdaptiveGradientClusteringPSO:
    def __init__(
        self,
        budget=10000,
        population_size=150,
        initial_inertia=0.9,
        final_inertia=0.4,
        cognitive_weight=2.2,
        social_weight=1.5,
        cluster_factor=0.1,
        adaptation_rate=0.1,
    ):
        self.budget = budget
        self.population_size = population_size
        self.initial_inertia = initial_inertia
        self.final_inertia = final_inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.cluster_factor = cluster_factor
        self.adaptation_rate = adaptation_rate
        self.dim = 5
        self.lb, self.ub = -5.0, 5.0
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
            cluster_center = np.mean(particles, axis=0)

            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                exploration_component = (
                    (np.random.rand(self.dim) - 0.5) * 2 * self.cluster_factor * (self.ub - self.lb)
                )  # Random exploratory moves
                personal_component = r1 * self.cognitive_weight * (personal_best_positions[i] - particles[i])
                social_component = r2 * self.social_weight * (global_best_position - particles[i])
                cluster_component = self.cluster_factor * (cluster_center - particles[i])  # Cluster force

                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + personal_component
                    + social_component
                    + cluster_component
                    + exploration_component
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

            # Adaptive clustering adjustment
            if evaluation_counter % (self.budget // 10) == 0:
                self.cluster_factor *= 1 - self.adaptation_rate  # Gradually reduce clustering influence

        return global_best_score, global_best_position
