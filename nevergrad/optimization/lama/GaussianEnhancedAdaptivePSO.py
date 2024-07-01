import numpy as np


class GaussianEnhancedAdaptivePSO:
    def __init__(
        self,
        budget=10000,
        population_size=250,
        initial_inertia=0.95,
        final_inertia=0.35,
        cognitive_weight=2.1,
        social_weight=2.1,
        mutate_prob=0.15,
        mutate_scale=0.03,
        gradient_weight=0.15,
    ):
        self.budget = budget
        self.population_size = population_size
        self.initial_inertia = initial_inertia
        self.final_inertia = final_inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.mutate_prob = mutate_prob
        self.mutate_scale = mutate_scale
        self.gradient_weight = gradient_weight
        self.dim = 5
        self.lb, self.ub = -5.0, 5.0
        self.inertia_reduction = (self.initial_inertia - self.final_inertia) / budget

    def __call__(self, func):
        particles = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.zeros_like(particles)
        personal_bests = particles.copy()
        personal_best_scores = np.array([func(p) for p in particles])
        global_best = personal_bests[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        evaluations = self.population_size

        while evaluations < self.budget:
            inertia = max(self.initial_inertia - evaluations * self.inertia_reduction, self.final_inertia)

            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)

                cognitive_component = self.cognitive_weight * r1 * (personal_bests[i] - particles[i])
                social_component = self.social_weight * r2 * (global_best - particles[i])
                gradient_component = (
                    self.gradient_weight
                    * (global_best - particles[i])
                    / (np.linalg.norm(global_best - particles[i]) + 1e-10)
                )

                velocities[i] = (
                    inertia * velocities[i] + cognitive_component + social_component + gradient_component
                )

                if np.random.rand() < self.mutate_prob:
                    velocities[i] += np.random.normal(0, self.mutate_scale, self.dim)

                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.lb, self.ub)

                score = func(particles[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_bests[i] = particles[i]
                    personal_best_scores[i] = score

                    if score < global_best_score:
                        global_best = particles[i]
                        global_best_score = score

                if evaluations >= self.budget:
                    break

        return global_best_score, global_best
