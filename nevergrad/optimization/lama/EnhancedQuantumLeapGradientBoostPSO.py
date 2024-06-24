import numpy as np


class EnhancedQuantumLeapGradientBoostPSO:
    def __init__(
        self,
        budget=10000,
        population_size=100,
        initial_inertia=0.9,
        final_inertia=0.4,
        cognitive_weight=1.8,
        social_weight=2.2,
        quantum_prob=0.2,
        gradient_boost=0.25,
        adaptative_gradient=0.15,
        quantum_radius_factor=1.0,
    ):
        self.budget = budget
        self.population_size = population_size
        self.initial_inertia = initial_inertia
        self.final_inertia = final_inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.quantum_prob = quantum_prob
        self.gradient_boost = gradient_boost
        self.adaptative_gradient = adaptative_gradient
        self.quantum_radius_factor = quantum_radius_factor
        self.dim = 5
        self.lb, self.ub = -5.0, 5.0

    def __call__(self, func):
        particles = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.zeros_like(particles)
        personal_bests = particles.copy()
        personal_best_scores = np.array([func(p) for p in particles])
        global_best = personal_bests[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        evaluations = self.population_size
        inertia_reduction = (self.initial_inertia - self.final_inertia) / self.budget

        while evaluations < self.budget:
            inertia = max(self.initial_inertia - evaluations * inertia_reduction, self.final_inertia)

            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)

                cognitive_component = self.cognitive_weight * r1 * (personal_bests[i] - particles[i])
                social_component = self.social_weight * r2 * (global_best - particles[i])
                gradient_component = (
                    self.adaptative_gradient
                    * (global_best - particles[i])
                    / (np.linalg.norm(global_best - particles[i]) + 1e-10)
                )

                velocities[i] = (
                    inertia * velocities[i]
                    + cognitive_component
                    + social_component
                    + gradient_component * self.gradient_boost
                )

                if np.random.rand() < self.quantum_prob:
                    quantum_radius = self.quantum_radius_factor * np.linalg.norm(
                        global_best - personal_bests[i]
                    )
                    quantum_jump = np.random.normal(0, quantum_radius, self.dim)
                    particles[i] = global_best + quantum_jump

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
