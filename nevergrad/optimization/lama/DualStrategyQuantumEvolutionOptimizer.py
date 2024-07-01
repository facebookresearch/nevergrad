import numpy as np


class DualStrategyQuantumEvolutionOptimizer:
    def __init__(
        self,
        budget=10000,
        population_size=100,
        inertia_weight=0.5,
        cognitive_coefficient=2.1,
        social_coefficient=2.1,
        adaptive_decay=0.99,
        quantum_probability=0.1,
        quantum_scale=0.05,
    ):
        self.budget = budget
        self.population_size = population_size
        self.dim = 5
        self.lb, self.ub = -5.0, 5.0
        self.inertia_weight = inertia_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient
        self.adaptive_decay = adaptive_decay
        self.quantum_probability = quantum_probability
        self.quantum_scale = quantum_scale

    def __call__(self, func):
        particles = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.zeros_like(particles)
        personal_bests = particles.copy()
        personal_best_scores = np.array([func(p) for p in particles])
        global_best = personal_bests[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                if np.random.rand() < self.quantum_probability:
                    # Quantum jump strategy for exploration
                    particles[i] = global_best + np.random.normal(0, self.quantum_scale, self.dim)
                    particles[i] = np.clip(particles[i], self.lb, self.ub)
                else:
                    # Classical PSO movement for exploitation
                    r1, r2 = np.random.rand(2)
                    velocities[i] = (
                        self.inertia_weight * velocities[i]
                        + self.cognitive_coefficient * r1 * (personal_bests[i] - particles[i])
                        + self.social_coefficient * r2 * (global_best - particles[i])
                    )
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

            # Adaptive updates to strategy parameters
            self.quantum_probability *= self.adaptive_decay
            self.quantum_scale *= self.adaptive_decay
            self.inertia_weight *= self.adaptive_decay

            if evaluations >= self.budget:
                break

        return global_best_score, global_best
