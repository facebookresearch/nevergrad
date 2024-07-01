import numpy as np


class EnhancedQuantumAdaptiveOptimizer:
    def __init__(
        self,
        budget=10000,
        population_size=60,
        inertia_weight=0.8,
        cognitive_coef=1.5,
        social_coef=1.7,
        quantum_probability=0.15,
        damping_factor=0.99,
    ):
        self.budget = budget
        self.population_size = population_size
        self.inertia_weight = inertia_weight
        self.cognitive_coef = cognitive_coef
        self.social_coef = social_coef
        self.quantum_probability = quantum_probability
        self.damping_factor = damping_factor
        self.dim = 5
        self.lb, self.ub = -5.0, 5.0

    def __call__(self, func):
        particles = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_bests = particles.copy()
        personal_best_scores = np.array([func(p) for p in particles])
        global_best = personal_bests[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)

                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_coef * r1 * (personal_bests[i] - particles[i])
                    + self.social_coef * r2 * (global_best - particles[i])
                )

                # Quantum Leap Mechanism
                if np.random.rand() < self.quantum_probability:
                    quantum_leap = global_best + np.random.normal(0, 1, self.dim) * (
                        global_best - personal_bests[i]
                    )
                    particles[i] = quantum_leap
                else:
                    particles[i] += velocities[i]

                # Boundary handling
                particles[i] = np.clip(particles[i], self.lb, self.ub)

                # Evaluate
                score = func(particles[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_bests[i] = particles[i]
                    personal_best_scores[i] = score

                    if score < global_best_score:
                        global_best = particles[i]
                        global_best_score = score

            self.inertia_weight *= self.damping_factor
            self.quantum_probability *= 1.05

            if evaluations >= self.budget:
                break

        return global_best_score, global_best
