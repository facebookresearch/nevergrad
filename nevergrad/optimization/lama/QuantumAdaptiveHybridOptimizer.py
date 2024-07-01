import numpy as np


class QuantumAdaptiveHybridOptimizer:
    def __init__(
        self,
        budget=10000,
        population_size=50,
        inertia_weight=0.7,
        cognitive_coef=1.4,
        social_coef=1.6,
        adaptive_intensity=0.05,
        quantum_rate=0.1,
    ):
        self.budget = budget
        self.population_size = population_size
        self.inertia_weight = inertia_weight
        self.cognitive_coef = cognitive_coef
        self.social_coef = social_coef
        self.adaptive_intensity = adaptive_intensity
        self.quantum_rate = quantum_rate
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

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)

                # Regular PSO update components
                cognitive_component = self.cognitive_coef * r1 * (personal_bests[i] - particles[i])
                social_component = self.social_coef * r2 * (global_best - particles[i])

                # Quantum-inspired leap
                if np.random.rand() < self.quantum_rate:
                    quantum_jump = np.random.randn(self.dim) * np.abs(global_best - personal_bests[i])
                    particles[i] = global_best + quantum_jump
                else:
                    velocities[i] = (
                        self.inertia_weight * velocities[i] + cognitive_component + social_component
                    )
                    particles[i] += velocities[i]

                # Ensure particles stay within bounds
                particles[i] = np.clip(particles[i], self.lb, self.ub)

                # Function evaluation
                score = func(particles[i])
                evaluations += 1

                # Update personal and global bests
                if score < personal_best_scores[i]:
                    personal_bests[i] = particles[i]
                    personal_best_scores[i] = score
                    if score < global_best_score:
                        global_best = particles[i]
                        global_best_score = score

                # Adaptive intensity adjustment
                if evaluations % (self.budget // 10) == 0:
                    self.inertia_weight *= 1 - self.adaptive_intensity
                    self.quantum_rate *= 1 + self.adaptive_intensity

                if evaluations >= self.budget:
                    break

        return global_best_score, global_best
