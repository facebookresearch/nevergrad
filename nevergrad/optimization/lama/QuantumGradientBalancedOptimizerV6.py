import numpy as np


class QuantumGradientBalancedOptimizerV6:
    def __init__(
        self,
        budget=10000,
        population_size=100,
        inertia_weight=0.7,
        cognitive_coefficient=2.5,
        social_coefficient=2.5,
        quantum_probability=0.15,
        damping_factor=0.99,
        quantum_scale=0.1,
    ):
        self.budget = budget
        self.population_size = population_size
        self.inertia_weight = inertia_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient
        self.quantum_probability = quantum_probability
        self.damping_factor = damping_factor
        self.quantum_scale = quantum_scale
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

                # Adaptively update inertia weight for enhanced convergence
                inertia = self.inertia_weight * (self.damping_factor ** (evaluations / self.budget))

                velocities[i] = (
                    inertia * velocities[i]
                    + self.cognitive_coefficient * r1 * (personal_bests[i] - particles[i])
                    + self.social_coefficient * r2 * (global_best - particles[i])
                )

                if np.random.rand() < self.quantum_probability:
                    # Quantum leap with scaling factor to control the step size
                    quantum_leap = global_best + np.random.normal(0, self.quantum_scale, self.dim)
                    particles[i] = np.clip(quantum_leap, self.lb, self.ub)
                else:
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

            # Dynamically adjust quantum probability and scale for balanced exploration
            self.quantum_probability *= self.damping_factor
            self.quantum_scale *= self.damping_factor

            if evaluations >= self.budget:
                break

        return global_best_score, global_best
