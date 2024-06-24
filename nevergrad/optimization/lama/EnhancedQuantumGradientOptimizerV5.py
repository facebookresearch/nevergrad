import numpy as np


class EnhancedQuantumGradientOptimizerV5:
    def __init__(
        self,
        budget=10000,
        population_size=100,
        inertia_weight=0.8,
        cognitive_coef=2.2,
        social_coef=2.2,
        quantum_probability=0.2,
        damping_factor=0.95,
        adaptive_quantum_shift=0.005,
    ):
        self.budget = budget
        self.population_size = population_size
        self.inertia_weight = inertia_weight
        self.cognitive_coef = cognitive_coef
        self.social_coef = social_coef
        self.quantum_probability = quantum_probability
        self.damping_factor = damping_factor
        self.adaptive_quantum_shift = adaptive_quantum_shift
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

                # Gradual inertia weight reduction for precision
                inertia = self.inertia_weight * (self.damping_factor ** (evaluations / self.budget))

                velocities[i] = (
                    inertia * velocities[i]
                    + self.cognitive_coef * r1 * (personal_bests[i] - particles[i])
                    + self.social_coef * r2 * (global_best - particles[i])
                )

                if np.random.rand() < self.quantum_probability:
                    # Quantum leap using a Gaussian distribution centered at global best
                    quantum_leap = global_best + np.random.normal(0, 1.0 / evaluations**0.5, self.dim)
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

            # Dynamic adjustment for exploration and exploitation balance
            self.quantum_probability += self.adaptive_quantum_shift
            self.inertia_weight *= self.damping_factor

            if evaluations >= self.budget:
                break

        return global_best_score, global_best
