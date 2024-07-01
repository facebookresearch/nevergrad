import numpy as np


class RefinedQuantumAdaptiveOptimizerV2:
    def __init__(
        self,
        budget=10000,
        population_size=80,
        inertia_weight=0.9,
        cognitive_coef=2.0,
        social_coef=2.0,
        quantum_probability=0.10,
        damping_factor=0.98,
        adaptive_quantum_shift=0.01,
        elite_strategy=True,
    ):
        self.budget = budget
        self.population_size = population_size
        self.inertia_weight = inertia_weight
        self.cognitive_coef = cognitive_coef
        self.social_coef = social_coef
        self.quantum_probability = quantum_probability
        self.damping_factor = damping_factor
        self.adaptive_quantum_shift = adaptive_quantum_shift
        self.elite_strategy = elite_strategy
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

                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_coef * r1 * (personal_bests[i] - particles[i])
                    + self.social_coef * r2 * (global_best - particles[i])
                )

                if np.random.rand() < self.quantum_probability:
                    # Enhanced Quantum movement
                    quantum_leap = global_best + np.random.normal(0, 1, self.dim) * (
                        global_best - personal_bests[i]
                    )
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

            self.inertia_weight *= self.damping_factor
            self.quantum_probability += self.adaptive_quantum_shift

            # Elite strategy: include random re-initialization of worst performers
            if self.elite_strategy:
                worst_indices = np.argsort(-personal_best_scores)[: self.population_size // 10]
                for idx in worst_indices:
                    particles[idx] = np.random.uniform(self.lb, self.ub, self.dim)
                    velocities[idx] = np.zeros(self.dim)
                    personal_best_scores[idx] = func(particles[idx])
                    personal_bests[idx] = particles[idx]
                    evaluations += 1

            if evaluations >= self.budget:
                break

        return global_best_score, global_best
