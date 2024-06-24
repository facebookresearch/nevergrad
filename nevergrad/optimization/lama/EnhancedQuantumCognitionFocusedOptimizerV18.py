import numpy as np


class EnhancedQuantumCognitionFocusedOptimizerV18:
    def __init__(
        self,
        budget=10000,
        population_size=100,
        inertia_weight=0.9,
        cognitive_coeff=2.5,
        social_coeff=2.5,
        inertia_decay=0.99,
        quantum_jump_rate=0.01,
        quantum_scale=0.008,
        quantum_decay=0.97,
    ):
        self.budget = budget
        self.population_size = population_size
        self.dim = 5  # Dimensionality of the problem
        self.lb, self.ub = -5.0, 5.0  # Bounds of the search space
        self.inertia_weight = inertia_weight
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.inertia_decay = inertia_decay
        self.quantum_jump_rate = quantum_jump_rate
        self.quantum_scale = quantum_scale
        self.quantum_decay = quantum_decay

    def __call__(self, func):
        # Initialize particle positions and velocities
        particles = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.zeros_like(particles)
        personal_bests = particles.copy()
        personal_best_scores = np.array([func(p) for p in particles])
        global_best = personal_bests[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Adaptive Quantum Jump Strategy
                if np.random.rand() < self.quantum_jump_rate:
                    quantum_deviation = np.random.normal(0, self.quantum_scale, self.dim)
                    candidate_position = global_best + quantum_deviation
                else:
                    r1, r2 = np.random.rand(2)
                    velocities[i] = (
                        self.inertia_weight * velocities[i]
                        + self.cognitive_coeff * r1 * (personal_bests[i] - particles[i])
                        + self.social_coeff * r2 * (global_best - particles[i])
                    )
                    candidate_position = particles[i] + velocities[i]

                candidate_position = np.clip(candidate_position, self.lb, self.ub)
                score = func(candidate_position)
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_bests[i] = candidate_position
                    personal_best_scores[i] = score

                    if score < global_best_score:
                        global_best = candidate_position
                        global_best_score = score

            # Adjust decay rates and adapt quantum mechanics dynamically
            self.inertia_weight *= self.inertia_decay
            self.quantum_jump_rate *= self.quantum_decay
            self.quantum_scale *= self.quantum_decay

            if evaluations >= self.budget:
                break

        return global_best_score, global_best
