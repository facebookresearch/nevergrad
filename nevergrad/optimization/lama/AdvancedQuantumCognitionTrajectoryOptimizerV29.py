import numpy as np


class AdvancedQuantumCognitionTrajectoryOptimizerV29:
    def __init__(
        self,
        budget=10000,
        population_size=500,
        inertia_weight=0.95,
        cognitive_coeff=2.0,
        social_coeff=2.0,
        inertia_decay=0.99,
        quantum_jump_rate=0.5,
        quantum_scale=0.35,
        quantum_decay=0.95,
        mutation_rate=0.02,
        mutation_scale=0.05,
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
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale

    def __call__(self, func):
        # Initialize particle positions and velocities
        particles = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.zeros_like(particles)
        best_individual_positions = particles.copy()
        best_individual_scores = np.array([func(p) for p in particles])
        global_best_position = best_individual_positions[np.argmin(best_individual_scores)]
        global_best_score = min(best_individual_scores)

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                if np.random.rand() < self.quantum_jump_rate:
                    # Perform a quantum jump for global exploration
                    quantum_deviation = np.random.normal(
                        0, self.quantum_scale * (self.ub - self.lb), self.dim
                    )
                    candidate_position = global_best_position + quantum_deviation
                else:
                    r1, r2 = np.random.rand(2)
                    velocities[i] = (
                        self.inertia_weight * velocities[i]
                        + self.cognitive_coeff * r1 * (best_individual_positions[i] - particles[i])
                        + self.social_coeff * r2 * (global_best_position - particles[i])
                    )
                    candidate_position = particles[i] + velocities[i]

                # Mutation for enhanced local exploration
                if np.random.rand() < self.mutation_rate:
                    mutation = np.random.normal(0, self.mutation_scale, self.dim)
                    candidate_position += mutation

                candidate_position = np.clip(candidate_position, self.lb, self.ub)
                score = func(candidate_position)
                evaluations += 1

                if score < best_individual_scores[i]:
                    best_individual_positions[i] = candidate_position
                    best_individual_scores[i] = score

                    if score < global_best_score:
                        global_best_position = candidate_position
                        global_best_score = score

            self.inertia_weight *= self.inertia_decay
            self.quantum_jump_rate *= self.quantum_decay
            self.quantum_scale *= self.quantum_decay

            if evaluations >= self.budget:
                break

        return global_best_score, global_best_position
