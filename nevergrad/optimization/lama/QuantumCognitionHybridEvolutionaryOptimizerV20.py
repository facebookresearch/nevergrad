import numpy as np


class QuantumCognitionHybridEvolutionaryOptimizerV20:
    def __init__(
        self,
        budget=10000,
        population_size=100,
        inertia_weight=0.9,
        cognitive_coeff=2.1,
        social_coeff=2.1,
        inertia_decay=0.99,
        quantum_jump_rate=0.05,
        quantum_scale=0.015,
        quantum_decay=0.97,
        mutation_rate=0.05,
        mutation_scale=0.1,
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
        best_particles = particles.copy()
        best_values = np.array([func(p) for p in particles])
        global_best = best_particles[np.argmin(best_values)]
        global_best_value = min(best_values)

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                if np.random.rand() < self.quantum_jump_rate:
                    quantum_deviation = np.random.normal(
                        0, self.quantum_scale * (self.ub - self.lb), self.dim
                    )
                    candidate_position = global_best + quantum_deviation
                else:
                    r1, r2 = np.random.rand(2)
                    velocities[i] = (
                        self.inertia_weight * velocities[i]
                        + self.cognitive_coeff * r1 * (best_particles[i] - particles[i])
                        + self.social_coeff * r2 * (global_best - particles[i])
                    )
                    candidate_position = particles[i] + velocities[i]

                # Mutation mechanism
                if np.random.rand() < self.mutation_rate:
                    mutation = np.random.normal(0, self.mutation_scale, self.dim)
                    candidate_position += mutation

                candidate_position = np.clip(candidate_position, self.lb, self.ub)
                score = func(candidate_position)
                evaluations += 1

                if score < best_values[i]:
                    best_particles[i] = candidate_position
                    best_values[i] = score

                    if score < global_best_value:
                        global_best = candidate_position
                        global_best_value = score

            self.inertia_weight *= self.inertia_decay
            self.quantum_jump_rate *= self.quantum_decay
            self.quantum_scale *= self.quantum_decay

            if evaluations >= self.budget:
                break

        return global_best_value, global_best
