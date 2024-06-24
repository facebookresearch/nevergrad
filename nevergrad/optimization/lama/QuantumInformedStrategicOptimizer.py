import numpy as np


class QuantumInformedStrategicOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed problem dimensionality
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        current_budget = 0
        population_size = 100  # Enhanced population for broader initial search
        inertia_weight = 0.7  # Optimized for balanced initial exploration
        cognitive_coefficient = 1.5  # Decreased to reduce local premature convergence
        social_coefficient = 1.5  # Decreased for the same reason as above
        velocity_limit = 0.2  # Optimized for controlled explorative moves
        quantum_momentum = 0.05  # Stronger quantum influences for better global search

        # Initialize population, velocities, and personal bests
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        velocity = np.zeros((population_size, self.dim))
        personal_best_position = np.copy(population)
        personal_best_fitness = np.array([func(ind) for ind in population])
        current_budget += population_size

        global_best_position = personal_best_position[np.argmin(personal_best_fitness)]
        global_best_fitness = np.min(personal_best_fitness)

        # Main optimization loop
        while current_budget < self.budget:
            # Dynamic inertia weight adjustment for exploration to exploitation transition
            w = inertia_weight * (0.5 + 0.5 * np.exp(-4 * current_budget / self.budget))

            for i in range(population_size):
                if current_budget >= self.budget:
                    break

                # Quantum jump dynamics incorporated strategically
                if np.random.rand() < 0.02 * np.exp(-10 * current_budget / self.budget):
                    quantum_jump = np.random.normal(0, quantum_momentum, self.dim)
                    population[i] += quantum_jump

                # Update velocities and positions with strategic constraints
                inertia_component = w * velocity[i]
                cognitive_component = (
                    cognitive_coefficient
                    * np.random.rand(self.dim)
                    * (personal_best_position[i] - population[i])
                )
                social_component = (
                    social_coefficient * np.random.rand(self.dim) * (global_best_position - population[i])
                )
                velocity[i] = np.clip(
                    inertia_component + cognitive_component + social_component,
                    -velocity_limit,
                    velocity_limit,
                )
                population[i] = np.clip(population[i] + velocity[i], self.lower_bound, self.upper_bound)

                # Fitness evaluation and personal and global best updates
                fitness = func(population[i])
                current_budget += 1

                if fitness < personal_best_fitness[i]:
                    personal_best_position[i] = population[i]
                    personal_best_fitness[i] = fitness

                if fitness < global_best_fitness:
                    global_best_position = population[i]
                    global_best_fitness = fitness

        return global_best_fitness, global_best_position
