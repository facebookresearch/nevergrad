import numpy as np


class QuantumRefinedAdaptiveStrategicOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed problem dimensionality
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        current_budget = 0
        population_size = 300  # Further increased population size for more diversity
        inertia_weight = 0.75  # Further reduction in initial inertia for quicker exploitation
        cognitive_coefficient = 2.1  # Slightly increased cognitive learning factor
        social_coefficient = 2.1  # Slightly increased social learning factor
        quantum_momentum = 0.25  # Increased quantum influence for better global search
        exploration_factor = 0.6  # Adjusted control parameter for exploration phase duration

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
            # Adaptive inertia weight adjustment for improved strategic balance
            w = inertia_weight * (
                0.3 + 0.7 * np.exp(-4 * current_budget / (self.budget * exploration_factor))
            )

            for i in range(population_size):
                if current_budget >= self.budget:
                    break

                # Quantum jump dynamics with adaptive momentum
                if np.random.rand() < 0.1 * (1 - w):
                    quantum_jump = np.random.normal(0, quantum_momentum * (1 - w), self.dim)
                    population[i] += quantum_jump

                # Update velocities and positions with refined strategic constraints
                inertia_component = w * velocity[i]
                cognitive_component = (
                    cognitive_coefficient
                    * np.random.rand(self.dim)
                    * (personal_best_position[i] - population[i])
                )
                social_component = (
                    social_coefficient * np.random.rand(self.dim) * (global_best_position - population[i])
                )
                velocity[i] = inertia_component + cognitive_component + social_component
                velocity[i] = np.clip(velocity[i], -2, 2)  # Adjusted clamping on velocities
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
