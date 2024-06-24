import numpy as np


class QuantumInformedAdaptiveInertiaOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        current_budget = 0
        population_size = 40  # Adjusted population size
        inertia_weight = 0.9  # Initial inertia
        cognitive_coefficient = 1.5  # Scaled down personal learning effect
        social_coefficient = 1.5  # Scaled down social influence
        quantum_probability = 0.15  # Adjusted probability of quantum jumps

        # Initialize population and velocities
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        velocity = np.zeros((population_size, self.dim))
        personal_best_position = np.copy(population)
        personal_best_fitness = np.array([func(ind) for ind in population])
        current_budget += population_size

        global_best_position = personal_best_position[np.argmin(personal_best_fitness)]
        global_best_fitness = np.min(personal_best_fitness)

        # Main optimization loop
        while current_budget < self.budget:
            w = inertia_weight * (1 - (current_budget / self.budget) ** 2)  # Adaptive inertia weight

            for i in range(population_size):
                if current_budget >= self.budget:
                    break

                if np.random.rand() < quantum_probability:
                    # Quantum jump strategy
                    population[i] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                else:
                    # Standard PSO update
                    inertia = w * velocity[i]
                    cognitive_component = (
                        cognitive_coefficient
                        * np.random.rand(self.dim)
                        * (personal_best_position[i] - population[i])
                    )
                    social_component = (
                        social_coefficient * np.random.rand(self.dim) * (global_best_position - population[i])
                    )
                    velocity[i] = inertia + cognitive_component + social_component

                population[i] = np.clip(population[i] + velocity[i], self.lower_bound, self.upper_bound)

                # Function evaluation
                fitness = func(population[i])
                current_budget += 1

                # Personal best update
                if fitness < personal_best_fitness[i]:
                    personal_best_position[i] = population[i]
                    personal_best_fitness[i] = fitness

                    # Global best update
                    if fitness < global_best_fitness:
                        global_best_position = population[i]
                        global_best_fitness = fitness

        return global_best_fitness, global_best_position
