import numpy as np


class EnhancedQuantumDynamicOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Problem dimensionality
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        current_budget = 0
        population_size = 50  # Increased population for better exploration
        inertia_weight = 0.9  # Higher inertia for initial global exploration
        cognitive_coefficient = 2.1  # Slightly reduced to prevent premature convergence
        social_coefficient = 2.1  # Balanced to enhance information sharing
        velocity_limit = 0.2  # Increased for faster coverage of the search space
        quantum_momentum = 0.02  # Increased momentum for enhanced quantum jumps

        # Initialize population and velocities
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        velocity = np.zeros((population_size, self.dim))
        personal_best_position = np.copy(population)
        personal_best_fitness = np.array([func(ind) for ind in population])
        current_budget += population_size

        global_best_position = personal_best_position[np.argmin(personal_best_fitness)]
        global_best_fitness = np.min(personal_best_fitness)

        # Main loop
        while current_budget < self.budget:
            w = inertia_weight * (
                0.99 ** (current_budget / self.budget)
            )  # Dynamic weight decay for fine-tuned exploration-to-exploitation shift

            for i in range(population_size):
                if current_budget >= self.budget:
                    break

                # Quantum jump with dynamically decreasing probability
                quantum_probability = 0.05 * np.exp(-10 * (current_budget / self.budget))
                if np.random.rand() < quantum_probability:
                    quantum_jump = np.random.normal(0, quantum_momentum, self.dim)
                    population[i] += quantum_jump

                # Update velocities and positions using PSO rules
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

                # Function evaluation
                fitness = func(population[i])
                current_budget += 1

                # Update personal and global bests
                if fitness < personal_best_fitness[i]:
                    personal_best_position[i] = population[i]
                    personal_best_fitness[i] = fitness

                if fitness < global_best_fitness:
                    global_best_position = population[i]
                    global_best_fitness = fitness

        return global_best_fitness, global_best_position
