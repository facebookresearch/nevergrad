import numpy as np


class AdvancedQuantumVelocityOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        current_budget = 0
        population_size = 30  # Further optimized population size
        inertia_weight = 0.75  # Further reduction in inertia for better dynamic adaptation
        cognitive_coefficient = 2.5  # Increased cognitive learning for finer individual adaptation
        social_coefficient = 2.5  # Increased social learning for stronger group influence
        velocity_limit = 0.1  # Further reduction in velocity limit for finer control
        quantum_momentum = 0.01  # Further reduced momentum for very subtle quantum jumps

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
            inertia_decay = np.power(
                (1 - (current_budget / self.budget)), 3
            )  # Stronger exponential decay for inertia
            w = inertia_weight * inertia_decay

            for i in range(population_size):
                if current_budget >= self.budget:
                    break

                # Adaptive quantum jump
                quantum_probability = 0.05 * np.exp(
                    -10 * (current_budget / self.budget)
                )  # Smaller probability of quantum jump
                if np.random.rand() < quantum_probability:
                    quantum_jump = np.random.normal(0, quantum_momentum, self.dim)
                    population[i] += quantum_jump

                # PSO velocity updates with clamping
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

                # Personal and global best updates
                if fitness < personal_best_fitness[i]:
                    personal_best_position[i] = population[i]
                    personal_best_fitness[i] = fitness

                if fitness < global_best_fitness:
                    global_best_position = population[i]
                    global_best_fitness = fitness

        return global_best_fitness, global_best_position
