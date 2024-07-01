import numpy as np


class AdaptiveQuantumDynamicTuningOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Problem dimensionality
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        current_budget = 0
        population_size = 60  # Slightly increased population size for better initial exploration
        inertia_weight = 0.9  # Initial high inertia for broad exploration
        cognitive_coefficient = 2.05  # Fine-tuned cognitive learning rate
        social_coefficient = 2.05  # Fine-tuned social learning rate
        velocity_limit = 0.25  # Optimized velocity limit for enhanced particle movement
        quantum_momentum = 0.03  # Increased quantum momentum for stronger quantum jumps

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
            # Dynamic decay of inertia weight for smooth transition from exploration to exploitation
            w = inertia_weight * (1 - 2 * (current_budget / self.budget))

            for i in range(population_size):
                if current_budget >= self.budget:
                    break

                # Conditionally apply quantum jumps with dynamically decreasing probability
                quantum_probability = 0.1 * np.exp(-12 * (current_budget / self.budget))
                if np.random.rand() < quantum_probability:
                    quantum_jump = np.random.normal(0, quantum_momentum, self.dim)
                    population[i] += quantum_jump

                # Update velocities and positions using modified PSO dynamics
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

                # Evaluate fitness and update personal and global bests
                fitness = func(population[i])
                current_budget += 1

                if fitness < personal_best_fitness[i]:
                    personal_best_position[i] = population[i]
                    personal_best_fitness[i] = fitness

                if fitness < global_best_fitness:
                    global_best_position = population[i]
                    global_best_fitness = fitness

        return global_best_fitness, global_best_position
