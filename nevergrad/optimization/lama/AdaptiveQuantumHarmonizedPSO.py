import numpy as np


class AdaptiveQuantumHarmonizedPSO:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        current_budget = 0
        population_size = 50  # Smaller population size to increase focus
        inertia_weight = 0.9  # High initial inertia for exploration
        cognitive_coefficient = 1.5  # Moderately high for personal learning
        social_coefficient = 1.5  # Moderately high for social influence
        final_inertia_weight = 0.2  # Lower final inertia for sharper focus towards end
        adaptive_cognitive = 0.05  # Incremental increase factor for cognitive component
        adaptive_social = 0.05  # Incremental increase factor for social component

        # Initialize population and velocities
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        velocity = np.zeros((population_size, self.dim))
        personal_best_position = np.copy(population)
        personal_best_fitness = np.array([func(ind) for ind in population])
        current_budget += population_size

        global_best_position = personal_best_position[np.argmin(personal_best_fitness)]
        global_best_fitness = np.min(personal_best_fitness)

        # Optimization loop
        while current_budget < self.budget:
            w = inertia_weight - ((inertia_weight - final_inertia_weight) * (current_budget / self.budget))

            for i in range(population_size):
                if current_budget >= self.budget:
                    break

                # Quantum-inspired stochastic component with dampening factor
                quantum_factor = np.random.normal(0, 0.1, self.dim)  # Dynamic adjustments through noise

                # Update velocity with adaptive increments in cognitive and social coefficients
                inertia = w * velocity[i]
                cognitive_component = (
                    (cognitive_coefficient + adaptive_cognitive * (current_budget / self.budget))
                    * np.random.rand(self.dim)
                    * (personal_best_position[i] - population[i])
                )
                social_component = (
                    (social_coefficient + adaptive_social * (current_budget / self.budget))
                    * np.random.rand(self.dim)
                    * (global_best_position - population[i])
                )
                velocity[i] = inertia + cognitive_component + social_component + quantum_factor

                # Update position
                population[i] += velocity[i]
                population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)

                # Evaluate new position
                fitness = func(population[i])
                current_budget += 1

                # Update personal best
                if fitness < personal_best_fitness[i]:
                    personal_best_position[i] = population[i]
                    personal_best_fitness[i] = fitness

                    # Update global best
                    if fitness < global_best_fitness:
                        global_best_position = population[i]
                        global_best_fitness = fitness

        return global_best_fitness, global_best_position
