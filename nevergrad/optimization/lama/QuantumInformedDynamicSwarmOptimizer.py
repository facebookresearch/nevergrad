import numpy as np


class QuantumInformedDynamicSwarmOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        current_budget = 0
        population_size = 30  # Reduced population size for effective exploration
        inertia_weight = 0.95  # Initial strong inertia for exploration
        cognitive_coefficient = 2.0  # Enhanced personal learning effect
        social_coefficient = 2.0  # Enhanced social influence
        final_inertia_weight = 0.1  # Sharper final focus
        quantum_probability = 0.1  # Probability of using quantum jumps instead of regular updates

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

                if np.random.rand() < quantum_probability:
                    # Quantum jump: Generate new position with a random quantum leap
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
