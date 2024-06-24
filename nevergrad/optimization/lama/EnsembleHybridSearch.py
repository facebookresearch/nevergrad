import numpy as np


class EnsembleHybridSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        population_size = 30
        differential_weight = 0.8
        crossover_rate = 0.9
        inertia_weight = 0.7
        cognitive_coefficient = 1.5
        social_coefficient = 1.5

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        velocity = np.random.uniform(-1, 1, (population_size, self.dim))

        personal_best_positions = np.copy(population)
        personal_best_fitness = np.copy(fitness)

        global_best_position = population[np.argmin(fitness)]
        global_best_fitness = np.min(fitness)

        self.f_opt = global_best_fitness
        self.x_opt = global_best_position

        evaluations = population_size

        while evaluations < self.budget:
            for i in range(population_size):
                # Particle Swarm Optimization Part
                inertia = inertia_weight * velocity[i]
                cognitive = (
                    cognitive_coefficient
                    * np.random.rand(self.dim)
                    * (personal_best_positions[i] - population[i])
                )
                social = (
                    social_coefficient * np.random.rand(self.dim) * (global_best_position - population[i])
                )
                velocity[i] = inertia + cognitive + social
                population[i] = np.clip(population[i] + velocity[i], self.lb, self.ub)

                # Differential Evolution Part
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant_vector = np.clip(a + differential_weight * (b - c), self.lb, self.ub)

                crossover_mask = np.random.rand(self.dim) < crossover_rate
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True

                trial_vector = np.where(crossover_mask, mutant_vector, population[i])

                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

                if trial_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_fitness[i] = trial_fitness

                if trial_fitness < self.f_opt:
                    self.f_opt = trial_fitness
                    self.x_opt = trial_vector

            global_best_position = population[np.argmin(fitness)]
            global_best_fitness = np.min(fitness)

        return self.f_opt, self.x_opt


# Example usage:
# def sample_func(x):
#     return np.sum(x**2)

# optimizer = EnsembleHybridSearch(budget=10000)
# best_fitness, best_solution = optimizer(sample_func)
# print("Best fitness:", best_fitness)
# print("Best solution:", best_solution)
