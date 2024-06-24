import numpy as np


class AdaptiveExploratoryOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population_size = 50  # Reduced for more focused search
        mutation_factor = 1.0  # Increased initial mutation for broader initial exploration
        crossover_rate = 0.8  # Slightly reduced crossover for preserving diversity
        elite_size = 2  # More focused elitism
        learning_period = max(100, self.budget // 100)  # Introduce a learning period for adaptation

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(x) for x in population])
        evaluations = population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        # Main loop
        while evaluations < self.budget:
            new_population = np.empty_like(population)
            new_fitness = np.empty_like(fitness)

            # Elitism: carry forward best solutions
            sorted_indices = np.argsort(fitness)[:elite_size]
            new_population[:elite_size] = population[sorted_indices]
            new_fitness[:elite_size] = fitness[sorted_indices]

            # Generate new candidates for the rest of the population
            for i in range(elite_size, population_size):
                # Differential Evolution Strategy: "best/1/bin" for more exploitation
                best = population[best_index]
                idxs = [idx for idx in range(population_size) if idx != best_index]
                x1, x2 = population[np.random.choice(idxs, 2, replace=False)]

                # Mutation
                mutant = best + mutation_factor * (x1 - x2)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < crossover_rate
                trial = np.where(crossover_mask, mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = trial_fitness
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]

            population = new_population
            fitness = new_fitness

            # Update the best solution found
            current_best_index = np.argmin(fitness)
            current_best_fitness = fitness[current_best_index]
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_solution = population[current_best_index]
                best_index = current_best_index

            # Adaptive strategies
            if evaluations % learning_period == 0:
                mutation_factor *= 0.98  # Gradually reduce mutation factor for finer exploitation
                crossover_rate = min(
                    0.95, crossover_rate + 0.02
                )  # Increase crossover rate for better exploration

        return best_fitness, best_solution
