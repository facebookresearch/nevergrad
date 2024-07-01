import numpy as np


class RefinedQuantumEvolutionaryAdaptiveOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initial setup
        current_budget = 0
        population_size = 50  # Increased population for better exploration
        mutation_factor = 0.8  # Less aggressive mutation for stability
        crossover_prob = 0.7  # Lower crossover probability for maintaining diversity

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        current_budget += population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        # Modifying quantum behavior for refined control
        quantum_beta = 0.6  # Reduced quantum randomness
        quantum_alpha = 0.01  # Smaller quantum step size

        quantum_population = quantum_beta * np.random.randn(population_size, self.dim) + population
        quantum_population = np.clip(quantum_population, self.lower_bound, self.upper_bound)

        while current_budget < self.budget:
            new_population = np.empty_like(population)
            for i in range(population_size):
                if current_budget >= self.budget:
                    break

                indices = np.delete(np.arange(population_size), i)
                random_indices = np.random.choice(indices, 3, replace=False)
                x1, x2, x3 = population[random_indices]
                q1, q2, q3 = quantum_population[random_indices]

                mutant = x1 + mutation_factor * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                quantum_mutant = q1 + quantum_alpha * (q2 - q3)
                quantum_mutant = np.clip(quantum_mutant, self.lower_bound, self.upper_bound)

                trial = np.where(np.random.rand(self.dim) < crossover_prob, mutant, population[i])
                quantum_trial = np.where(
                    np.random.rand(self.dim) < crossover_prob, quantum_mutant, quantum_population[i]
                )

                trial_fitness = func(trial)
                quantum_trial_fitness = func(quantum_trial)
                current_budget += 2  # Two function evaluations per iteration

                if quantum_trial_fitness < trial_fitness:
                    trial_fitness = quantum_trial_fitness
                    trial = quantum_trial

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial
                else:
                    new_population[i] = population[i]

            population = new_population

            # Dynamic adaptation of mutation factor and crossover probability
            mutation_factor *= 0.985  # Gradual decrease to fine-tune exploration
            crossover_prob *= 1.015  # Incremental increase to enhance exploring crossover possibilities
            quantum_alpha *= 0.97  # Reduce quantum steps gradually, focusing on exploitation

        return best_fitness, best_solution
