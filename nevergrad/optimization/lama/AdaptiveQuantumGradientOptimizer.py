import numpy as np


class AdaptiveQuantumGradientOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is constant at 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        current_budget = 0
        population_size = 30  # Smaller population size to increase individual scrutiny
        mutation_rate = 0.2  # Initial mutation rate
        crossover_rate = 0.7  # Initial crossover rate
        gradient_step = 0.01  # Step size for gradient estimation

        # Initialize population within bounds
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        current_budget += population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        # Optimization loop
        while current_budget < self.budget:
            new_population = population.copy()

            for i in range(population_size):
                if current_budget >= self.budget:
                    break

                # Quantum mutation
                quantum_perturbation = np.random.normal(0, mutation_rate, self.dim)
                child = population[i] + quantum_perturbation

                # Gradient-based refinement
                grad = np.zeros(self.dim)
                for d in range(self.dim):
                    if current_budget + 2 > self.budget:
                        break
                    plus = np.array(population[i])
                    minus = np.array(population[i])
                    plus[d] += gradient_step
                    minus[d] -= gradient_step

                    f_plus = func(plus)
                    f_minus = func(minus)
                    current_budget += 2

                    grad[d] = (f_plus - f_minus) / (2 * gradient_step)

                child -= grad * gradient_step  # Move against the gradient

                # Crossover
                if np.random.rand() < crossover_rate:
                    partner_idx = np.random.randint(population_size)
                    for j in range(self.dim):
                        if np.random.rand() < 0.5:
                            child[j] = population[partner_idx][j]

                child = np.clip(child, self.lower_bound, self.upper_bound)
                child_fitness = func(child)
                current_budget += 1

                # Selection
                if child_fitness < fitness[i]:
                    new_population[i] = child
                    fitness[i] = child_fitness

                if child_fitness < best_fitness:
                    best_fitness = child_fitness
                    best_solution = child

            population = new_population
            mutation_rate *= 0.99  # Gradual decrease in mutation rate
            crossover_rate *= 0.99  # Gradual decrease in crossover rate

        return best_fitness, best_solution
