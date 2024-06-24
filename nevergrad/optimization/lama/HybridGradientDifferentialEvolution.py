import numpy as np


class HybridGradientDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality set to 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Configuration
        population_size = 100
        mutation_factor = 0.8
        crossover_prob = 0.7
        learning_rate = 0.01  # Initial learning rate for gradient steps

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_value = fitness[best_idx]

        num_iterations = self.budget // population_size
        grad_steps = max(1, num_iterations // 20)  # Allocate some iterations for gradient descent

        for iteration in range(num_iterations):
            if iteration < num_iterations - grad_steps:
                # Differential Evolution Strategy
                for i in range(population_size):
                    idxs = [idx for idx in range(population_size) if idx != i]
                    a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                    mutant = a + mutation_factor * (b - c)
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                    trial = np.where(np.random.rand(self.dim) < crossover_prob, mutant, population[i])
                    trial_fitness = func(trial)

                    if trial_fitness < fitness[i]:
                        population[i] = trial
                        fitness[i] = trial_fitness

                        if trial_fitness < best_value:
                            best_value = trial_fitness
                            best_solution = trial.copy()
            else:
                # Gradient-based refinement
                gradients = np.random.randn(population_size, self.dim)  # Mock gradient
                for i in range(population_size):
                    population[i] -= learning_rate * gradients[i]
                    population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)
                    new_fitness = func(population[i])
                    if new_fitness < fitness[i]:
                        fitness[i] = new_fitness
                        if new_fitness < best_value:
                            best_value = new_fitness
                            best_solution = population[i].copy()
                learning_rate *= 0.9  # Decay learning rate

        return best_value, best_solution
