import numpy as np


class DualPhaseAdaptiveGradientEvolution:
    def __init__(self, budget, dim=5, threshold=0.5, initial_phase_ratio=0.7):
        self.budget = budget
        self.dim = dim
        self.threshold = threshold  # Threshold to switch from exploration to exploitation
        self.initial_phase_budget = int(budget * initial_phase_ratio)  # Budget for the exploration phase
        self.exploitation_phase_budget = budget - self.initial_phase_budget
        self.bounds = np.array([-5.0, 5.0])

    def initialize_population(self, size):
        return np.random.uniform(self.bounds[0], self.bounds[1], size=(size, self.dim))

    def evaluate_population(self, func, population):
        return np.array([func(ind) for ind in population])

    def mutate(self, individual, mutation_strength):
        mutant = individual + mutation_strength * np.random.randn(self.dim)
        return np.clip(mutant, self.bounds[0], self.bounds[1])

    def __call__(self, func):
        # Phase 1: Exploration with randomly mutating individuals
        population_size = 20
        mutation_strength = 0.5
        population = self.initialize_population(population_size)
        f_values = self.evaluate_population(func, population)
        evaluations = population_size

        while evaluations < self.initial_phase_budget:
            new_population = []
            for individual in population:
                mutated = self.mutate(individual, mutation_strength)
                new_population.append(mutated)
            new_f_values = self.evaluate_population(func, new_population)
            evaluations += population_size

            # Select the best individuals
            combined_f_values = np.concatenate((f_values, new_f_values))
            combined_population = np.vstack((population, new_population))
            best_indices = np.argsort(combined_f_values)[:population_size]
            population = combined_population[best_indices]
            f_values = combined_f_values[best_indices]

        # Phase 2: Exploitation using gradient descent
        best_individual = population[np.argmin(f_values)]
        best_f_value = np.min(f_values)
        learning_rate = 0.1

        while evaluations < self.budget:
            grad = self.estimate_gradient(func, best_individual)
            best_individual = np.clip(best_individual - learning_rate * grad, self.bounds[0], self.bounds[1])
            best_f_value_new = func(best_individual)
            evaluations += 1

            if best_f_value_new < best_f_value:
                best_f_value = best_f_value_new
            else:
                # Reduce learning rate if no improvement
                learning_rate *= 0.9

        return best_f_value, best_individual

    def estimate_gradient(self, func, individual):
        grad = np.zeros(self.dim)
        base_value = func(individual)
        eps = 1e-5  # Small perturbation for numerical gradient

        for i in range(self.dim):
            perturbed_individual = np.copy(individual)
            perturbed_individual[i] += eps
            perturbed_value = func(perturbed_individual)
            grad[i] = (perturbed_value - base_value) / eps

        return grad
