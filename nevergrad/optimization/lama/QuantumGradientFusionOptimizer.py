import numpy as np


class QuantumGradientFusionOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is set to 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        current_budget = 0
        population_size = 50  # Reduced population size for concentrated exploration
        mutation_factor = 1.0  # More intense initial mutation
        crossover_prob = 0.8  # Higher crossover probability for enhanced exploration
        learning_rate = 0.2  # Higher initial learning rate for quicker global convergence

        # Initialize population within bounds
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        current_budget += population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        # Optimization loop
        while current_budget < self.budget:
            gradients = np.zeros_like(population)

            # Enhanced gradient estimation with smaller perturbation
            h = 0.01  # Smaller step for gradient calculation
            for i in range(population_size):
                if current_budget >= self.budget:
                    break

                base_ind = population[i]
                for d in range(self.dim):
                    perturbed_ind_plus = np.array(base_ind)
                    perturbed_ind_minus = np.array(base_ind)
                    perturbed_ind_plus[d] += h
                    perturbed_ind_minus[d] -= h

                    if current_budget + 2 <= self.budget:
                        fitness_plus = func(perturbed_ind_plus)
                        fitness_minus = func(perturbed_ind_minus)
                        current_budget += 2
                        gradient = (fitness_plus - fitness_minus) / (2 * h)
                        gradients[i, d] = gradient

            new_population = population.copy()

            for i in range(population_size):
                if current_budget >= self.budget:
                    break

                # Quantum inspired mutation
                noise = np.random.normal(0, 1, self.dim)
                quantum_jumps = np.where(np.random.rand(self.dim) < 0.1, noise, 0)

                # Apply gradient and quantum jump
                child = population[i] - learning_rate * gradients[i] + quantum_jumps
                child += mutation_factor * np.random.randn(self.dim)

                # Perform crossover
                if np.random.rand() < crossover_prob:
                    partner_idx = np.random.randint(population_size)
                    crossover_mask = np.random.rand(self.dim) < 0.5
                    child = child * crossover_mask + population[partner_idx] * (1 - crossover_mask)

                child = np.clip(child, self.lower_bound, self.upper_bound)
                child_fitness = func(child)
                current_budget += 1

                if child_fitness < fitness[i]:
                    new_population[i] = child
                    fitness[i] = child_fitness

                if child_fitness < best_fitness:
                    best_fitness = child_fitness
                    best_solution = child

            population = new_population

            # Adaptive adjustments
            mutation_factor *= 0.95
            learning_rate *= 0.95
            crossover_prob *= 0.95

        return best_fitness, best_solution
