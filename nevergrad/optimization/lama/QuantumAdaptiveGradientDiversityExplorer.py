import numpy as np


class QuantumAdaptiveGradientDiversityExplorer:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=50,
        elite_fraction=0.1,
        mutation_intensity=1.0,
        crossover_rate=0.7,
        quantum_prob=0.95,
        gradient_prob=0.1,
        gamma=0.95,
        beta=0.08,
        epsilon=0.02,
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(population_size * elite_fraction)
        self.mutation_intensity = mutation_intensity
        self.crossover_rate = crossover_rate
        self.quantum_prob = quantum_prob
        self.gradient_prob = gradient_prob
        self.gamma = gamma  # Quantum state update influence
        self.beta = beta  # Mutation decay rate
        self.epsilon = epsilon  # Minimum mutation factor

    def __call__(self, func):
        # Initialize population uniformly within the bounds
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        while evaluations < self.budget:
            elite_indices = np.argsort(fitness)[: self.elite_count]
            elites = population[elite_indices]

            new_population = np.empty_like(population)
            for i in range(self.population_size):
                if np.random.random() < self.crossover_rate:
                    parent_indices = np.random.choice(elite_indices, 2, replace=False)
                    child = self.crossover(population[parent_indices[0]], population[parent_indices[1]])
                else:
                    parent_idx = np.random.choice(elite_indices)
                    child = self.mutate(population[parent_idx], evaluations)

                if np.random.random() < self.gradient_prob:
                    child = self.gradient_step(child, func)

                if np.random.random() < self.quantum_prob:
                    child = self.quantum_state_update(child, best_individual)

                new_population[i] = np.clip(child, -5, 5)

            population = new_population
            fitness = np.array([func(ind) for ind in population])
            evaluations += self.population_size

            new_best_idx = np.argmin(fitness)
            if fitness[new_best_idx] < best_fitness:
                best_fitness = fitness[new_best_idx]
                best_individual = population[new_best_idx]

        return best_fitness, best_individual

    def mutate(self, individual, evaluations):
        intensity = max(
            self.epsilon, self.mutation_intensity * np.exp(-self.beta * evaluations / self.budget)
        )
        return individual + np.random.normal(0, intensity, self.dimension)

    def crossover(self, parent1, parent2):
        alpha = np.random.rand(self.dimension)
        return alpha * parent1 + (1 - alpha) * parent2

    def quantum_state_update(self, individual, best_individual):
        perturbation = np.random.normal(0, self.gamma, self.dimension) * (best_individual - individual)
        return individual + perturbation

    def gradient_step(self, individual, func, lr=0.01):
        grad_est = np.zeros(self.dimension)
        fx = func(individual)
        h = 1e-5
        for i in range(self.dimension):
            x_new = np.array(individual)
            x_new[i] += h
            grad_est[i] = (func(x_new) - fx) / h
        return individual - lr * grad_est
