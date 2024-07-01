import numpy as np


class RefinedAdaptiveQuantumGradientHybridOptimizer:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=250,
        elite_ratio=0.2,
        mutation_intensity=1.2,
        crossover_rate=0.8,
        quantum_prob=0.85,
        gradient_boost_prob=0.35,
        adaptive_factor=0.08,
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(population_size * elite_ratio)
        self.mutation_intensity = mutation_intensity
        self.crossover_rate = crossover_rate
        self.quantum_prob = quantum_prob
        self.gradient_boost_prob = gradient_boost_prob
        self.adaptive_factor = adaptive_factor

    def __call__(self, func):
        # Initialize population within bounds
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
                    child = population[parent_idx].copy()

                if np.random.random() < self.gradient_boost_prob:
                    child = self.gradient_boost(child, func)

                if np.random.random() < self.quantum_prob:
                    child = self.quantum_state_update(child, best_individual)

                mutation_scale = self.adaptive_mutation_scale(evaluations)
                child = np.clip(child + np.random.normal(0, mutation_scale, self.dimension), -5, 5)

                new_population[i] = child

            population = new_population
            fitness = np.array([func(ind) for ind in population])
            evaluations += self.population_size

            new_best_idx = np.argmin(fitness)
            if fitness[new_best_idx] < best_fitness:
                best_fitness = fitness[new_best_idx]
                best_individual = population[new_best_idx]

        return best_fitness, best_individual

    def crossover(self, parent1, parent2):
        alpha = np.random.rand(self.dimension)
        return alpha * parent1 + (1 - alpha) * parent2

    def gradient_boost(self, individual, func, lr=0.01):
        grad_est = np.zeros(self.dimension)
        fx = func(individual)
        h = 1e-5
        for i in range(self.dimension):
            x_new = np.array(individual)
            x_new[i] += h
            grad_est[i] = (func(x_new) - fx) / h
        return individual - lr * grad_est

    def quantum_state_update(self, individual, best_individual):
        return individual + np.random.normal(0, self.adaptive_factor, self.dimension) * (
            best_individual - individual
        )

    def adaptive_mutation_scale(self, evaluations):
        return self.mutation_intensity * np.exp(-self.adaptive_factor * evaluations / self.budget)
