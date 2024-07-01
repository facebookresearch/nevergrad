import numpy as np


class QuantumEnhancedDiversityExplorerV8:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=100,
        elite_fraction=0.1,
        mutation_intensity=0.1,
        crossover_rate=0.9,
        quantum_prob=0.7,
        gamma=0.4,
        beta=0.4,
        epsilon=0.0001,
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(population_size * elite_fraction)
        self.mutation_intensity = mutation_intensity
        self.crossover_rate = crossover_rate
        self.quantum_prob = quantum_prob  # Further increased probability for quantum-inspired state update
        self.gamma = gamma  # Increased gamma for deeper exploration in quantum state updates
        self.beta = beta  # Adjusted beta to manage mutation intensity more aggressively
        self.epsilon = epsilon  # Lower threshold for mutation intensity to ensure fine-tuning

    def __call__(self, func):
        # Initialize population randomly within bounds
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(individual) for individual in population])
        evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        # Main optimization loop
        while evaluations < self.budget:
            # Select elites based on the fitness
            elite_indices = np.argsort(fitness)[: self.elite_count]
            elites = population[elite_indices]

            new_population = np.empty_like(population)
            for i in range(self.population_size):
                parent1 = elites[np.random.choice(len(elites))]
                if np.random.random() < self.crossover_rate:
                    parent2 = elites[np.random.choice(len(elites))]
                    child = self.crossover(parent1, parent2)
                else:
                    child = self.mutate(parent1, evaluations)

                # Apply quantum state update with a higher probability
                if np.random.random() < self.quantum_prob:
                    child = self.quantum_state_update(child, best_individual)

                new_population[i] = np.clip(child, -5, 5)

            # Evaluate new population and update the best solution found
            for i in range(self.population_size):
                new_fitness = func(new_population[i])
                evaluations += 1

                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_individual = new_population[i]

                if evaluations >= self.budget:
                    break

            population = new_population
            fitness = np.array([func(individual) for individual in population])

        return best_fitness, best_individual

    def mutate(self, individual, evaluations):
        intensity = max(
            self.epsilon, self.mutation_intensity * np.exp(-self.beta * evaluations / self.budget)
        )
        return individual + np.random.normal(0, intensity, self.dimension)

    def crossover(self, parent1, parent2):
        alpha = np.random.rand()
        return alpha * parent1 + (1 - alpha) * parent2

    def quantum_state_update(self, individual, best_individual):
        """Enhanced quantum state update for potentially better solutions"""
        perturbation = (
            np.random.uniform(-1, 1, self.dimension) * self.gamma * np.abs(best_individual - individual)
        )
        return individual + perturbation
