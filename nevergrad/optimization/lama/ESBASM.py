import numpy as np


class ESBASM:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 50
        self.memory_size = 10
        self.memory = []

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def update_memory(self, best_individual, func):
        if len(self.memory) < self.memory_size:
            self.memory.append(best_individual)
        else:
            worst_idx = np.argmax([func(m) for m in self.memory])
            if func(best_individual) < func(self.memory[worst_idx]):
                self.memory[worst_idx] = best_individual

    def memory_guided_mutation(self, individual, func):
        if not self.memory:
            return individual  # No memory yet, return individual unchanged
        # Select a random memory element and apply mutation
        memory_individual = self.memory[np.random.randint(len(self.memory))]
        mutation_strength = np.random.rand(self.dimension)
        # Apply mutation based on difference with a memory individual
        mutated = individual + mutation_strength * (memory_individual - individual)
        return np.clip(mutated, self.bounds[0], self.bounds[1])

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self.memory_guided_mutation(population[i], func)
                mutant_fitness = func(mutant)
                evaluations += 1

                if mutant_fitness < fitness[i]:
                    population[i] = mutant
                    fitness[i] = mutant_fitness

                    if mutant_fitness < fitness[best_idx]:
                        best_idx = i
                        best_individual = mutant

                if evaluations >= self.budget:
                    break

            # Update the memory with the current best individual
            self.update_memory(best_individual, func)

        return fitness[best_idx], best_individual
