import numpy as np


class RASES:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 50
        self.elite_size = 5

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutate(self, population, sigma):
        mutation = np.random.normal(0, sigma, (self.population_size, self.dimension))
        return np.clip(population + mutation, self.bounds[0], self.bounds[1])

    def crossover(self, parent1, parent2):
        alpha = np.random.uniform(0.3, 0.7)
        return alpha * parent1 + (1 - alpha) * parent2

    def select_elite(self, population, fitness):
        elite_indices = np.argsort(fitness)[: self.elite_size]
        return population[elite_indices], fitness[elite_indices]

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)
        evaluations = self.population_size
        best_fitness = np.min(fitness)
        best_solution = population[np.argmin(fitness)]

        sigma = 0.5  # Initial standard deviation for mutation

        while evaluations < self.budget:
            sigma *= 0.99  # Decay of mutation rate
            if evaluations % (self.budget // 10) == 0:  # Mutation burst
                sigma = 0.5

            new_population = self.mutate(population, sigma)
            new_fitness = self.evaluate(new_population, func)
            evaluations += self.population_size

            # Elite preservation and crossover
            elite_population, elite_fitness = self.select_elite(population, fitness)
            for i in range(self.population_size):
                if np.random.rand() < 0.5:  # 50% chance to crossover with elite
                    elite_partner = elite_population[np.random.randint(self.elite_size)]
                    new_population[i] = self.crossover(new_population[i], elite_partner)

            # Re-evaluate after crossover
            fitness = self.evaluate(new_population, func)

            # Update best solution
            current_best_fitness = np.min(fitness)
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_solution = new_population[np.argmin(fitness)]

            population = new_population

        return best_fitness, best_solution
