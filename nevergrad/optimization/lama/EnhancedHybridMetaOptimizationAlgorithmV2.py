import numpy as np


class EnhancedHybridMetaOptimizationAlgorithmV2:
    def __init__(self, budget=10000, num_pop=10, num_children=5, mutation_rate=0.1):
        self.budget = budget
        self.num_pop = num_pop
        self.num_children = num_children
        self.mutation_rate = mutation_rate
        self.dim = 5
        self.population = np.random.uniform(-5.0, 5.0, size=(self.num_pop, self.dim))
        self.best_fitness = np.Inf
        self.best_solution = None

    def mutate_solution(self, solution):
        mutated_solution = solution + np.random.normal(0, self.mutation_rate, size=self.dim)
        return np.clip(mutated_solution, -5.0, 5.0)

    def evaluate_population(self, func):
        fitness = np.array([func(sol) for sol in self.population])
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < self.best_fitness:
            self.best_fitness = fitness[min_idx]
            self.best_solution = self.population[min_idx].copy()
        return fitness

    def selection(self, fitness):
        idx = np.argsort(fitness)[: self.num_pop]
        self.population = self.population[idx].copy()

    def recombine(self):
        children = []
        for _ in range(self.num_children):
            idx1, idx2 = np.random.choice(self.num_pop, 2, replace=False)
            child = 0.5 * (self.population[idx1] + self.population[idx2])
            children.append(child)
        return np.array(children)

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = self.evaluate_population(func)
            self.selection(fitness)
            children = self.recombine()

            for i in range(self.num_children):
                mutated_child = self.mutate_solution(children[i])
                fitness_child = func(mutated_child)
                if fitness_child < np.max(fitness):
                    idx = np.argmax(fitness)
                    self.population[idx] = mutated_child
                    fitness[idx] = fitness_child

        aocc = 1 - np.std(fitness) / np.mean(fitness)
        return aocc, self.best_solution
