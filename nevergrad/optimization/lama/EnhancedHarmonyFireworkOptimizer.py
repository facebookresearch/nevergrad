import numpy as np


class EnhancedHarmonyFireworkOptimizer:
    def __init__(
        self,
        budget=10000,
        population_size=20,
        dim=5,
        bw=0.1,
        sr=0.3,
        amp_min=0.5,
        amp_max=2.0,
        memory_rate=0.4,
    ):
        self.budget = budget
        self.population_size = population_size
        self.dim = dim
        self.bw = bw  # bandwidth for mutation
        self.sr = sr  # success rate of mutation
        self.amp_min = amp_min  # minimum explosion amplitude
        self.amp_max = amp_max  # maximum explosion amplitude
        self.memory_rate = memory_rate  # rate of retaining memory
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.best_fitness = np.inf
        self.best_solution = None

    def calculate_fitness(self, func, solution):
        return func(solution)

    def mutate_solution(self, solution):
        mutated_solution = np.clip(solution + np.random.normal(0, self.bw, self.dim), -5.0, 5.0)
        return mutated_solution

    def firework_explosion(self, solution):
        explosion_amp = np.random.uniform(self.amp_min, self.amp_max)
        new_solution = solution + np.random.uniform(-1, 1, self.dim) * explosion_amp
        return new_solution

    def update_population(self, func, memory):
        for i in range(self.population_size):
            mutated_solution = self.mutate_solution(self.population[i])
            if np.random.rand() < self.sr:
                new_solution = mutated_solution
            else:
                new_solution = self.firework_explosion(self.population[i])

            new_fitness = self.calculate_fitness(func, new_solution)
            if new_fitness < self.calculate_fitness(func, self.population[i]):
                self.population[i] = new_solution

            if new_fitness < self.best_fitness:
                self.best_fitness = new_fitness
                self.best_solution = new_solution

            if new_fitness < memory[i]:
                memory[i] = new_fitness

    def __call__(self, func):
        memory = np.full(self.population_size, np.inf)
        for itr in range(1, self.budget + 1):
            self.update_population(func, memory)

        aocc = 1 - np.std(memory) / np.mean(memory) if np.mean(memory) != 0 else 0
        return aocc, self.best_solution
