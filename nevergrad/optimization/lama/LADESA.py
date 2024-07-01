import numpy as np


class LADESA:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 100
        self.archive_size = 20
        self.mutation_scale = 0.5  # Initial mutation scale
        self.crossover_prob = 0.7  # Initial crossover probability
        self.elite_size = int(self.population_size * 0.1)

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutate(self, population, best_idx, F):
        mutants = np.empty_like(population)
        for i in range(len(population)):
            idxs = np.random.choice(np.delete(np.arange(len(population)), best_idx), 3, replace=False)
            x1, x2, x3 = population[idxs]
            mutant_vector = np.clip(x1 + F * (x2 - x3), self.bounds[0], self.bounds[1])
            mutants[i] = mutant_vector
        return mutants

    def crossover(self, target, mutant, strategy="binomial"):
        if strategy == "uniform":
            cross_points = np.random.rand(self.dimension) < self.crossover_prob
        else:  # binomial
            cross_points = np.random.rand(self.dimension) < self.crossover_prob
            j_rand = np.random.randint(self.dimension)
            cross_points[j_rand] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)
        evaluations = self.population_size
        archive = population[np.argsort(fitness)[: self.archive_size]]  # Initialize archive
        best_fitness = np.min(fitness)

        while evaluations < self.budget:
            F = np.clip(np.random.normal(self.mutation_scale, 0.1), 0.1, 1.0)
            mutants = self.mutate(population, np.argmin(fitness), F)
            trials = np.array(
                [
                    self.crossover(
                        population[i], mutants[i], strategy=np.random.choice(["uniform", "binomial"])
                    )
                    for i in range(self.population_size)
                ]
            )
            fitness_trials = self.evaluate(trials, func)
            evaluations += self.population_size

            improvement_mask = fitness_trials < fitness
            population[improvement_mask] = trials[improvement_mask]
            fitness[improvement_mask] = fitness_trials[improvement_mask]

            # Update archive and re-introduce archived solutions
            archive_fitness = self.evaluate(archive, func)
            combined_population = np.vstack([population, archive])
            combined_fitness = np.hstack([fitness, archive_fitness])
            best_indices = np.argsort(combined_fitness)[: self.archive_size]
            archive = combined_population[best_indices]

            if evaluations % 500 == 0:
                # Re-seed to maintain diversity
                reseed_indices = np.random.choice(
                    self.population_size, size=int(self.population_size * 0.1), replace=False
                )
                population[reseed_indices] = np.random.uniform(
                    self.bounds[0], self.bounds[1], (len(reseed_indices), self.dimension)
                )
                fitness[reseed_indices] = self.evaluate(population[reseed_indices], func)

            best_fitness = np.min(fitness)

            # Learning adaptation of parameters based on current performance
            if evaluations % 100 == 0:
                current_best_fitness = np.min(fitness)
                if current_best_fitness < best_fitness:
                    best_fitness = current_best_fitness
                    self.mutation_scale *= 0.9
                    self.crossover_prob = min(self.crossover_prob + 0.05, 1.0)
                else:
                    self.mutation_scale = min(self.mutation_scale + 0.05, 1.0)
                    self.crossover_prob *= 0.95

        return best_fitness, population[np.argmin(fitness)]
