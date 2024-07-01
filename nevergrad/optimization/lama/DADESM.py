import numpy as np


class DADESM:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 100
        self.mutation_scale = 0.5
        self.crossover_prob = 0.7

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutate(self, population, fitness, best_idx, F):
        mutants = np.empty_like(population)
        for i in range(len(population)):
            if np.random.rand() < 0.5:  # Dynamic strategy based on uniform probability
                idxs = np.random.choice(np.delete(np.arange(len(population)), best_idx), 3, replace=False)
                x1, x2, x3 = population[idxs]
                mutant_vector = np.clip(x1 + F * (x2 - x3), self.bounds[0], self.bounds[1])
            else:
                best = population[best_idx]
                random_idx = np.random.randint(len(population))
                random_individual = population[random_idx]
                mutant_vector = np.clip(best + F * (random_individual - best), self.bounds[0], self.bounds[1])
            mutants[i] = mutant_vector
        return mutants

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dimension) < self.crossover_prob
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)
        evaluations = self.population_size

        while evaluations < self.budget:
            F = np.clip(np.random.normal(self.mutation_scale, 0.1), 0.1, 1.0)
            mutants = self.mutate(population, fitness, np.argmin(fitness), F)
            trials = np.array(
                [self.crossover(population[i], mutants[i]) for i in range(self.population_size)]
            )
            fitness_trials = self.evaluate(trials, func)
            evaluations += self.population_size

            improvement_mask = fitness_trials < fitness
            population[improvement_mask] = trials[improvement_mask]
            fitness[improvement_mask] = fitness_trials[improvement_mask]

            # Adaptive parameter control based on recent improvements
            recent_improvements = np.mean(improvement_mask)
            self.mutation_scale *= 0.95 if recent_improvements > 0.2 else 1.05
            self.crossover_prob = min(
                max(self.crossover_prob + (0.05 if recent_improvements > 0.2 else -0.05), 0.1), 1.0
            )

        return np.min(fitness), population[np.argmin(fitness)]
