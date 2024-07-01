import numpy as np


class DEAMC_LSI:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 100
        self.F_base = 0.8  # Base mutation scaling factor
        self.CR_base = 0.9  # Base crossover probability
        self.local_search_interval = 50  # Perform local search every 50 evaluations

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutation(self, idx, population):
        indices = np.delete(np.arange(self.population_size), idx)
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        mutant = a + self.F_base * (b - c)
        return np.clip(mutant, self.bounds[0], self.bounds[1])

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dimension) < self.CR_base
        return np.where(cross_points, mutant, target)

    def select(self, target, trial, f_target, f_trial):
        return trial if f_trial < f_target else target

    def local_search(self, best_individual, func):
        step_size = 0.1
        for _ in range(10):  # Perform 10 steps of local search
            neighbor = best_individual + np.random.uniform(-step_size, step_size, self.dimension)
            neighbor = np.clip(neighbor, self.bounds[0], self.bounds[1])
            if func(neighbor) < func(best_individual):
                best_individual = neighbor
        return best_individual

    def adapt_parameters(self, successes, trials):
        self.F_base *= 0.95 if successes / trials < 0.2 else 1.05
        self.F_base = min(max(self.F_base, 0.1), 1.0)  # Keep F within [0.1, 1.0]
        self.CR_base = 0.1 + 0.8 * successes / trials  # Keep CR adaptive within [0.1, 0.9]

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)
        evaluations = self.population_size
        best_idx = np.argmin(fitness)

        while evaluations < self.budget:
            new_population = np.copy(population)
            successful_trials = 0

            for i in range(self.population_size):
                mutant = self.mutation(i, population)
                trial = self.crossover(population[i], mutant)
                f_trial = func(trial)
                f_target = fitness[i]

                if f_trial < f_target:
                    new_population[i] = trial
                    fitness[i] = f_trial
                    successful_trials += 1

                evaluations += 1
                if evaluations >= self.budget:
                    break

            # Adapt parameters and possibly perform local search
            self.adapt_parameters(successful_trials, self.population_size)
            if evaluations % self.local_search_interval == 0:
                best_idx = np.argmin(fitness)
                population[best_idx] = self.local_search(population[best_idx], func)
                fitness[best_idx] = func(population[best_idx])

            population = new_population

        return fitness[best_idx], population[best_idx]
