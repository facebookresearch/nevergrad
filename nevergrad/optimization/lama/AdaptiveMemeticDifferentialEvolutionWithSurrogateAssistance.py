import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class AdaptiveMemeticDifferentialEvolutionWithSurrogateAssistance:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0
        self.surrogate_update_frequency = 50

    def differential_mutation(self, target, r1, r2, r3, F):
        mutant = r1 + F * (r2 - r3)
        return np.clip(mutant, self.lb, self.ub)

    def crossover(self, target, mutant, CR):
        crossover_mask = np.random.rand(self.dim) < CR
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        offspring = np.where(crossover_mask, mutant, target)
        return offspring

    def local_search(self, x, func, step_size, max_iter=5):
        best_x = x
        best_f = func(x)
        for _ in range(max_iter):
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            new_x = np.clip(best_x + perturbation, self.lb, self.ub)
            new_f = func(new_x)
            if new_f < best_f:
                best_x, best_f = new_x, new_f
        return best_x, best_f

    def adaptive_parameters(self, iteration, max_iterations):
        F = 0.5 + 0.3 * np.cos(np.pi * iteration / max_iterations)
        CR = 0.9 - 0.8 * (iteration / max_iterations)
        return F, CR

    def reset_population(self, population, fitness, func):
        threshold = np.percentile(fitness, 75)
        for i in range(len(population)):
            if fitness[i] > threshold:
                population[i] = np.random.uniform(self.lb, self.ub, self.dim)
                fitness[i] = func(population[i])
        return population, fitness

    def elitist_selection(self, population, fitness):
        elite_size = max(1, len(population) // 10)
        elite_indices = np.argsort(fitness)[:elite_size]
        return population[elite_indices], fitness[elite_indices]

    def update_surrogate_model(self, population, fitness):
        kernel = Matern(nu=2.5)
        self.surrogate_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        self.surrogate_model.fit(population, fitness)

    def surrogate_assisted_evaluation(self, x):
        return self.surrogate_model.predict([x], return_std=True)

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        population_size = 60
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = len(fitness)
        max_iterations = self.budget // population_size

        self.update_surrogate_model(population, fitness)

        iteration = 0
        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                F, CR = self.adaptive_parameters(iteration, max_iterations)

                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant_vector = self.differential_mutation(population[i], a, b, c, F)
                trial_vector = self.crossover(population[i], mutant_vector, CR)

                if evaluations % self.surrogate_update_frequency == 0:
                    self.update_surrogate_model(population, fitness)

                trial_fitness, uncertainty = self.surrogate_assisted_evaluation(trial_vector)
                if uncertainty < 0.1:
                    trial_fitness = func(trial_vector)
                    evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial_vector

                if np.random.rand() < 0.5:
                    local_best_x, local_best_f = self.local_search(
                        population[i], func, step_size=0.01, max_iter=3
                    )
                    evaluations += 1
                    if local_best_f < fitness[i]:
                        population[i] = local_best_x
                        fitness[i] = local_best_f
                        if local_best_f < self.f_opt:
                            self.f_opt = local_best_f
                            self.x_opt = local_best_x

            elites, elite_fitness = self.elitist_selection(population, fitness)

            if evaluations + population_size <= self.budget and iteration % 5 == 0:
                population, fitness = self.reset_population(population, fitness, func)

            elite_size = len(elites)
            population[:elite_size] = elites
            fitness[:elite_size] = elite_fitness

            iteration += 1

        return self.f_opt, self.x_opt


# Example usage:
# optimizer = AdaptiveMemeticDifferentialEvolutionWithSurrogateAssistance(budget=10000)
# best_fitness, best_solution = optimizer(func)
