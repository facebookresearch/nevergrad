import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class AdaptiveDifferentialEvolutionWithSurrogateAssistance:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

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
        F = 0.5 + 0.3 * np.random.rand()
        CR = 0.2 + 0.6 * (1 - np.exp(-iteration / max_iterations))
        return F, CR

    def surrogate_model(self, X, y):
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        gp.fit(X, y)
        return gp

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        population_size = 50
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = len(fitness)
        max_iterations = self.budget // population_size

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

                # Use surrogate model to approximate fitness
                if evaluations < self.budget / 2:
                    trial_fitness = func(trial_vector)
                else:
                    X = population[: i + 1]
                    y = fitness[: i + 1]
                    surrogate = self.surrogate_model(X, y)
                    trial_fitness = surrogate.predict([trial_vector])[0]

                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial_vector

                # Apply local search more aggressively
                if np.random.rand() < 0.5:
                    local_best_x, local_best_f = self.local_search(
                        population[i], func, step_size=0.01, max_iter=10
                    )
                    evaluations += 1
                    if local_best_f < fitness[i]:
                        population[i] = local_best_x
                        fitness[i] = local_best_f
                        if local_best_f < self.f_opt:
                            self.f_opt = local_best_f
                            self.x_opt = local_best_x

            # Reinitialize worst individuals more frequently
            if evaluations + int(0.20 * population_size) <= self.budget:
                worst_indices = np.argsort(fitness)[-int(0.20 * population_size) :]
                for idx in worst_indices:
                    population[idx] = np.random.uniform(self.lb, self.ub, self.dim)
                    fitness[idx] = func(population[idx])
                    evaluations += 1

            # Elite Preservation with larger perturbations
            elite_size = int(0.2 * population_size)
            elite_indices = np.argsort(fitness)[:elite_size]
            elites = population[elite_indices]
            for idx in elite_indices:
                perturbation = np.random.uniform(-0.05, 0.05, self.dim)
                new_elite = np.clip(elites[np.random.randint(elite_size)] + perturbation, self.lb, self.ub)
                new_elite_fitness = func(new_elite)
                evaluations += 1
                if new_elite_fitness < fitness[idx]:
                    population[idx] = new_elite
                    fitness[idx] = new_elite_fitness
                    if new_elite_fitness < self.f_opt:
                        self.f_opt = new_elite_fitness
                        self.x_opt = new_elite

            iteration += 1

        return self.f_opt, self.x_opt
