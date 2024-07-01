import numpy as np


class SelfAdaptiveDifferentialSwarmOptimization:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

    def differential_mutation(self, target, best, r1, r2, F):
        mutant = target + F * (best - target) + F * (r1 - r2)
        return np.clip(mutant, self.lb, self.ub)

    def crossover(self, target, mutant, CR):
        crossover_mask = np.random.rand(self.dim) < CR
        offspring = np.where(crossover_mask, mutant, target)
        return offspring

    def local_search(self, x, func, max_iter, step_size):
        best_x = x.copy()
        best_f = func(x)
        for _ in range(max_iter):
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            new_x = np.clip(best_x + perturbation, self.lb, self.ub)
            new_f = func(new_x)
            if new_f < best_f:
                best_x = new_x
                best_f = new_f
        return best_x, best_f

    def adaptive_parameters(self, iteration, max_iterations):
        F = 0.5 + 0.3 * np.random.rand()
        CR = 0.9 - 0.6 * (iteration / max_iterations)
        return F, CR

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
                best = population[np.argmin(fitness)]
                mutant_vector = self.differential_mutation(population[i], best, a, b, F)
                trial_vector = self.crossover(population[i], mutant_vector, CR)

                trial_fitness = func(trial_vector)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial_vector

                if np.random.rand() < 0.8 and evaluations + 5 <= self.budget:
                    local_best_x, local_best_f = self.local_search(
                        population[i], func, max_iter=10, step_size=0.02
                    )
                    evaluations += 5
                    if local_best_f < fitness[i]:
                        population[i] = local_best_x
                        fitness[i] = local_best_f
                        if local_best_f < self.f_opt:
                            self.f_opt = local_best_f
                            self.x_opt = local_best_x

            # Re-initialize half of the worst individuals to maintain diversity
            worst_indices = np.argsort(fitness)[-int(0.5 * population_size) :]
            for idx in worst_indices:
                population[idx] = np.random.uniform(self.lb, self.ub, self.dim)
                fitness[idx] = func(population[idx])
                evaluations += 1

            if iteration % (max_iterations // 3) == 0 and population_size > 20:
                elite_indices = np.argsort(fitness)[: int(0.6 * population_size)]
                population = population[elite_indices]
                fitness = fitness[elite_indices]
                population_size = len(population)

            iteration += 1

        return self.f_opt, self.x_opt
