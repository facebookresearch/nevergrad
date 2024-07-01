import numpy as np


class AdaptiveCulturalDifferentialMemeticEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

    def differential_mutation(self, target, best, r1, r2, F=0.8):
        mutant = target + F * (best - target) + F * (r1 - r2)
        return np.clip(mutant, self.lb, self.ub)

    def crossover(self, target, mutant, CR=0.9):
        crossover_mask = np.random.rand(self.dim) < CR
        offspring = np.where(crossover_mask, mutant, target)
        return offspring

    def guided_local_search(self, x, func, max_iter=5):
        best_x = x.copy()
        best_f = func(x)
        step_size = 0.01
        for _ in range(max_iter):
            gradient = self.estimate_gradient(best_x, func)
            new_x = np.clip(best_x - step_size * gradient, self.lb, self.ub)
            new_f = func(new_x)
            if new_f < best_f:
                best_x = new_x
                best_f = new_f
        return best_x, best_f

    def estimate_gradient(self, x, func, epsilon=1e-8):
        gradient = np.zeros(self.dim)
        f_x = func(x)
        for i in range(self.dim):
            x_step = np.copy(x)
            x_step[i] += epsilon
            gradient[i] = (func(x_step) - f_x) / epsilon
        return gradient

    def update_knowledge_base(self, knowledge_base, population, fitness):
        best_individual = population[np.argmin(fitness)]
        mean_position = np.mean(population, axis=0)
        knowledge_base["best_solution"] = best_individual
        knowledge_base["best_fitness"] = np.min(fitness)
        knowledge_base["mean_position"] = mean_position

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        population_size = 50
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = len(fitness)

        # Cultural knowledge
        knowledge_base = {
            "best_solution": population[np.argmin(fitness)],
            "best_fitness": np.min(fitness),
            "mean_position": np.mean(population, axis=0),
        }

        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                # Differential Evolution mutation and crossover
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                best = population[np.argmin(fitness)]
                mutant_vector = self.differential_mutation(population[i], best, a, b)
                trial_vector = self.crossover(population[i], mutant_vector)

                # Evaluate trial vector
                trial_fitness = func(trial_vector)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

                    # Update cultural knowledge
                    if trial_fitness < knowledge_base["best_fitness"]:
                        knowledge_base["best_solution"] = trial_vector
                        knowledge_base["best_fitness"] = trial_fitness
                    knowledge_base["mean_position"] = np.mean(population, axis=0)

                    # Update global best
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial_vector

                # Apply guided local search on selected individuals
                if np.random.rand() < 0.2:
                    guided_best_x, guided_best_f = self.guided_local_search(population[i], func)
                    evaluations += 5
                    if guided_best_f < fitness[i]:
                        population[i] = guided_best_x
                        fitness[i] = guided_best_f
                        if guided_best_f < self.f_opt:
                            self.f_opt = guided_best_f
                            self.x_opt = guided_best_x

            # Cultural-based guidance: periodically update population based on cultural knowledge
            if evaluations % (population_size * 2) == 0:
                if knowledge_base["best_solution"] is None:
                    continue

                # Adjust cultural shift influence dynamically based on fitness diversity
                fitness_std = np.std(fitness)
                cultural_influence = 0.5 + (0.2 * fitness_std / (np.mean(fitness) + 1e-9))
                cultural_shift = (
                    knowledge_base["best_solution"] - knowledge_base["mean_position"]
                ) * cultural_influence

                # Cooperative cultural influence updates with mean position
                for i in range(population_size):
                    cooperation_factor = np.random.normal(0.5, 0.1)
                    shift = cooperation_factor * cultural_shift + (1 - cooperation_factor) * (
                        knowledge_base["best_solution"] - population[i]
                    ) * np.random.normal(0, 0.05, self.dim)
                    population[i] = np.clip(population[i] + shift, self.lb, self.ub)

                fitness = np.array([func(ind) for ind in population])
                evaluations += population_size

            self.update_knowledge_base(knowledge_base, population, fitness)

        return self.f_opt, self.x_opt
