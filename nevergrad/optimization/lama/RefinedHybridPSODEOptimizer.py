import numpy as np


class RefinedHybridPSODEOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

    def adaptive_parameters(self, evaluations, max_evaluations, start_param, end_param):
        progress = evaluations / max_evaluations
        return start_param + (end_param - start_param) * progress

    def __call__(self, func):
        population_size = 50
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        velocity = np.random.uniform(-0.1, 0.1, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        personal_best_positions = np.copy(population)
        personal_best_fitness = np.copy(fitness)
        global_best_position = population[np.argmin(fitness)]
        global_best_fitness = np.min(fitness)

        self.f_opt = global_best_fitness
        self.x_opt = global_best_position

        while evaluations < self.budget:
            inertia_weight = self.adaptive_parameters(evaluations, self.budget, 0.9, 0.4)
            cognitive_coefficient = self.adaptive_parameters(evaluations, self.budget, 2.0, 1.0)
            social_coefficient = self.adaptive_parameters(evaluations, self.budget, 2.0, 1.0)
            differential_weight = self.adaptive_parameters(evaluations, self.budget, 0.8, 0.2)
            crossover_rate = self.adaptive_parameters(evaluations, self.budget, 0.9, 0.3)

            for i in range(population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                inertia = inertia_weight * velocity[i]
                cognitive = cognitive_coefficient * r1 * (personal_best_positions[i] - population[i])
                social = social_coefficient * r2 * (global_best_position - population[i])
                velocity[i] = inertia + cognitive + social
                new_position = np.clip(population[i] + velocity[i], self.lb, self.ub)
                new_fitness = func(new_position)
                evaluations += 1

                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

                    if new_fitness < personal_best_fitness[i]:
                        personal_best_positions[i] = new_position
                        personal_best_fitness[i] = new_fitness

                        if new_fitness < self.f_opt:
                            self.f_opt = new_fitness
                            self.x_opt = new_position
                            global_best_position = new_position
                            global_best_fitness = new_fitness

                if evaluations >= self.budget:
                    break

                indices = list(range(population_size))
                indices.remove(i)

                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant_vector = np.clip(a + differential_weight * (b - c), self.lb, self.ub)

                crossover_mask = np.random.rand(self.dim) < crossover_rate
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True

                trial_vector = np.where(crossover_mask, mutant_vector, population[i])
                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

                    if trial_fitness < personal_best_fitness[i]:
                        personal_best_positions[i] = trial_vector
                        personal_best_fitness[i] = trial_fitness

                        if trial_fitness < self.f_opt:
                            self.f_opt = trial_fitness
                            self.x_opt = trial_vector
                            global_best_position = trial_vector
                            global_best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt
