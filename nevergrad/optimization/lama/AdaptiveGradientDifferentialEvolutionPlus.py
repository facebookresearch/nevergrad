import numpy as np
from scipy.stats import qmc


class AdaptiveGradientDifferentialEvolutionPlus:
    def __init__(self, budget, population_size=20, crossover_rate=0.7, mutation_factor=0.8):
        self.budget = budget
        self.dim = 5
        self.bounds = [-5.0, 5.0]
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_factor = mutation_factor
        self.base_lr = 0.1
        self.temperature = 1.0
        self.cooling_rate = 0.99
        self.epsilon = 1e-8

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        def random_vector():
            return np.random.uniform(self.bounds[0], self.bounds[1], self.dim)

        def gradient_estimate(x, h=1e-5):
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x1 = np.copy(x)
                x2 = np.copy(x)
                x1[i] += h
                x2[i] -= h
                grad[i] = (func(x1) - func(x2)) / (2 * h)
            return grad

        def maintain_diversity(population, fitness):
            for i in range(len(population)):
                for j in range(i + 1, len(population)):
                    if np.linalg.norm(population[i] - population[j]) < 1e-3:
                        if fitness[i] > fitness[j]:
                            population[i] = random_vector()
                            fitness[i] = func(population[i])
                        else:
                            population[j] = random_vector()
                            fitness[j] = func(population[j])

        def select_parents(population, fitness):
            fitness = np.array(fitness)
            fitness = fitness - np.min(fitness) + 1e-8
            probabilities = 1 / fitness
            probabilities /= probabilities.sum()
            parents_idx = np.random.choice(np.arange(len(population)), size=3, p=probabilities, replace=False)
            return population[parents_idx[0]], population[parents_idx[1]], population[parents_idx[2]]

        def quasi_random_sequence(size):
            sampler = qmc.Sobol(d=self.dim, scramble=True)
            samples = sampler.random(size)
            return qmc.scale(samples, self.bounds[0], self.bounds[1])

        population = quasi_random_sequence(self.population_size)
        fitness = [func(ind) for ind in population]

        for ind, fit in zip(population, fitness):
            if fit < self.f_opt:
                self.f_opt = fit
                self.x_opt = ind

        evaluations = len(population)

        while evaluations < self.budget:
            success_count = 0

            elite_count = int(0.1 * self.population_size)
            elite_indices = np.argsort(fitness)[:elite_count]
            elite_population = [population[i] for i in elite_indices]
            new_population = elite_population.copy()
            new_fitness = [fitness[i] for i in elite_indices]

            for j in range(elite_count, self.population_size):
                if evaluations >= self.budget:
                    break

                target = population[j]
                a, b, c = select_parents(population, fitness)
                mutant = np.clip(a + self.mutation_factor * (b - c), self.bounds[0], self.bounds[1])

                trial = np.copy(target)
                for k in range(self.dim):
                    if np.random.rand() < self.crossover_rate:
                        trial[k] = mutant[k]

                grad = gradient_estimate(trial)
                perturbation = np.random.randn(self.dim) * self.base_lr
                new_x = trial - self.epsilon * grad + perturbation
                new_x = np.clip(new_x, self.bounds[0], self.bounds[1])
                new_f = func(new_x)
                evaluations += 1

                if new_f < fitness[j] or np.exp(-(new_f - fitness[j]) / self.temperature) > np.random.rand():
                    new_population.append(new_x)
                    new_fitness.append(new_f)
                    success_count += 1
                    if new_f < self.f_opt:
                        self.f_opt = new_f
                        self.x_opt = new_x
                else:
                    new_population.append(target)
                    new_fitness.append(fitness[j])

            population = new_population
            fitness = new_fitness

            self.temperature *= self.cooling_rate
            maintain_diversity(population, fitness)

            if success_count / self.population_size > 0.2:
                self.base_lr *= 1.05
                self.crossover_rate *= 1.05
                self.mutation_factor = min(1.0, self.mutation_factor * 1.05)
            else:
                self.base_lr *= 0.95
                self.crossover_rate *= 0.95
                self.mutation_factor = max(0.5, self.mutation_factor * 0.95)

            self.base_lr = np.clip(self.base_lr, 1e-4, 1.0)
            self.crossover_rate = np.clip(self.crossover_rate, 0.1, 0.9)

        return self.f_opt, self.x_opt


# Example of usage:
# optimizer = AdaptiveGradientDifferentialEvolutionPlus(budget=1000)
# best_value, best_solution = optimizer(some_black_box_function)
