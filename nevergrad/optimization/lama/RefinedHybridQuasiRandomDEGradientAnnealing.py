import numpy as np
from scipy.stats import qmc


class RefinedHybridQuasiRandomDEGradientAnnealing:
    def __init__(self, budget, population_size=30, initial_crossover_rate=0.7, initial_mutation_factor=0.8):
        self.budget = budget
        self.dim = 5
        self.bounds = [-5.0, 5.0]
        self.population_size = population_size
        self.crossover_rate = initial_crossover_rate
        self.mutation_factor = initial_mutation_factor
        self.base_lr = 0.1
        self.epsilon = 1e-8
        self.temperature = 1.0
        self.cooling_rate = 0.99

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
            # Normalize fitness values to select parents based on their inverse fitness
            fitness = np.array(fitness)
            fitness = fitness - np.min(fitness) + 1e-8  # Ensure all fitness values are positive
            probabilities = 1 / fitness
            probabilities /= probabilities.sum()
            parents_idx = np.random.choice(np.arange(len(population)), size=3, p=probabilities, replace=False)
            return population[parents_idx[0]], population[parents_idx[1]], population[parents_idx[2]]

        def quasi_random_sequence(size):
            sampler = qmc.Sobol(d=self.dim, scramble=True)
            samples = sampler.random(size)
            samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
            return samples

        # Initialize population
        population = quasi_random_sequence(self.population_size)
        fitness = [func(ind) for ind in population]

        for ind, fit in zip(population, fitness):
            if fit < self.f_opt:
                self.f_opt = fit
                self.x_opt = ind

        evaluations = len(population)

        while evaluations < self.budget:
            success_count = 0

            # Differential Evolution
            for j in range(self.population_size):
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

                # Simulated Annealing acceptance criterion
                if new_f < fitness[j] or np.exp(-(new_f - fitness[j]) / self.temperature) > np.random.rand():
                    population[j] = new_x
                    fitness[j] = new_f
                    success_count += 1

                if new_f < self.f_opt:
                    self.f_opt = new_f
                    self.x_opt = new_x

            # Cool down the temperature
            self.temperature *= self.cooling_rate

            # Maintain diversity
            maintain_diversity(population, fitness)

            # Adaptive mutation and crossover strategies based on success count
            if success_count / self.population_size > 0.2:
                self.base_lr *= 1.05
                self.crossover_rate *= 1.05
                self.mutation_factor = min(1.0, self.mutation_factor * 1.05)
            else:
                self.base_lr *= 0.95
                self.crossover_rate *= 0.95
                self.mutation_factor = max(0.5, self.mutation_factor * 0.95)

            # Additional perturbation to improve exploration
            if evaluations % 100 == 0:
                for l in range(self.population_size):
                    population[l] += np.random.randn(self.dim) * self.base_lr * 0.1
                    population[l] = np.clip(population[l], self.bounds[0], self.bounds[1])
                    fitness[l] = func(population[l])
                    evaluations += 1
                    if fitness[l] < self.f_opt:
                        self.f_opt = fitness[l]
                        self.x_opt = population[l]

            self.base_lr = np.clip(self.base_lr, 1e-4, 1.0)
            self.crossover_rate = np.clip(self.crossover_rate, 0.1, 0.9)

        return self.f_opt, self.x_opt


# Example of usage:
# optimizer = RefinedHybridQuasiRandomDEGradientAnnealing(budget=1000)
# best_value, best_solution = optimizer(some_black_box_function)
