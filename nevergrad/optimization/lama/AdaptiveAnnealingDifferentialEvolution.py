import numpy as np


class AdaptiveAnnealingDifferentialEvolution:
    def __init__(self, budget, population_size=20, init_crossover_rate=0.7, init_mutation_factor=0.8):
        self.budget = budget
        self.dim = 5
        self.bounds = [-5.0, 5.0]
        self.population_size = population_size
        self.crossover_rate = init_crossover_rate
        self.mutation_factor = init_mutation_factor
        self.base_lr = 0.1
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

        def fitness_sharing(population, fitness, sigma_share=0.1):
            shared_fitness = np.copy(fitness)
            for i in range(len(population)):
                for j in range(len(population)):
                    if i != j and np.linalg.norm(population[i] - population[j]) < sigma_share:
                        shared_fitness[i] += fitness[j]
            return shared_fitness

        def select_parents(population, fitness):
            # Normalize fitness values to select parents based on their inverse fitness
            fitness = np.array(fitness)
            fitness = fitness - np.min(fitness) + 1e-8  # Ensure all fitness values are positive
            probabilities = 1 / fitness
            probabilities /= probabilities.sum()
            parents_idx = np.random.choice(np.arange(len(population)), size=3, p=probabilities, replace=False)
            return population[parents_idx[0]], population[parents_idx[1]], population[parents_idx[2]]

        # Initialize population
        population = [random_vector() for _ in range(self.population_size)]
        fitness = [func(ind) for ind in population]

        for ind, fit in zip(population, fitness):
            if fit < self.f_opt:
                self.f_opt = fit
                self.x_opt = ind

        max_generations = self.budget // self.population_size
        temperature = 1.0

        for generation in range(max_generations):
            success_count = 0

            for j in range(self.population_size):
                target = population[j]
                a, b, c = select_parents(population, fitness)
                mutation_factor = self.mutation_factor * (1 - generation / max_generations)
                mutant = np.clip(a + mutation_factor * (b - c), self.bounds[0], self.bounds[1])

                trial = np.copy(target)
                for k in range(self.dim):
                    if np.random.rand() < self.crossover_rate:
                        trial[k] = mutant[k]

                grad = gradient_estimate(trial)
                perturbation = temperature * np.random.randn(self.dim) * self.base_lr
                new_x = trial - self.epsilon * grad + perturbation

                new_x = np.clip(new_x, self.bounds[0], self.bounds[1])
                new_f = func(new_x)

                if new_f < fitness[j]:
                    population[j] = new_x
                    fitness[j] = new_f
                    success_count += 1

                if new_f < self.f_opt:
                    self.f_opt = new_f
                    self.x_opt = new_x

            maintain_diversity(population, fitness)

            shared_fitness = fitness_sharing(population, fitness)
            population = [population[i] for i in np.argsort(shared_fitness)[: self.population_size]]
            fitness = [fitness[i] for i in np.argsort(shared_fitness)[: self.population_size]]

            temperature *= 0.99

            if success_count / self.population_size > 0.2:
                self.base_lr *= 1.05
                self.crossover_rate *= 1.05
            else:
                self.base_lr *= 0.95
                self.crossover_rate *= 0.95

            self.base_lr = np.clip(self.base_lr, 1e-4, 1.0)
            self.crossover_rate = np.clip(self.crossover_rate, 0.1, 0.9)

        return self.f_opt, self.x_opt


# Example of usage:
# optimizer = AdaptiveAnnealingDifferentialEvolution(budget=1000)
# best_value, best_solution = optimizer(some_black_box_function)
