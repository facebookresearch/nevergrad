import numpy as np
from sklearn.cluster import KMeans


class HybridDynamicDifferentialEvolutionGradient:
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
            clustering = KMeans(n_clusters=int(np.sqrt(self.population_size)) + 1)
            labels = clustering.fit_predict(population)
            new_population = []
            new_fitness = []
            for cluster_idx in range(max(labels) + 1):
                cluster_members = [i for i, lbl in enumerate(labels) if lbl == cluster_idx]
                if len(cluster_members) > 0:
                    best_member = min(cluster_members, key=lambda idx: fitness[idx])
                    new_population.append(population[best_member])
                    new_fitness.append(fitness[best_member])

            while len(new_population) < self.population_size:
                new_population.append(random_vector())
                new_fitness.append(func(new_population[-1]))

            return new_population, new_fitness

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
                perturbation = np.random.randn(self.dim) * self.base_lr
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

            population, fitness = maintain_diversity(population, fitness)

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
# optimizer = HybridDynamicDifferentialEvolutionGradient(budget=1000)
# best_value, best_solution = optimizer(some_black_box_function)
