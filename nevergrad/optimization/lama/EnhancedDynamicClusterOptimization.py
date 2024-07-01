import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import qmc


class EnhancedDynamicClusterOptimization:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

    def adaptive_parameters(self, evaluations, max_evaluations, param_range):
        progress = evaluations / max_evaluations
        return param_range[0] + (param_range[1] - param_range[0]) * progress

    def fractional_order_velocity_update(self, velocity, order=0.5):
        return np.sign(velocity) * (np.abs(velocity) ** order)

    def local_search(self, position, func, step_size=0.1):
        best_position = position
        best_fitness = func(position)
        for i in range(self.dim):
            for direction in [-1, 1]:
                new_position = np.copy(position)
                new_position[i] += direction * step_size
                new_position = np.clip(new_position, self.lb, self.ub)
                new_fitness = func(new_position)
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_position = new_position
        return best_position, best_fitness

    def __call__(self, func):
        population_size = 80

        # Hybrid Initialization using Sobol Sequence and Random Initialization
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        sample = sampler.random(population_size // 2)
        population = qmc.scale(sample, self.lb, self.ub)
        random_population = np.random.uniform(self.lb, self.ub, (population_size // 2, self.dim))
        population = np.vstack((population, random_population))

        velocity = np.random.uniform(-0.1, 0.1, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        personal_best_positions = np.copy(population)
        personal_best_fitness = np.copy(fitness)
        global_best_position = population[np.argmin(fitness)]
        global_best_fitness = np.min(fitness)

        self.f_opt = global_best_fitness
        self.x_opt = global_best_position

        memory = []
        last_improvement = 0

        while evaluations < self.budget:
            inertia_weight = self.adaptive_parameters(evaluations, self.budget, (0.9, 0.4))
            cognitive_coefficient = self.adaptive_parameters(evaluations, self.budget, (2.0, 1.0))
            social_coefficient = self.adaptive_parameters(evaluations, self.budget, (2.0, 1.0))
            differential_weight = self.adaptive_parameters(evaluations, self.budget, (0.8, 0.2))
            crossover_rate = self.adaptive_parameters(evaluations, self.budget, (0.9, 0.3))

            # Adaptive Clustering Strategy with KMeans
            num_clusters = max(2, int(np.sqrt(population_size)))
            kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(population)
            cluster_centers = kmeans.cluster_centers_

            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                if evaluations - last_improvement > self.budget // 10:
                    strategy = "DE"
                else:
                    strategy = "PSO"

                if strategy == "PSO":
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    inertia = inertia_weight * self.fractional_order_velocity_update(velocity[i])
                    cognitive = cognitive_coefficient * r1 * (personal_best_positions[i] - population[i])
                    cluster_index = kmeans.predict([population[i]])[0]
                    social = social_coefficient * r2 * (cluster_centers[cluster_index] - population[i])
                    velocity[i] = inertia + cognitive + social
                    new_position = np.clip(population[i] + velocity[i], self.lb, self.ub)
                else:
                    # Apply DE Strategy with Adaptive Crossover Mechanism
                    indices = list(range(population_size))
                    indices.remove(i)
                    a, b, c = population[np.random.choice(indices, 3, replace=False)]
                    scaling_factor = 0.5 + np.random.rand() * 0.5
                    mutant_vector = np.clip(a + scaling_factor * (b - c), self.lb, self.ub)
                    crossover_mask = np.random.rand(self.dim) < crossover_rate
                    if not np.any(crossover_mask):
                        crossover_mask[np.random.randint(0, self.dim)] = True
                    new_position = np.where(crossover_mask, mutant_vector, population[i])

                new_fitness = func(new_position)
                evaluations += 1

                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness
                    last_improvement = evaluations

                    if new_fitness < personal_best_fitness[i]:
                        personal_best_positions[i] = new_position
                        personal_best_fitness[i] = new_fitness

                        if new_fitness < self.f_opt:
                            self.f_opt = new_fitness
                            self.x_opt = new_position
                            global_best_position = new_position
                            global_best_fitness = new_fitness

            # Reintroduce promising individuals from memory
            if len(memory) > 0 and evaluations < self.budget:
                for mem_pos, mem_fit in memory:
                    if np.random.rand() < 0.1:
                        index = np.random.randint(0, population_size)
                        population[index] = mem_pos
                        fitness[index] = mem_fit
                        evaluations += 1

            # Update memory with top individuals
            sorted_indices = np.argsort(fitness)
            top_individuals = sorted_indices[: max(1, population_size // 10)]
            memory.extend([(population[idx], fitness[idx]) for idx in top_individuals])
            if len(memory) > population_size:
                memory = memory[:population_size]

            # Apply local search for exploitation within clusters
            for cluster_center in cluster_centers:
                cluster_mask = np.all(np.isclose(population, cluster_center, atol=1.0), axis=1)
                cluster_population = population[cluster_mask]
                if len(cluster_population) > 0:
                    best_idx = np.argmin(fitness[cluster_mask])
                    best_position = cluster_population[best_idx]
                    new_position, new_fitness = self.local_search(best_position, func)
                    if new_fitness < fitness[cluster_mask][best_idx]:
                        population[cluster_mask][best_idx] = new_position
                        fitness[cluster_mask][best_idx] = new_fitness
                        if new_fitness < self.f_opt:
                            self.f_opt = new_fitness
                            self.x_opt = new_position
                            global_best_position = new_position
                            global_best_fitness = new_fitness

        return self.f_opt, self.x_opt
