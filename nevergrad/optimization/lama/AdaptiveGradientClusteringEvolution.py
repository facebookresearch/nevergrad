import numpy as np


class AdaptiveGradientClusteringEvolution:
    def __init__(
        self, budget, dim=5, pop_size=50, num_clusters=5, sigma_init=0.3, learning_rate=0.05, gradient_steps=5
    ):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.num_clusters = num_clusters  # Number of clusters to group individuals
        self.sigma_init = sigma_init  # Initial mutation strength
        self.learning_rate = learning_rate  # Learning rate for gradient updates
        self.gradient_steps = gradient_steps  # Steps to approximate gradient
        self.bounds = np.array([-5.0, 5.0])

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], size=(self.pop_size, self.dim))

    def mutate(self, individual, sigma):
        mutant = individual + sigma * np.random.randn(self.dim)
        return np.clip(mutant, self.bounds[0], self.bounds[1])

    def estimate_gradient(self, func, individual):
        grad = np.zeros(self.dim)
        f_base = func(individual)
        for i in range(self.dim):
            perturb = np.zeros(self.dim)
            eps = self.sigma_init / np.sqrt(self.dim)
            perturb[i] = eps
            f_plus = func(individual + perturb)
            grad[i] = (f_plus - f_base) / eps
        return grad

    def __call__(self, func):
        population = self.initialize_population()
        f_values = np.array([func(x) for x in population])
        evaluations = len(population)
        sigma = np.full(self.pop_size, self.sigma_init)

        while evaluations < self.budget:
            # Cluster population based on feature similarity
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=min(self.num_clusters, len(population)))
            labels = kmeans.fit_predict(population)

            new_population = []
            new_f_values = []

            for cluster_id in range(self.num_clusters):
                cluster_indices = np.where(labels == cluster_id)[0]
                if len(cluster_indices) == 0:
                    continue

                # Calculate cluster centroid gradient
                cluster_gradient = np.mean(
                    [self.estimate_gradient(func, population[idx]) for idx in cluster_indices], axis=0
                )
                cluster_f_values = f_values[cluster_indices]
                best_idx = cluster_indices[np.argmin(cluster_f_values)]

                # Update best individual in the cluster using the average gradient
                best_individual = population[best_idx]
                updated_individual = np.clip(
                    best_individual - self.learning_rate * cluster_gradient, self.bounds[0], self.bounds[1]
                )
                new_f_value = func(updated_individual)
                evaluations += 1

                # Updating population and tracking best solution
                new_population.append(updated_individual)
                new_f_values.append(new_f_value)

                if evaluations >= self.budget:
                    break

            if evaluations >= self.budget:
                break

            population = np.array(new_population)
            f_values = np.array(new_f_values)

        self.f_opt = np.min(f_values)
        self.x_opt = population[np.argmin(f_values)]

        return self.f_opt, self.x_opt
