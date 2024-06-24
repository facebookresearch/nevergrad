import numpy as np


class RefinedDynamicClusteringPSO:
    def __init__(
        self, budget=10000, population_size=50, omega=0.7, phi_p=0.15, phi_g=0.25, cluster_ratio=0.1
    ):
        self.budget = budget
        self.population_size = population_size
        self.omega = omega  # Inertia weight
        self.phi_p = phi_p  # Personal coefficient
        self.phi_g = phi_g  # Global coefficient
        self.dim = 5  # Dimension of the problem
        self.cluster_ratio = cluster_ratio  # Ratio of population to form clusters

    def __call__(self, func):
        lb, ub = -5.0, 5.0  # Search space bounds
        # Initialize particles
        particles = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.zeros((self.population_size, self.dim))
        personal_best = particles.copy()
        personal_best_fitness = np.array([func(p) for p in particles])

        global_best = particles[np.argmin(personal_best_fitness)]
        global_best_fitness = min(personal_best_fitness)

        evaluations = self.population_size
        cluster_best = np.copy(global_best)  # Initialize cluster best

        # Optimization loop
        while evaluations < self.budget:
            # Clustering phase with proper initialization of cluster_best
            if evaluations % int(self.budget * self.cluster_ratio) == 0:
                # Use k-means clustering to identify clusters in the particle positions
                from sklearn.cluster import KMeans

                num_clusters = int(self.population_size * self.cluster_ratio)
                kmeans = KMeans(n_clusters=max(2, num_clusters))
                clusters = kmeans.fit_predict(particles)
                cluster_bests = [
                    particles[clusters == i][np.argmin(personal_best_fitness[clusters == i])]
                    for i in range(max(2, num_clusters))
                ]

            for i in range(self.population_size):
                # Update velocity and position
                velocity[i] = (
                    self.omega * velocity[i]
                    + self.phi_p * np.random.rand(self.dim) * (personal_best[i] - particles[i])
                    + self.phi_g * np.random.rand(self.dim) * (global_best - particles[i])
                )

                # Use cluster best for the particle's specific cluster
                if evaluations % int(self.budget * self.cluster_ratio) == 0:
                    cluster_id = clusters[i]
                    cluster_best = cluster_bests[cluster_id]

                velocity[i] += self.phi_g * np.random.rand(self.dim) * (cluster_best - particles[i])
                particles[i] += velocity[i]
                particles[i] = np.clip(particles[i], lb, ub)

                # Evaluate particle's fitness
                current_fitness = func(particles[i])
                evaluations += 1

                # Update personal and global bests
                if current_fitness < personal_best_fitness[i]:
                    personal_best[i] = particles[i]
                    personal_best_fitness[i] = current_fitness

                    if current_fitness < global_best_fitness:
                        global_best = particles[i]
                        global_best_fitness = current_fitness

            # Checkpoint to print or log the best found solution
            if evaluations % 1000 == 0:
                print(f"Evaluation: {evaluations}, Best Fitness: {global_best_fitness}")

        return global_best_fitness, global_best
