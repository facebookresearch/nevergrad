import numpy as np


class EnhancedSpectralHybridOptimization:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        population_size = 100
        elite_size = 15
        mutation_factor = 0.8
        crossover_probability = 0.7
        learning_rate = 0.1
        catastrophe_frequency = 750

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        evaluations = population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        # Momentum for velocity update in particles
        velocity = np.zeros_like(population)

        while evaluations < self.budget:
            new_population = []
            for i in range(population_size):
                # Spectral mutation
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant_vector = a + mutation_factor * (b - c)
                mutant_vector = np.clip(mutant_vector, self.lb, self.ub)

                # Hybrid crossover with momentum-based Particle Swarm Optimization
                p_best = population[i]  # Personal best (simplified assumption)
                g_best = self.x_opt  # Global best
                velocity[i] = (
                    velocity[i]
                    + learning_rate * (p_best - population[i])
                    + learning_rate * (g_best - population[i])
                )
                trial_vector = population[i] + velocity[i]

                # Adaptive crossover
                trial_vector = np.array(
                    [
                        mutant_vector[j] if np.random.rand() < crossover_probability else trial_vector[j]
                        for j in range(self.dim)
                    ]
                )
                trial_vector = np.clip(trial_vector, self.lb, self.ub)

                # Fitness evaluation and selection
                trial_fitness = func(trial_vector)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    new_population.append(trial_vector)
                    fitness[i] = trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial_vector
                else:
                    new_population.append(population[i])

            population = np.array(new_population)

            # Catastrophic mutation after fixed intervals to avoid local optima
            if evaluations % catastrophe_frequency == 0:
                for j in range(int(population_size * 0.15)):  # Affect 15% of the population
                    catastrophic_idx = np.random.randint(population_size)
                    population[catastrophic_idx] = np.random.uniform(self.lb, self.ub, self.dim)

            # Maintain an elite set of individuals
            elite_indices = np.argsort(fitness)[:elite_size]
            elite_individuals = population[elite_indices]
            for idx in np.random.choice(range(population_size), elite_size, replace=False):
                population[idx] = elite_individuals[np.random.randint(elite_size)]

        return self.f_opt, self.x_opt
