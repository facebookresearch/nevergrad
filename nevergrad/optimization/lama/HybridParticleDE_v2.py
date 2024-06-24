import numpy as np


class HybridParticleDE_v2:
    def __init__(self, budget=10000, population_size=30):
        self.budget = budget
        self.population_size = population_size
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.9  # Adjusted for better exploration
        self.crossover_probability = 0.8
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.4
        self.social_coeff = 1.4

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        eval_count = self.population_size

        personal_best = np.copy(population)
        personal_best_fitness = np.copy(fitness)
        global_best_idx = np.argmin(fitness)
        global_best = population[global_best_idx]
        global_best_fitness = fitness[global_best_idx]

        while eval_count < self.budget:
            for i in range(self.population_size):
                # Particle Swarm Optimization (PSO) Update
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_coeff * r1 * (personal_best[i] - population[i])
                social_velocity = self.social_coeff * r2 * (global_best - population[i])
                velocity[i] = self.inertia_weight * velocity[i] + cognitive_velocity + social_velocity
                population[i] = np.clip(population[i] + velocity[i], self.bounds[0], self.bounds[1])

                # Differential Evolution (DE) Mutation and Crossover
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.bounds[0], self.bounds[1])

                trial = np.copy(population[i])
                crossover_mask = np.random.rand(self.dim) < self.crossover_probability
                trial[crossover_mask] = mutant[crossover_mask]

                # Fitness Evaluation
                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update personal best
                    if trial_fitness < personal_best_fitness[i]:
                        personal_best[i] = trial
                        personal_best_fitness[i] = trial_fitness

                        # Update global best
                        if trial_fitness < global_best_fitness:
                            global_best = trial
                            global_best_fitness = trial_fitness

                if eval_count >= self.budget:
                    break

            # Update inertia weight for better convergence
            self.inertia_weight = max(0.4, self.inertia_weight * 0.98)

        self.f_opt = global_best_fitness
        self.x_opt = global_best
        return self.f_opt, self.x_opt
