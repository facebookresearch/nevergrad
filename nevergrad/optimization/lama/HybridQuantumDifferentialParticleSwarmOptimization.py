import numpy as np


class HybridQuantumDifferentialParticleSwarmOptimization:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

        # Parameters for DE
        self.population_size = 100
        self.F_min = 0.5
        self.F_max = 1.0
        self.CR_min = 0.1
        self.CR_max = 0.9

        # Quantum Inspired Parameters
        self.alpha = 0.75
        self.beta = 0.25

        # PSO Parameters
        self.inertia_weight = 0.9
        self.cognitive_constant = 2.0
        self.social_constant = 2.0

        # Stagnation control
        self.stagnation_threshold = 10
        self.stagnation_counter = 0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))

        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)].copy()

        personal_best_positions = population.copy()
        personal_best_fitness = fitness.copy()

        evaluations = self.population_size
        best_fitness_history = [self.f_opt]

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Select three random vectors a, b, c from population
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Adaptive Mutation and Crossover
                F_adaptive = self.F_min + np.random.rand() * (self.F_max - self.F_min)
                CR_adaptive = self.CR_min + np.random.rand() * (self.CR_max - self.CR_min)

                mutant_vector = np.clip(a + F_adaptive * (b - c), self.lb, self.ub)

                trial_vector = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < CR_adaptive:
                        trial_vector[j] = mutant_vector[j]

                # Quantum Inspired Adjustment
                quantum_perturbation = np.random.normal(0, 1, self.dim) * (
                    self.alpha * (self.x_opt - population[i]) + self.beta * (population[i] - self.lb)
                )
                trial_vector = np.clip(trial_vector + quantum_perturbation, self.lb, self.ub)

                f_candidate = func(trial_vector)
                evaluations += 1

                if f_candidate < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = f_candidate

                    if f_candidate < self.f_opt:
                        self.f_opt = f_candidate
                        self.x_opt = trial_vector
                        self.stagnation_counter = 0
                    else:
                        self.stagnation_counter += 1
                else:
                    self.stagnation_counter += 1

                if evaluations >= self.budget:
                    break

                # Update personal best
                if f_candidate < personal_best_fitness[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_fitness[i] = f_candidate

            # Update velocities and positions (PSO component)
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            velocities = (
                self.inertia_weight * velocities
                + self.cognitive_constant * r1 * (personal_best_positions - population)
                + self.social_constant * r2 * (self.x_opt - population)
            )
            population = np.clip(population + velocities, self.lb, self.ub)

            # Evaluate new population
            for i in range(self.population_size):
                f_candidate = func(population[i])
                evaluations += 1

                if f_candidate < fitness[i]:
                    fitness[i] = f_candidate
                    if f_candidate < self.f_opt:
                        self.f_opt = f_candidate
                        self.x_opt = population[i]
                        self.stagnation_counter = 0
                    else:
                        self.stagnation_counter += 1
                else:
                    self.stagnation_counter += 1

                if evaluations >= self.budget:
                    break

            # Store best fitness
            best_fitness_history.append(self.f_opt)

            # Adaptive Parameter Adjustment based on Stagnation Counter
            if self.stagnation_counter > self.stagnation_threshold:
                self.F_max = min(1.0, self.F_max + 0.1)
                self.CR_max = min(1.0, self.CR_max + 0.1)
                self.stagnation_counter = 0
            else:
                self.F_max = max(self.F_min, self.F_max - 0.1)
                self.CR_max = max(self.CR_min, self.CR_max - 0.1)

        return self.f_opt, self.x_opt
