import numpy as np


class HybridDifferentialEvolutionParticleSwarmOptimizer:
    def __init__(
        self, budget=10000, pop_size=50, init_F=0.8, init_CR=0.9, inertia=0.5, cognitive=1.5, social=1.5
    ):
        self.budget = budget
        self.pop_size = pop_size
        self.init_F = init_F
        self.init_CR = init_CR
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.dim = 5  # As stated, dimensionality is 5
        self.bounds = (-5.0, 5.0)  # Bounds are given as [-5.0, 5.0]

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.eval_count = self.pop_size

        # Initialize velocities for PSO
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))

        # Personal best tracking for PSO
        p_best = population.copy()
        p_best_fitness = fitness.copy()

        # Global best tracking for PSO
        g_best_idx = np.argmin(fitness)
        g_best = population[g_best_idx]
        g_best_fitness = fitness[g_best_idx]

        # Differential weights and crossover probabilities for each individual
        F_values = np.full(self.pop_size, self.init_F)
        CR_values = np.full(self.pop_size, self.init_CR)

        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                # Mutation
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F = F_values[i]
                mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])

                # Crossover
                CR = CR_values[i]
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                self.eval_count += 1
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    # Self-adapting parameters
                    F_values[i] = F * 1.1 if F < 1 else F
                    CR_values[i] = CR * 1.1 if CR < 1 else CR
                else:
                    F_values[i] = F * 0.9 if F > 0 else F
                    CR_values[i] = CR * 0.9 if CR > 0 else CR

                # PSO update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    self.inertia * velocities[i]
                    + self.cognitive * r1 * (p_best[i] - population[i])
                    + self.social * r2 * (g_best - population[i])
                )
                velocities[i] = np.clip(
                    velocities[i], self.bounds[0] - population[i], self.bounds[1] - population[i]
                )
                population[i] += velocities[i]
                population[i] = np.clip(population[i], self.bounds[0], self.bounds[1])

                # Personal best update
                f_new = func(population[i])
                self.eval_count += 1
                if f_new < p_best_fitness[i]:
                    p_best_fitness[i] = f_new
                    p_best[i] = population[i]

                # Global best update
                if f_new < g_best_fitness:
                    g_best_fitness = f_new
                    g_best = population[i]

                if self.eval_count >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        return self.f_opt, self.x_opt
