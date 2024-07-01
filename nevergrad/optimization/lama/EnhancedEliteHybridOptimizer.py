import numpy as np


class EnhancedEliteHybridOptimizer:
    def __init__(
        self,
        budget=10000,
        pop_size=50,
        init_F=0.8,
        init_CR=0.9,
        w=0.5,
        c1=1.5,
        c2=1.5,
        local_search_budget_ratio=0.1,
    ):
        self.budget = budget
        self.pop_size = pop_size
        self.init_F = init_F
        self.init_CR = init_CR
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.local_search_budget_ratio = local_search_budget_ratio
        self.dim = 5  # As stated, dimensionality is 5
        self.bounds = (-5.0, 5.0)  # Bounds are given as [-5.0, 5.0]

    def simulated_annealing(self, x, func, budget, initial_temp=100, cooling_rate=0.95):
        best_x = x.copy()
        best_f = func(x)
        current_x = x.copy()
        current_f = best_f
        temp = initial_temp

        for _ in range(budget):
            if temp <= 0:
                break
            candidate_x = np.clip(
                current_x + np.random.uniform(-0.5, 0.5, self.dim), self.bounds[0], self.bounds[1]
            )
            candidate_f = func(candidate_x)

            if candidate_f < current_f or np.random.rand() < np.exp((current_f - candidate_f) / temp):
                current_x = candidate_x
                current_f = candidate_f
                if candidate_f < best_f:
                    best_x = candidate_x
                    best_f = candidate_f

            temp *= cooling_rate

        return best_x, best_f

    def __call__(self, func):
        # Initialize population and velocities for PSO
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.eval_count = self.pop_size

        # Differential weights and crossover probabilities for each individual
        F_values = np.full(self.pop_size, self.init_F)
        CR_values = np.full(self.pop_size, self.init_CR)

        # Initialize the best known positions
        p_best = population.copy()
        p_best_fitness = fitness.copy()
        g_best = population[np.argmin(fitness)]
        g_best_fitness = np.min(fitness)

        local_search_budget = int(self.budget * self.local_search_budget_ratio)
        global_search_budget = self.budget - local_search_budget
        local_search_budget_per_individual = local_search_budget // self.pop_size

        stagnant_iterations = 0
        max_stagnant_iterations = 100

        while self.eval_count < global_search_budget:
            for i in range(self.pop_size):
                # PSO update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (p_best[i] - population[i])
                    + self.c2 * r2 * (g_best - population[i])
                )
                population[i] = np.clip(population[i] + velocities[i], self.bounds[0], self.bounds[1])

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

                # Incorporate additional crossover mechanism
                if np.random.rand() < 0.3:  # 30% chance to apply blend crossover
                    partner_idx = np.random.choice([idx for idx in range(self.pop_size) if idx != i])
                    partner = population[partner_idx]
                    trial = 0.5 * (trial + partner)
                    trial = np.clip(trial, self.bounds[0], self.bounds[1])

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

                # Update personal best
                if fitness[i] < p_best_fitness[i]:
                    p_best[i] = population[i]
                    p_best_fitness[i] = fitness[i]

                # Update global best
                if fitness[i] < g_best_fitness:
                    g_best = population[i]
                    g_best_fitness = fitness[i]
                    stagnant_iterations = 0
                else:
                    stagnant_iterations += 1

                if self.eval_count >= global_search_budget:
                    break

            # Check for stagnation and reinitialize part of the population
            if stagnant_iterations >= max_stagnant_iterations:
                reinit_indices = np.random.choice(self.pop_size, self.pop_size // 2, replace=False)
                for idx in reinit_indices:
                    population[idx] = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
                    fitness[idx] = func(population[idx])
                    F_values[idx] = self.init_F
                    CR_values[idx] = self.init_CR
                stagnant_iterations = 0
                self.eval_count += len(reinit_indices)

        # Perform local search on the best individuals
        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            local_budget = min(local_search_budget_per_individual, self.budget - self.eval_count)
            new_x, new_f = self.simulated_annealing(population[i], func, local_budget)
            self.eval_count += local_budget
            if new_f < fitness[i]:
                fitness[i] = new_f
                population[i] = new_x

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        return self.f_opt, self.x_opt
