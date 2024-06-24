import numpy as np


class EnhancedAdaptiveHybridOptimizer:
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
        self.dim = 5  # Dimensionality is 5
        self.bounds = (-5.0, 5.0)  # Bounds are given as [-5.0, 5.0]

    def gradient_descent(self, x, func, budget, step_size=0.01):
        best_x = x.copy()
        best_f = func(x)
        grad = np.zeros(self.dim)
        for _ in range(budget):
            for i in range(self.dim):
                x_plus = x.copy()
                x_plus[i] += step_size
                f_plus = func(x_plus)
                grad[i] = (f_plus - best_f) / step_size

            x = np.clip(x - step_size * grad, self.bounds[0], self.bounds[1])
            f = func(x)
            if f < best_f:
                best_x = x
                best_f = f

        return best_x, best_f

    def local_search(self, x, func, budget):
        best_x = x.copy()
        best_f = func(x)
        for _ in range(budget):
            perturbation = np.random.uniform(-0.5, 0.5, self.dim)
            candidate_x = np.clip(x + perturbation, self.bounds[0], self.bounds[1])
            candidate_f = func(candidate_x)
            if candidate_f < best_f:
                best_f = candidate_f
                best_x = candidate_x
        return best_x, best_f

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.eval_count = self.pop_size

        F_values = np.full(self.pop_size, self.init_F)
        CR_values = np.full(self.pop_size, self.init_CR)

        p_best = population.copy()
        p_best_fitness = fitness.copy()
        g_best = population[np.argmin(fitness)]
        g_best_fitness = np.min(fitness)

        local_search_budget = int(self.budget * self.local_search_budget_ratio)
        global_search_budget = self.budget - local_search_budget
        local_search_budget_per_individual = local_search_budget // self.pop_size

        while self.eval_count < global_search_budget:
            for i in range(self.pop_size):
                progress = self.eval_count / global_search_budget
                self.w = 0.4 + 0.5 * (1 - progress)
                self.c1 = 1.5 - 0.5 * progress
                self.c2 = 1.5 + 0.5 * progress

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (p_best[i] - population[i])
                    + self.c2 * r2 * (g_best - population[i])
                )
                population[i] = np.clip(population[i] + velocities[i], self.bounds[0], self.bounds[1])

                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F = F_values[i]
                mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])

                CR = CR_values[i]
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                if np.random.rand() < 0.3:
                    partner_idx = np.random.choice([idx for idx in range(self.pop_size) if idx != i])
                    partner = population[partner_idx]
                    trial = 0.5 * (trial + partner)
                    trial = np.clip(trial, self.bounds[0], self.bounds[1])

                f_trial = func(trial)
                self.eval_count += 1
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    F_values[i] = F * 1.1 if F < 1 else F
                    CR_values[i] = CR * 1.1 if CR < 1 else CR
                else:
                    F_values[i] = F * 0.9 if F > 0 else F
                    CR_values[i] = CR * 0.9 if CR > 0 else CR

                if fitness[i] < p_best_fitness[i]:
                    p_best[i] = population[i]
                    p_best_fitness[i] = fitness[i]

                if fitness[i] < g_best_fitness:
                    g_best = population[i]
                    g_best_fitness = fitness[i]

                if self.eval_count >= global_search_budget:
                    break

        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            local_budget = min(local_search_budget_per_individual, self.budget - self.eval_count)
            if np.random.rand() < 0.5:
                new_x, new_f = self.gradient_descent(population[i], func, local_budget // 2)
                self.eval_count += local_budget // 2
            else:
                new_x, new_f = self.local_search(population[i], func, local_budget // 2)
                self.eval_count += local_budget // 2
            if new_f < fitness[i]:
                fitness[i] = new_f
                population[i] = new_x

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        return self.f_opt, self.x_opt
