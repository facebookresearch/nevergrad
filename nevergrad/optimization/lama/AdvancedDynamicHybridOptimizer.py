import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor


class AdvancedDynamicHybridOptimizer:
    def __init__(
        self,
        budget=10000,
        init_pop_size=50,
        init_F=0.8,
        init_CR=0.9,
        w=0.5,
        c1=1.5,
        c2=1.5,
        local_search_budget_ratio=0.2,
        max_workers=4,
    ):
        self.budget = budget
        self.init_pop_size = init_pop_size
        self.init_F = init_F
        self.init_CR = init_CR
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.local_search_budget_ratio = local_search_budget_ratio
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.eval_count = 0
        self.max_workers = max_workers

    def local_search(self, x, func, budget):
        result = minimize(
            func, x, method="L-BFGS-B", bounds=[self.bounds] * self.dim, options={"maxiter": budget}
        )
        self.eval_count += result.nfev
        return result.x, result.fun

    def simulated_annealing(self, x, func, budget):
        T = 1.0
        T_min = 0.0001
        alpha = 0.9
        best = x
        best_score = func(x)
        self.eval_count += 1
        while self.eval_count < budget and T > T_min:
            i = 1
            while i <= 100:
                candidate = x + np.random.normal(0, 1, self.dim)
                candidate = np.clip(candidate, self.bounds[0], self.bounds[1])
                candidate_score = func(candidate)
                self.eval_count += 1
                if candidate_score < best_score:
                    best = candidate
                    best_score = candidate_score
                else:
                    ap = np.exp((best_score - candidate_score) / T)
                    if np.random.rand() < ap:
                        best = candidate
                        best_score = candidate_score
                i += 1
            T = T * alpha
        return best, best_score

    def evaluate_population(self, func, population):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            fitness = list(executor.map(func, population))
        self.eval_count += len(population)
        return np.array(fitness)

    def __call__(self, func):
        # Initialize population and velocities for PSO
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.init_pop_size, self.dim))
        fitness = self.evaluate_population(func, population)
        velocities = np.random.uniform(-1, 1, (self.init_pop_size, self.dim))

        # Differential weights and crossover probabilities
        F_values = np.full(self.init_pop_size, self.init_F)
        CR_values = np.full(self.init_pop_size, self.init_CR)

        # Initialize the best known positions
        p_best = population.copy()
        p_best_fitness = fitness.copy()
        g_best = population[np.argmin(fitness)]
        g_best_fitness = np.min(fitness)
        best_fitness_history = [g_best_fitness]

        local_search_budget = int(self.budget * self.local_search_budget_ratio)
        global_search_budget = self.budget - local_search_budget
        local_search_budget_per_individual = local_search_budget // self.init_pop_size

        while self.eval_count < global_search_budget:
            for i in range(self.init_pop_size):
                # PSO update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (p_best[i] - population[i])
                    + self.c2 * r2 * (g_best - population[i])
                )
                population[i] = np.clip(population[i] + velocities[i], self.bounds[0], self.bounds[1])

                # Mutation
                idxs = [idx for idx in range(self.init_pop_size) if idx != i]
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
                    partner_idx = np.random.choice(idxs)
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
                    best_fitness_history.append(g_best_fitness)

                # Population resizing based on convergence rate
                if len(best_fitness_history) > 10 and best_fitness_history[-10] == g_best_fitness:
                    self.init_pop_size = max(5, self.init_pop_size // 2)
                    population = population[: self.init_pop_size]
                    fitness = fitness[: self.init_pop_size]
                    velocities = velocities[: self.init_pop_size]
                    F_values = F_values[: self.init_pop_size]
                    CR_values = CR_values[: self.init_pop_size]
                    p_best = p_best[: self.init_pop_size]
                    p_best_fitness = p_best_fitness[: self.init_pop_size]

                if self.eval_count >= global_search_budget:
                    break

        # Perform local search on the best individuals
        for i in range(self.init_pop_size):
            if self.eval_count >= self.budget:
                break
            local_budget = min(local_search_budget_per_individual, self.budget - self.eval_count)
            new_x, new_f = self.local_search(population[i], func, local_budget)
            if new_f < fitness[i]:
                fitness[i] = new_f
                population[i] = new_x

        # Hybridization with Simulated Annealing
        if self.eval_count < self.budget:
            remaining_budget = self.budget - self.eval_count
            g_best, g_best_fitness = self.simulated_annealing(g_best, func, remaining_budget)

        self.f_opt = g_best_fitness
        self.x_opt = g_best

        return self.f_opt, self.x_opt
