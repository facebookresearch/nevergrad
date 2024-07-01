import numpy as np


class AdaptiveQuantumMemeticGradientBoost:
    def __init__(
        self,
        budget,
        population_size=100,
        memetic_rate=0.6,
        alpha=0.2,
        learning_rate=0.01,
        elite_fraction=0.2,
        mutation_factor=0.8,
        crossover_prob=0.9,
    ):
        self.budget = budget
        self.population_size = population_size
        self.memetic_rate = memetic_rate
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.elite_fraction = elite_fraction
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob

    def gradient_estimation(self, func, x, h=1e-8):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x1 = np.copy(x)
            x2 = np.copy(x)
            x1[i] += h
            x2[i] -= h
            grad[i] = (func(x1) - func(x2)) / (2 * h)
        return grad

    def quantum_walk(self, x, global_best):
        return np.clip(x + self.alpha * (global_best - x) * np.random.normal(size=x.shape), -5.0, 5.0)

    def evolutionary_step(self, func, pop, scores):
        new_pop = np.copy(pop)
        new_scores = np.copy(scores)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.mutation_factor * (b - c), -5.0, 5.0)
            cross_points = np.random.rand(len(mutant)) < self.crossover_prob
            if not np.any(cross_points):
                cross_points[np.random.randint(0, len(mutant))] = True
            trial = np.where(cross_points, mutant, pop[i])
            f = func(trial)
            if f < scores[i]:
                new_scores[i] = f
                new_pop[i] = trial
        return new_pop, new_scores

    def local_search(self, func, x, score):
        grad = self.gradient_estimation(func, x)
        candidate = np.clip(x - self.learning_rate * grad, -5.0, 5.0)
        f = func(candidate)
        if f < score:
            return candidate, f
        return x, score

    def ensemble_step(self, func, pop, scores, global_best):
        new_pop, new_scores = self.evolutionary_step(func, pop, scores)
        for i in range(self.population_size):
            if np.random.rand() < self.memetic_rate:
                new_pop[i], new_scores[i] = self.local_search(func, new_pop[i], new_scores[i])
            else:
                new_pop[i] = self.quantum_walk(new_pop[i], global_best)
                new_scores[i] = func(new_pop[i])
        return new_pop, new_scores

    def elite_preservation(self, pop, scores):
        elite_count = int(self.population_size * self.elite_fraction)
        elite_idx = np.argsort(scores)[:elite_count]
        return pop[elite_idx], scores[elite_idx]

    def __call__(self, func):
        np.random.seed(0)
        dim = 5
        lower_bound = -5.0
        upper_bound = 5.0

        # Initialize population
        pop = np.random.uniform(lower_bound, upper_bound, (self.population_size, dim))
        scores = np.array([func(ind) for ind in pop])

        # Global best initialization
        best_idx = np.argmin(scores)
        global_best_position = pop[best_idx]
        global_best_score = scores[best_idx]

        evaluations = self.population_size
        max_iterations = self.budget // self.population_size

        for iteration in range(max_iterations):
            # Perform hybrid step
            pop, scores = self.ensemble_step(func, pop, scores, global_best_position)

            # Perform elite preservation
            elite_pop, elite_scores = self.elite_preservation(pop, scores)
            pop[: len(elite_pop)] = elite_pop
            scores[: len(elite_scores)] = elite_scores

            best_idx = np.argmin(scores)
            if scores[best_idx] < global_best_score:
                global_best_score = scores[best_idx]
                global_best_position = pop[best_idx]

            evaluations += self.population_size
            if evaluations >= self.budget:
                break

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt
