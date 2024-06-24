import numpy as np


class CoevolutionaryDualPopulationSearch:
    def __init__(
        self, budget, population_size=30, mutation_rate=0.1, crossover_rate=0.7, learning_rate=0.01, alpha=0.5
    ):
        self.budget = budget
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.learning_rate = learning_rate
        self.alpha = alpha  # Weight for adaptive learning

    def gradient_estimation(self, func, x):
        epsilon = 1e-8
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_pos = np.copy(x)
            x_neg = np.copy(x)
            x_pos[i] += epsilon
            x_neg[i] -= epsilon
            grad[i] = (func(x_pos) - func(x_neg)) / (2 * epsilon)
        return grad

    def __call__(self, func):
        np.random.seed(0)
        dim = 5
        lower_bound = -5.0
        upper_bound = 5.0

        # Initialize two populations
        pop_exploratory = np.random.uniform(lower_bound, upper_bound, (self.population_size, dim))
        pop_exploitative = np.random.uniform(lower_bound, upper_bound, (self.population_size, dim))

        scores_exploratory = np.array([func(ind) for ind in pop_exploratory])
        scores_exploitative = np.array([func(ind) for ind in pop_exploitative])

        best_idx_exploratory = np.argmin(scores_exploratory)
        best_idx_exploitative = np.argmin(scores_exploitative)

        global_best_position = pop_exploratory[best_idx_exploratory]
        global_best_score = scores_exploratory[best_idx_exploratory]

        if scores_exploitative[best_idx_exploitative] < global_best_score:
            global_best_score = scores_exploitative[best_idx_exploitative]
            global_best_position = pop_exploitative[best_idx_exploitative]

        evaluations = 2 * self.population_size

        while evaluations < self.budget:
            # Exploratory Population: Tournament selection and blend crossover (BLX-α)
            selected_exploratory = []
            for _ in range(self.population_size):
                i, j = np.random.randint(0, self.population_size, 2)
                if scores_exploratory[i] < scores_exploitative[j]:
                    selected_exploratory.append(pop_exploratory[i])
                else:
                    selected_exploratory.append(pop_exploitative[j])
            selected_exploratory = np.array(selected_exploratory)

            offspring_exploratory = []
            for i in range(0, self.population_size, 2):
                if i + 1 >= self.population_size:
                    break
                parent1, parent2 = selected_exploratory[i], selected_exploratory[i + 1]
                if np.random.rand() < self.crossover_rate:
                    alpha = np.random.uniform(-self.alpha, 1 + self.alpha, dim)
                    child1 = alpha * parent1 + (1 - alpha) * parent2
                    child2 = alpha * parent2 + (1 - alpha) * parent1
                else:
                    child1, child2 = parent1, parent2
                offspring_exploratory.extend([child1, child2])
            offspring_exploratory = np.array(offspring_exploratory[: self.population_size])

            # Mutation for Exploratory Population
            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    offspring_exploratory[i] += np.random.normal(0, 0.1, dim)
                offspring_exploratory[i] = np.clip(offspring_exploratory[i], lower_bound, upper_bound)

            # Exploitative Population: Gradient-based local search with adaptive learning rate
            for i in range(self.population_size):
                grad = self.gradient_estimation(func, pop_exploitative[i])
                learning_rate_adaptive = self.learning_rate / (1 + evaluations / self.budget)
                pop_exploitative[i] = np.clip(
                    pop_exploitative[i] - learning_rate_adaptive * grad, lower_bound, upper_bound
                )

            # Evaluate offspring
            scores_offspring_exploratory = np.array([func(ind) for ind in offspring_exploratory])
            scores_exploitative = np.array([func(ind) for ind in pop_exploitative])

            evaluations += self.population_size  # Exploratory evaluations
            evaluations += self.population_size  # Exploitative evaluations

            # Update exploratory population and scores
            pop_exploratory, scores_exploratory = offspring_exploratory, scores_offspring_exploratory

            # Update global best from both populations
            best_idx_exploratory = np.argmin(scores_exploratory)
            if scores_exploratory[best_idx_exploratory] < global_best_score:
                global_best_score = scores_exploratory[best_idx_exploratory]
                global_best_position = pop_exploratory[best_idx_exploratory]

            best_idx_exploitative = np.argmin(scores_exploitative)
            if scores_exploitative[best_idx_exploitative] < global_best_score:
                global_best_score = scores_exploitative[best_idx_exploitative]
                global_best_position = pop_exploitative[best_idx_exploitative]

            # Swap roles if one population is stagnating
            if evaluations % (2 * self.population_size) == 0:
                if np.min(scores_exploratory) == global_best_score:
                    pop_exploratory = np.random.uniform(lower_bound, upper_bound, (self.population_size, dim))
                    scores_exploratory = np.array([func(ind) for ind in pop_exploratory])
                if np.min(scores_exploitative) == global_best_score:
                    pop_exploitative = np.random.uniform(
                        lower_bound, upper_bound, (self.population_size, dim)
                    )
                    scores_exploitative = np.array([func(ind) for ind in pop_exploitative])

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt
