import numpy as np


class EvolutionaryGradientSearch:
    def __init__(self, budget, population_size=50, mutation_rate=0.1, crossover_rate=0.7, learning_rate=0.01):
        self.budget = budget
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.learning_rate = learning_rate

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

        # Initialize the population
        population = np.random.uniform(lower_bound, upper_bound, (self.population_size, dim))
        scores = np.array([func(ind) for ind in population])

        best_idx = np.argmin(scores)
        global_best_position = population[best_idx]
        global_best_score = scores[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            # Selection: Tournament selection
            selected = []
            for _ in range(self.population_size):
                i, j = np.random.randint(0, self.population_size, 2)
                if scores[i] < scores[j]:
                    selected.append(population[i])
                else:
                    selected.append(population[j])
            selected = np.array(selected)

            # Crossover: Blend Crossover (BLX-α)
            offspring = []
            for i in range(0, self.population_size, 2):
                if i + 1 >= self.population_size:
                    break
                parent1, parent2 = selected[i], selected[i + 1]
                if np.random.rand() < self.crossover_rate:
                    alpha = np.random.uniform(-0.5, 1.5, dim)
                    child1 = alpha * parent1 + (1 - alpha) * parent2
                    child2 = alpha * parent2 + (1 - alpha) * parent1
                else:
                    child1, child2 = parent1, parent2
                offspring.extend([child1, child2])
            offspring = np.array(offspring[: self.population_size])

            # Mutation: Gaussian mutation
            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    offspring[i] += np.random.normal(0, 0.1, dim)
                offspring[i] = np.clip(offspring[i], lower_bound, upper_bound)

            # Gradient-based local search
            for i in range(self.population_size):
                grad = self.gradient_estimation(func, offspring[i])
                offspring[i] = np.clip(offspring[i] - self.learning_rate * grad, lower_bound, upper_bound)

            # Evaluate offspring
            offspring_scores = np.array([func(ind) for ind in offspring])
            evaluations += self.population_size

            # Elitism: Preserve the best solution
            if global_best_score < np.min(offspring_scores):
                worst_idx = np.argmax(offspring_scores)
                offspring[worst_idx] = global_best_position
                offspring_scores[worst_idx] = global_best_score

            # Update population and scores
            population, scores = offspring, offspring_scores

            # Update global best
            best_idx = np.argmin(scores)
            if scores[best_idx] < global_best_score:
                global_best_score = scores[best_idx]
                global_best_position = population[best_idx]

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt
