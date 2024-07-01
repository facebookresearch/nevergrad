import numpy as np


class AdaptiveMemeticParticleSwarmOptimization:
    def __init__(
        self,
        budget,
        population_size=50,
        w=0.5,
        c1=2,
        c2=2,
        local_search_budget_ratio=0.1,
        adaptivity_factor=0.7,
    ):
        self.budget = budget
        self.population_size = population_size
        self.w = w  # inertia weight
        self.c1 = c1  # cognitive coefficient
        self.c2 = c2  # social coefficient
        self.local_search_budget_ratio = local_search_budget_ratio
        self.adaptivity_factor = adaptivity_factor  # adaptive factor for tuning parameters dynamically

    def local_search(self, func, x, search_budget):
        best_score = func(x)
        best_x = np.copy(x)
        dim = len(x)

        for _ in range(search_budget):
            new_x = x + np.random.uniform(-0.1, 0.1, dim)
            new_x = np.clip(new_x, -5.0, 5.0)
            new_score = func(new_x)
            if new_score < best_score:
                best_score = new_score
                best_x = np.copy(new_x)

        return best_x, best_score

    def __call__(self, func):
        np.random.seed(0)
        dim = 5
        lower_bound = -5.0
        upper_bound = 5.0

        # Initialize the swarm
        population = np.random.uniform(lower_bound, upper_bound, (self.population_size, dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in personal_best_positions])

        best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[best_idx]
        global_best_score = personal_best_scores[best_idx]

        evaluations = self.population_size
        local_search_budget = int(self.budget * self.local_search_budget_ratio)

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Update velocity
                r1, r2 = np.random.rand(dim), np.random.rand(dim)
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (personal_best_positions[i] - population[i])
                    + self.c2 * r2 * (global_best_position - population[i])
                )

                # Update position
                population[i] = np.clip(population[i] + velocities[i], lower_bound, upper_bound)

                # Evaluate fitness
                score = func(population[i])
                evaluations += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = population[i]

                    # Update global best
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = population[i]

            # Apply local search on global best position for further refinement
            if evaluations + local_search_budget <= self.budget:
                global_best_position, global_best_score = self.local_search(
                    func, global_best_position, local_search_budget
                )
                evaluations += local_search_budget

            # Dynamically adapt parameters based on current best performance
            self.w *= self.adaptivity_factor
            self.c1 *= self.adaptivity_factor
            self.c2 *= self.adaptivity_factor

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt
