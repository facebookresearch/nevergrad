import numpy as np


class AdaptiveGradientBalancedCrossoverPSO:
    def __init__(
        self,
        budget=10000,
        population_size=200,
        initial_inertia=0.95,
        final_inertia=0.35,
        cognitive_weight=2.0,
        social_weight=1.8,
        crossover_rate=0.1,
    ):
        self.budget = budget
        self.population_size = population_size
        self.initial_inertia = initial_inertia
        self.final_inertia = final_inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.crossover_rate = crossover_rate
        self.dim = 5
        self.lb, self.ub = -5.0, 5.0
        self.evolution_rate = (initial_inertia - final_inertia) / budget

    def __call__(self, func):
        particles = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.zeros_like(particles)
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([func(p) for p in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        evaluation_counter = self.population_size
        while evaluation_counter < self.budget:
            self.inertia_weight = max(
                self.initial_inertia - (self.evolution_rate * evaluation_counter), self.final_inertia
            )

            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                personal_component = r1 * self.cognitive_weight * (personal_best_positions[i] - particles[i])
                social_component = r2 * self.social_weight * (global_best_position - particles[i])
                velocities[i] = self.inertia_weight * velocities[i] + personal_component + social_component
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.lb, self.ub)

                if np.random.rand() < self.crossover_rate:
                    j = np.random.choice([x for x in range(self.population_size) if x != i])
                    crossover_point = np.random.randint(self.dim)
                    particles[i][:crossover_point], particles[j][:crossover_point] = (
                        particles[j][:crossover_point].copy(),
                        particles[i][:crossover_point].copy(),
                    )

                current_score = func(particles[i])
                evaluation_counter += 1

                if current_score < personal_best_scores[i]:
                    personal_best_positions[i] = particles[i]
                    personal_best_scores[i] = current_score

                    if current_score < global_best_score:
                        global_best_position = particles[i]
                        global_best_score = current_score

                if evaluation_counter >= self.budget:
                    break

        return global_best_score, global_best_position
