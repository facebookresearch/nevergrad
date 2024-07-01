import numpy as np


class FocusedBalancedAdaptivePSO:
    def __init__(
        self,
        budget=10000,
        population_size=100,
        omega_initial=0.95,
        omega_final=0.2,
        phi_p=0.15,
        phi_g=0.45,
        adaptive_depth=3,
    ):
        self.budget = budget
        self.population_size = population_size
        self.omega_initial = omega_initial  # Initial inertia coefficient
        self.omega_final = omega_final  # Final inertia coefficient
        self.phi_p = phi_p  # Personal best influence factor
        self.phi_g = phi_g  # Global best influence factor
        self.dim = 5  # Problem dimensionality
        self.lb, self.ub = -5.0, 5.0  # Bounds of the search space
        self.adaptive_depth = adaptive_depth  # Depth of adaptive adjustment based on recent performance

    def __call__(self, func):
        particles = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([func(p) for p in particles])
        global_best_position = particles[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        evaluation_counter = self.population_size
        recent_scores = np.array([global_best_score])

        while evaluation_counter < self.budget:
            # Adaptive inertia adjustment based on performance trend
            omega = self.adaptive_inertia(recent_scores, evaluation_counter)

            for i in range(self.population_size):
                r_p = np.random.rand(self.dim)
                r_g = np.random.rand(self.dim)

                velocities[i] = (
                    omega * velocities[i]
                    + self.phi_p * r_p * (personal_best_positions[i] - particles[i])
                    + self.phi_g * r_g * (global_best_position - particles[i])
                )

                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.lb, self.ub)

                current_score = func(particles[i])
                evaluation_counter += 1

                if current_score < personal_best_scores[i]:
                    personal_best_positions[i] = particles[i]
                    personal_best_scores[i] = current_score

                    if current_score < global_best_score:
                        global_best_position = particles[i]
                        global_best_score = current_score
                        recent_scores = np.append(recent_scores, global_best_score)[-self.adaptive_depth :]

                if evaluation_counter >= self.budget:
                    break

        return global_best_score, global_best_position

    def adaptive_inertia(self, scores, evaluation_counter):
        if len(scores) > 1 and np.std(scores) < 0.01:
            # Intensify exploitation by reducing inertia when little improvement is seen
            return max(self.omega_final, self.omega_initial - (evaluation_counter / self.budget) * 1.5)
        else:
            # Regular update rule
            return self.omega_initial - (
                (self.omega_initial - self.omega_final) * (evaluation_counter / self.budget)
            )
