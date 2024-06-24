import numpy as np


class EnhancedStrategicPSO:
    def __init__(
        self,
        budget=10000,
        population_size=200,
        omega_initial=0.9,
        omega_final=0.4,
        phi_p=0.2,
        phi_g=0.8,
        critical_depth=50,
        adaptive_depth=20,
    ):
        self.budget = budget
        self.population_size = population_size
        self.omega_initial = omega_initial  # Initial inertia coefficient
        self.omega_final = omega_final  # Final inertia coefficient
        self.phi_p = phi_p  # Personal attraction coefficient
        self.phi_g = phi_g  # Global attraction coefficient
        self.dim = 5  # Problem dimensionality
        self.lb, self.ub = -5.0, 5.0  # Boundary limits of the search space
        self.critical_depth = critical_depth  # Performance evaluation depth for inertia adaptation
        self.adaptive_depth = adaptive_depth  # Frequency of inertia adjustments

    def __call__(self, func):
        particles = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([func(p) for p in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        evaluation_counter = self.population_size
        improvement_tracker = np.zeros(self.population_size, dtype=bool)
        recent_scores = np.array([global_best_score])

        while evaluation_counter < self.budget:
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
                    improvement_tracker[i] = True

                    if current_score < global_best_score:
                        global_best_position = particles[i]
                        global_best_score = current_score
                        recent_scores = np.append(recent_scores, global_best_score)[-self.critical_depth :]

                if evaluation_counter >= self.budget:
                    break

            # Dynamic update of coefficients based on recent improvements
            if np.any(improvement_tracker):
                self.phi_p, self.phi_g = self.dynamic_adjustment(improvement_tracker, self.phi_p, self.phi_g)
                improvement_tracker[:] = False

        return global_best_score, global_best_position

    def adaptive_inertia(self, scores, evaluation_counter):
        if len(scores) > self.adaptive_depth and np.std(scores[-self.adaptive_depth :]) < 0.01:
            return max(
                self.omega_final,
                self.omega_initial
                - (evaluation_counter / self.budget) * (self.omega_initial - self.omega_final) * 2,
            )
        else:
            return self.omega_initial - (
                (self.omega_initial - self.omega_final) * (evaluation_counter / self.budget)
            )

    def dynamic_adjustment(self, improvement_tracker, phi_p, phi_g):
        improvement_rate = np.mean(improvement_tracker)
        if improvement_rate > 0.1:
            phi_p *= 0.9  # Decrease personal coefficient if improvements are frequent
            phi_g *= 1.1  # Increase global coefficient to explore towards new areas
        elif improvement_rate < 0.05:
            phi_p *= 1.1  # Increase personal coefficient to refine local exploitation
            phi_g *= 0.9  # Decrease global coefficient to reduce jumping towards global best rapidly
        return phi_p, phi_g
