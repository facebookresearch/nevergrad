import numpy as np


class EnhancedAdaptiveQGSA_v11:
    def __init__(
        self, budget=1000, num_agents=10, G0=100.0, alpha=0.1, delta=0.1, lb=-5.0, ub=5.0, dimension=5
    ):
        self.budget = budget
        self.num_agents = num_agents
        self.G0 = G0
        self.alpha = alpha
        self.delta = delta
        self.lb = lb
        self.ub = ub
        self.dimension = dimension
        self.iteration = 0
        self.prev_best_fitness = np.Inf

    def _initialize_agents(self):
        return np.random.uniform(self.lb, self.ub, size=(self.num_agents, self.dimension))

    def _calculate_masses(self, fitness_values):
        return 1 / (fitness_values + 1e-10)

    def _calculate_gravitational_force(self, agent, mass, best_agent):
        return self.G0 * mass * (best_agent - agent)

    def _update_agent_position(self, agent, force):
        new_pos = agent + self.alpha * force + self.delta * np.random.uniform(-1, 1, size=agent.shape)
        return np.clip(new_pos, self.lb, self.ub)

    def _objective_function(self, func, x):
        return func(x)

    def _adaptive_parameters(self):
        self.G0 *= 0.95  # Adjust gravitational constant reduction rate
        self.alpha *= 0.98  # Adjust step size reduction rate
        if self.f_opt < self.prev_best_fitness:
            self.delta = min(0.1, self.delta * 1.05)  # Increase perturbation factor slightly if improvement
        else:
            self.delta = max(
                0.01, self.delta * 0.95
            )  # Decrease perturbation factor slightly if no improvement
        self.prev_best_fitness = self.f_opt

    def _update_best_agent(self, agents, fitness_values):
        best_agent_idx = np.argmin(fitness_values)
        best_agent = agents[best_agent_idx]
        return best_agent, best_agent_idx

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        agents = self._initialize_agents()
        fitness_values = np.array([self._objective_function(func, agent) for agent in agents])
        masses = self._calculate_masses(fitness_values)

        for _ in range(self.budget):
            best_agent, best_agent_idx = self._update_best_agent(agents, fitness_values)

            for i in range(self.num_agents):
                if i != best_agent_idx:
                    force = self._calculate_gravitational_force(agents[i], masses[i], best_agent)
                    agents[i] = self._update_agent_position(agents[i], force)
                    agents[i] = np.clip(agents[i], self.lb, self.ub)
                    fitness_values[i] = self._objective_function(func, agents[i])

                    if fitness_values[i] < self.f_opt:
                        self.f_opt = fitness_values[i]
                        self.x_opt = agents[i]

            self._adaptive_parameters()  # Update algorithm parameters

        return self.f_opt, self.x_opt
