import numpy as np


class AdaptiveCovarianceMatrixEvolution:
    def __init__(self, budget=10000, population_size=20):
        self.budget = budget
        self.dim = 5  # as given in the problem statement
        self.bounds = (-5.0, 5.0)
        self.population_size = population_size

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        # Track best individual
        best_idx = np.argmin(fitness)
        self.x_opt = population[best_idx]
        self.f_opt = fitness[best_idx]

        # Evolution parameters
        mu = self.population_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = np.sum(weights) ** 2 / np.sum(weights**2)
        sigma = 0.3
        cs = (mueff + 2) / (self.dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (self.dim + 1)) - 1) + cs
        enn = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))
        cc = (4 + mueff / self.dim) / (self.dim + 4 + 2 * mueff / self.dim)
        c1 = 2 / ((self.dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((self.dim + 2) ** 2 + mueff))
        hthresh = (1.4 + 2 / (self.dim + 1)) * enn

        # Evolution strategy state variables
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        B = np.eye(self.dim)
        D = np.ones(self.dim)
        C = np.eye(self.dim)
        invsqrtC = np.eye(self.dim)
        eigenval_update_freq = self.population_size / (c1 + cmu) / self.dim / 10
        eigenval_update_counter = 0

        while evaluations < self.budget:
            # Sample new population
            arz = np.random.randn(self.population_size, self.dim)
            arx = self.x_opt + sigma * np.dot(arz, B * D)

            # Boundary handling
            arx = np.clip(arx, self.bounds[0], self.bounds[1])

            # Evaluate new population
            new_fitness = np.array([func(ind) for ind in arx])
            evaluations += self.population_size

            # Sort by fitness
            sorted_indices = np.argsort(new_fitness)
            arx = arx[sorted_indices]
            arz = arz[sorted_indices]
            new_fitness = new_fitness[sorted_indices]

            # Update best solution found
            if new_fitness[0] < self.f_opt:
                self.f_opt = new_fitness[0]
                self.x_opt = arx[0]

            # Update evolution strategy state variables
            xmean = np.dot(weights, arx[:mu])
            zmean = np.dot(weights, arz[:mu])

            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * np.dot(invsqrtC, zmean)
            hsig = (
                np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * evaluations / self.population_size)) / enn
                < hthresh
            )
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * np.dot(B, D * zmean)

            artmp = (arx[:mu] - self.x_opt) / sigma
            C = (
                (1 - c1 - cmu) * C
                + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
                + cmu * np.dot((weights * artmp.T), artmp)
            )

            sigma *= np.exp((np.linalg.norm(ps) / enn - 1) * cs / ds)

            if eigenval_update_counter <= 0:
                eigenval_update_freq = self.population_size / (c1 + cmu) / self.dim / 10
                eigenval_update_counter = eigenval_update_freq
                C = np.triu(C) + np.triu(C, 1).T
                D, B = np.linalg.eigh(C)
                D = np.sqrt(D)
                invsqrtC = np.dot(B, np.dot(np.diag(D**-1), B.T))
            else:
                eigenval_update_counter -= 1

            self.x_opt = xmean

        return self.f_opt, self.x_opt
