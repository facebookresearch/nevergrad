import sys
sys.path.append("./")
import numpy as np
import nevergrad as ng
import copy
from nevergrad.functions.games.game import ng_game



# def ng_game(game=None, policy1=None, policy2=None):
#     """
#     None ==> returns the list of games
#     just the name of the game ==> returns the two dimensions
#     two policies ==> returns  the result
#     """
# 
#     if game == None:
#         return _Game().get_list_of_games()
# 
#     if policy1 is None or policy2 is None:
#         dim = Game(game).dimension / 2  # Same dimension for both players
#         return dim, dim
#                                                         
#     return Game(game).game_object.play_game(self.game, policy1, policy2)  # policy1 maximizes, policy2 minimizes.


# Creating a problem
games = list(ng_game()) + ["c0" + game for game in list(ng_game())]
game = np.random.choice(games)
style = np.random.choice(["robust", "stochastic"])
print("Working on ", game)

def gaussianize(x):
    return [np.random.RandomState(int(x_)).randn() for x_ in list(x)]

if game == "pkl":
    s = np.random.randint(32)
    game = game + str(s)
    N = int(np.random.RandomState(4*s).choice([6, 15, 27, 48]) )
    lambd = 1 #int(50 * np.log(N))  # Size of the Nash approximation; lambd is the number of pure policies in our mixed policies
    A = np.random.RandomState(4*s + 1).rand(*(N,N))
    xopt = np.random.RandomState(4*s+2).rand(N) > .5
    yopt = np.random.RandomState(4*s+3).rand(N) > .5
    lower = [0.] * N
    upper = [1.] * N
    c0 = False
    print(f"Specifications: seed{s}, dim={N}, pop={lambd}")
    b = np.matmul(A, yopt)
    c = np.matmul(xopt, np.transpose(A))
    num_calls = 0
    def f(x, y):
        global num_calls
        num_calls += 1
        assert len(x) == N and len(x.shape) == 1
        return np.matmul(np.matmul(x, A), y) - np.matmul(b, x) - np.matmul(c, y)
else:
    c0 = game[:2] == "c0"
    s = 0 
    subgame = game[2:] if c0 else game
    N, _ = ng_game(subgame)
    lambd = int(50 * np.log(N))  # Size of the Nash approximation; lambd is the number of pure policies in our mixed policies
    print(f"Dimension = {N}, pop = {lambd}")
    lower = [0.] * N
    upper = [100.] * N
    num_calls = 0
    def f(x, y):
        global num_calls
        num_calls += 1
        if int(np.sqrt(np.sqrt(num_calls))) > int(np.sqrt(np.sqrt(num_calls - 1))):
             print(num_calls)
        return ng_game(subgame, gaussianize(x), gaussianize(y))


algo = np.random.choice([o for o in list(ng.optimizers.registry.keys()) if ("CMA" in o or "OnePlusOne" in o or "PSO" in o or "DE" in o or "BFGS" in o or "Cobyla" in o or "NGOpt" == o or "NgIohTu" in o) ])
budget = np.random.choice([10, 31, 100, 314, 1000])
optim = ng.optimizers.registry[algo](ng.p.Array(shape=(N,), lower=lower, upper=upper).set_integer_casting(no_action=c0), budget)   

num_calls = 0
num_big_calls = 0
a = np.random.choice([0, 10, 20, 50, 75, 100, 200])
def loss(x):
    global num_big_calls
    num_big_calls += 1
    v = [f(np.random.randint(low=0,high=101, size=N), x) for _ in range(int(np.power(num_big_calls, a/100.)))]
    return np.average(v) if style == "stochastic" else np.max(v)
    

v = optim.minimize(loss).value
my_num_calls = num_calls
ex = loss(v)
print(f"Game{game}{style}_Algo{algo}{a}_budget{my_num_calls}_loss{ex}_seed{0}__result")
