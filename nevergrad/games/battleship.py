from __future__ import absolute_import, division, print_function, unicode_literals
#
# We have two versions of the game:
# version 1: oteytaud's version, simple.
# version 2: version as in https://stackoverflow.com/questions/1631414/what-is-the-best-battleship-ai
#        also consistent with https://github.com/Dbz/Battleship/blob/master/BattleShip.py
#
#
# Opponents:
#   undocumented opponents at https://stackoverflow.com/questions/1631414/what-is-the-best-battleship-ai
#   shooting policy in https://github.com/Dbz/Battleship/blob/master/BattleShip.py
#   full policies in https://github.com/Zulban/battleship-ai
#         random: uniform random shooting (even redundant!), one ship per row (max) with random rows, all horizontal
#         ocd: shoot the first place which was not shot; fixed positioning:
#           5: horizontal at (2,2)
#           4: horizontal at (3,2)
#           3: horizontal at (4,2)
#           3: horizontal at (5,2)
#           2: Vertical at   (6,7)
#   shooting policy in http://christopherstoll.org/2012/06/battleship-ai-algorithm-using-dynamic.html


import random
from collections import defaultdict


class battleship(object):
    # We do not announce when a ship is sunk; this variant exists and is presumably more complex.
    # board:
    #  " " sea
    #  "*" sunk missile
    #  "o" ship
    #  "x" destroyed ship

    def __init__(self, version=1):
        self.version = version
        N = self.boardsize()
        self.alive = 0
        board = []
        for _ in range(N):
            board += [[' '] * N]
        self.N = N
        self.board = board
        self.short_policy = 100
        self.long_policy = 5000

    def __str__(self):
        serialized = "".join(["".join(b) for b in self.board])
        print(serialized.count("o"), " alive ships")
        print(serialized.count("x"), " destroyed ships")
        print(serialized.count("*"), " failures")
        stri = "=====================\n"
        for b in self.board:
            stri += "".join(b) + "\n"
        stri += "=====================\n"
        return stri

    def shoot_consistently(self):
        candidate_board = self.compliant_board()
        possible_shoots = []
        for i in range(self.N):
            for j in range(self.N):
                if candidate_board.board[i][j] == "o" and self.board[i][j] != "x":
                    possible_shoots += [(i, j)]
        random.seed()
        s = random.choice(possible_shoots)
        self.shoot(s[0], s[1])

    def good_shoot(self, policy):
        random.seed()
        if policy is not None:
            index = 0
            random.seed(17)
            for i in range(self.N):
                for j in range(self.N):
                    if board[i][j] == "o":
                        index *= random.choice(30) + 1
                        index += random.choice(range(30))
                        index = index % 10000000
                    if board[i][j] == "x":
                        index *= random.choice(30) + 1
                        index += random.choice(range(30, 60))
                        index = index % 10000000
                    if board[i][j] == "*":
                        index *= random.choice(30) + 1
                        index += random.choice(range(60, 90))
                        index = index % 10000000
                    if board[i][j] == " ":
                        index *= random.choice(30) + 1
                        index += random.choice(range(90, 120))
                        index = index % 10000000            
            random.seed(policy[index % len(policy)])
        possible_shoots = defaultdict(lambda: 0)
        for _ in range(7):
            candidate_board = self.compliant_board()
            #      print "a candidate board for "
            #      print self
            #      print " is "
            #      print candidate_board
            for i in range(self.N):
                for j in range(self.N):
                    if candidate_board.board[i][j] == "o" and self.board[i][j] != "x":
                        possible_shoots[(i, j)] += 1
        #    print possible_shoots.keys()
        max_num = -1
        assert possible_shoots, str(self) + " has no possible shoots."
        for s in possible_shoots:
            if possible_shoots[s] > max_num:
                max_num = possible_shoots[s]
                best_shoot = s
        self.shoot(best_shoot[0], best_shoot[1])

    def compliant_board(self):
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        # TODO: refactor?
        verbose = False  # self.alive < 2
        N = self.N
        board = self.board
        # Finds a board which is compatible with the '*' (no ship) and the 'x' (ship).
        list_of_x = []
        for i, b in enumerate(board):
            list_of_x += [(i, j) for j, x in enumerate(b) if x == 'x']

        while True:
            if verbose:
                print("Creating an empty board.")
            candidate = battleship(version=self.version)
            failed = False
            available_ships = [x_ for x_ in self.sizes()]

            # First, put ships in order to cover the 'x' and not the '*'.
            random.shuffle(available_ships)
            random.shuffle(list_of_x)
            for x in list_of_x:
                if verbose:
                    print("we must cover " + str(x))
                h = x[0]
                v = x[1]
                if candidate.board[h][v] == 'o':  # It's ok, there is already a ship here, we don't have to cover it.
                    if verbose:
                        print("already covered, cool!")
                    continue
                # Let us pick up the next ship.
                if not available_ships:
                    if verbose:
                        print("no more ships!")
                    failed = True
                    break
                ship = available_ships[-1]

                # Randomly draw its direction and shift, so that it covers (h,v).
                while True:
                    if verbose:
                        print("trying to put it on the board...", h, v, ship)
                    shift = random.randint(1-ship, 0)
                    if random.choice([True, False]):
                        dx = 1
                        dy = 0
                    else:
                        dx = 0
                        dy = 1
                    if h + shift*dx >= 0 and h+(ship+shift)*dx <= N and v+shift*dy >= 0 and v+(ship+shift)*dy <= N:
                        break

                if verbose:
                    print("ship found, with its position.")
                # Put this ship!
                for a in range(shift, shift+ship):
                    if board[h+a*dx][v+a*dy] == '*' or candidate.board[h+a*dx][v+a*dy] == 'o':
                        failed = True  # There is no ship here! or we already put one.
                        break
                if failed:
                    break  # Restart from scratch!
                for a in range(shift, shift+ship):
                    candidate.board[h+a*dx][v+a*dy] = 'o'
                del available_ships[-1]

            if failed:  # We failed to cover this x!
                continue

            if verbose:
                print("Intermediate board with known ships covered:")
                print(candidate)
                print(" ... should cover:")
                print(self)
                print("Available ships:", available_ships)

            # Now let us put the remaining ships!
            while available_ships and not failed:
                if verbose:
                    print("Remaining ships :" + str(available_ships))
                # Pick a ship.
                s = available_ships[-1]
                if verbose:
                    print("We try to put ship " + str(s))
                # Try to put it!
                num_failures = 5000
                failures = 0
                while failures < num_failures:
                    try_other_position = False
                    x = random.randint(0, N-s)
                    y = random.randint(0, N-1)
                    dx = 1
                    dy = 0
                    if random.choice([False, True]):
                        tmp = x
                        x = y
                        y = tmp
                        dx = 0
                        dy = 1

                    for i in range(s):
                        if candidate.board[x+i*dx][y+i*dy] == 'o' or board[x+i*dx][y+i*dy] == '*':
                            #print " Arg!"
                            #print "candidateboard=" + (candidate.board[x+i*dx][y+i*dy]) + " where we try to put our ship. (o impossible!)"
                            #print "board=" + (board[x+i*dx][y+i*dy]) + " where we try to put our ship. (* impossible)"
                            try_other_position = True
                            break
                    if try_other_position:
                        #print failures, " failures."
                        failures += 1
                        continue
                    break  # Ok, we have found.

                # If we failed to put this ship we must restart from scratch...
                if failures >= num_failures:
                    failed = True
                    break

                # We can put this ship here.
                for i in range(s):
                    candidate.board[x+i*dx][y+i*dy] = 'o'
                del available_ships[-1]

            if failed:
                continue

            if verbose:
                print("All ships installed!")
                print(candidate)

            # We have a candidate!
            return candidate

    def boardsize(self):
        if self.version == 2:
            return 10
        return 7

    def sizes(self):
        if self.version == 2:
            return [2, 3, 3, 4, 5]
        return [5, 4]  # [5] + [4]*2 + [3]*7 + [2]*4
        # return [5, 4, 3, 2, 2] # [5] + [4]*2 + [3]*7 + [2]*4

    def fill(self, s=None):
        random.seed()
        if s is not None:
            s = random.choice([s[i] for i in range(self.short_policy)])
            random.seed(s)
        N = self.N
        for size in self.sizes():
            while True:
                x = random.choice(list(range(N+1-size)))
                y = random.choice(list(range(N)))
                dx = 1
                dy = 0
                if random.choice([False, True]):
                    frg = x
                    x = y
                    y = frg
                    dx = 0
                    dy = 1
                acceptable = True
                for i in range(size):
                    if self.board[x+i*dx][y+i*dy] is not ' ':
                        acceptable = False
                        break
                if acceptable:
                    for i in range(size):
                        self.board[x+i*dx][y+i*dy] = 'o'
                        self.alive += 1
                    break

    def shoot(self, x, y):
        assert x >= 0
        assert y >= 0
        assert x < self.N
        assert y < self.N
        if self.board[x][y] in [' ', '*']:
            self.board[x][y] = '*'
            return False
        else:
            if self.board[x][y] != 'x':
                self.alive -= 1
            self.board[x][y] = 'x'
            return True


# def test():
#     b = battleship()
#     b.fill()
#     for _ in range(3):
#         b.shoot(random.randint(0, b.N-1), random.randint(0, b.N-1))
#     c = b.compliant_board()
#     print(b, c)


def play_game(policy1=None, policy2=None, version=1):
    if policy1 is None and policy2 is None:
        return self.long_policy
    board1 = battleship(version=version)
    board2 = battleship(version=version)
    board1.fill(policy1)
    board2.fill(policy2)
    if random.choice([False, True]):
        board1.good_shoot(policy2)
    for _ in range(5000):
        #    print "iteration ", i
        #    print "real board1:"
        #    print board1
        board2.good_shoot(policy1)
        if board2.alive == 0:
            return 1
#    print "real board2:"
#    print board2
        board1.good_shoot(policy2)
        if board1.alive == 0:
            return 2
    return 0  # This should never happen.


#play_game([random.uniform(0., 1.) for _ in xrange(500)], [random.uniform(0., 1.) for _ in xrange(500)])

# for _ in xrange(30):
#  print "against other random:", play_game([random.uniform(0., 1.)], [random.uniform(0., 1.)], version=2)
#  print "against idiot:", play_game([random.uniform(0., 1.)], None, version=2)
