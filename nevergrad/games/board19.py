import itertools
from collections import namedtuple
N = 19
NN = N ** 2
WHITE, BLACK, EMPTY = 'O', 'X', '.'

def swap_colors(color):
    if color == BLACK:
        return WHITE
    elif color == WHITE:
        return BLACK
    else:
        return color

EMPTY_BOARD = EMPTY * NN

def flatten(c):
    return N * c[0] + c[1]

# Convention: coords that have been flattened have a "f" prefix
def unflatten(fc):
    return divmod(fc, N)

def is_on_board(c):
    return c[0] % N == c[0] and c[1] % N == c[1]

def get_valid_neighbors(fc):
    x, y = unflatten(fc)
    possible_neighbors = ((x+1, y), (x-1, y), (x, y+1), (x, y-1))
    return [flatten(n) for n in possible_neighbors if is_on_board(n)]

# Neighbors are indexed by flat coordinates
NEIGHBORS = [get_valid_neighbors(fc) for fc in range(NN)]

def find_reached(board, fc):
    color = board[fc]
    chain = set([fc])
    reached = set()
    frontier = [fc]
    while frontier:
        current_fc = frontier.pop()
        chain.add(current_fc)
        for fn in NEIGHBORS[current_fc]:
            if board[fn] == color and not fn in chain:
                frontier.append(fn)
            elif board[fn] != color:
                reached.add(fn)
    return chain, reached

class IllegalMove(Exception): pass

def place_stone(color, board, fc):
    return board[:fc] + color + board[fc+1:]

def bulk_place_stones(color, board, stones):
    byteboard = bytearray(board, encoding='ascii') # create mutable version of board
    color = ord(color)
    for fstone in stones:
        byteboard[fstone] = color
    return byteboard.decode('ascii') # and cast back to string when done

def maybe_capture_stones(board, fc):
    chain, reached = find_reached(board, fc)
    if not any(board[fr] == EMPTY for fr in reached):
        board = bulk_place_stones(EMPTY, board, chain)
        return board, chain
    else:
        return board, []

def play_move_incomplete(board, fc, color):
    if board[fc] != EMPTY:
        raise IllegalMove
    board = place_stone(color, board, fc)

    opp_color = swap_colors(color)
    opp_stones = []
    my_stones = []
    for fn in NEIGHBORS[fc]:
        if board[fn] == color:
            my_stones.append(fn)
        elif board[fn] == opp_color:
            opp_stones.append(fn)

    for fs in opp_stones:
        board, _ = maybe_capture_stones(board, fs)

    for fs in my_stones:
        board, _ = maybe_capture_stones(board, fs)

    return board

def is_koish(board, fc):
    'Check if fc is surrounded on all sides by 1 color, and return that color'
    if board[fc] != EMPTY: return None
    neighbor_colors = {board[fn] for fn in NEIGHBORS[fc]}
    if len(neighbor_colors) == 1 and not EMPTY in neighbor_colors:
        return list(neighbor_colors)[0]
    else:
        return None

class Position(namedtuple('Position', ['board', 'ko'])):
    @staticmethod
    def initial_state():
        return Position(board=EMPTY_BOARD, ko=None)

    #def __str__(self):
    #  return "\n".join([self.board[i*9:(i+1)*9-1] for i in xrange(9)]) + "\n\n"

    def get_board(self):
        return self.board

    def __str__(self):
        import textwrap
        board_string = '\n'.join(textwrap.wrap(self.board, N))
        return board_string

#        liberties = self.get_liberties()
#        assert len(liberties) == NN
#        liberties = "".join(["." if i == 0 else (str(i) if i < 10 else "9") for i in liberties])
#        return board_string + '\n\n\n' + '\n'.join(textwrap.wrap(liberties, N))
    
    def play_move(self, fc, color):
        board, ko = self
        if fc == ko or board[fc] != EMPTY:
            #print(self)
            raise IllegalMove

        possible_ko_color = is_koish(board, fc)
        new_board = place_stone(color, board, fc)

        opp_color = swap_colors(color)
        opp_stones = []
        my_stones = []
        for fn in NEIGHBORS[fc]:
            if new_board[fn] == color:
                my_stones.append(fn)
            elif new_board[fn] == opp_color:
                opp_stones.append(fn)
        if len(opp_stones) == len(NEIGHBORS[fc]):
            raise IllegalMove

        opp_captured = 0
        for fs in opp_stones:
            new_board, captured = maybe_capture_stones(new_board, fs)
            opp_captured += len(captured)

        for fs in my_stones:
            new_board, captured = maybe_capture_stones(new_board, fs)

        if opp_captured == 1 and possible_ko_color == opp_color:
            new_ko = list(opp_captured)[0]
        else:
            new_ko = None

        return Position(new_board, new_ko)

    def score(self):
        #print str(self)
        board = self.board
        while EMPTY in board:
            fempty = board.index(EMPTY)
            empties, borders = find_reached(board, fempty)
            possible_border_color = board[list(borders)[0]]
            if all(board[fb] == possible_border_color for fb in borders):
                board = bulk_place_stones(possible_border_color, board, empties)
            else:
                # if an empty intersection reaches both white and black,
                # then it belongs to neither player. 
                board = bulk_place_stones('?', board, empties)
        return board.count(BLACK) - board.count(WHITE)

    def get_liberties(self):
        board = self.board
        liberties = [0] * NN  #bytearray(NN)
        for color in (WHITE, BLACK):
            while color in board:
                fc = board.index(color)
                stones, borders = find_reached(board, fc)
                num_libs = len([fb for fb in borders if board[fb] == EMPTY])
                for fs in stones:
                    liberties[fs] = num_libs
                board = bulk_place_stones('?', board, stones)
        return list(liberties)
