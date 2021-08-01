# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Discussions with Tristan Cazenave, Bruno Bouzy, have been helpful.
# Dagstuhl's 2019 seminar on games has been helpful (seminar 19511).

import functools
import numpy as np
from nevergrad.functions import ExperimentFunction
from nevergrad.parametrization import parameter as p


class _Game:
    def __init__(self):
        self.verbose = False
        self.history1 = []
        self.history2 = []
        self.batawaf = False
        self.converter = {
            "flip": self.flip_play_game,
            "batawaf": functools.partial(self.war_play_game, batawaf=True),
            "war": self.war_play_game,
            "guesswho": self.guesswho_play_game,
            "bigguesswho": functools.partial(self.guesswho_play_game, init=96),
        }

    def get_list_of_games(self):
        return self.converter.keys()

    # If both policies are None, then we return the length of a policy (which is a list of float).
    # Otherwise we return 1 if policy1 wins, 2 if policy2 wins, and 0 in case of draw.
    def play_game(self, game, policy1=None, policy2=None):
        self.history1 = []
        self.history2 = []
        if game not in self.converter.keys():
            raise NotImplementedError(
                f"{game} is not implemented, choose among: {list(self.converter.keys())}"
            )
        return self.converter[game](policy1, policy2)

    def guesswho_play_noturn(self, decks, policy):
        assert decks[0] > 0
        assert decks[1] > 0
        baseline = decks[0] // 2  # The default dichotomic policy.
        if policy is None:
            return baseline
        difference = decks[0] - decks[1]
        late = max(0.0, decks[0] - decks[1])  # How late I am.
        try:
            return int(
                0.5
                + baseline
                + policy[0] * difference
                + policy[1] * late
                + policy[2] * late * difference / (1 + decks[0])
                + policy[3] * late * difference / (1 + decks[1])
            )
        except Exception:  # pylint: disable=broad-except
            return baseline

    def guesswho_play(self, policy, decks, turn):
        if turn == 0:
            choice = self.guesswho_play_noturn(decks, policy)
        else:
            choice = self.guesswho_play_noturn([decks[1], decks[0]], policy)
        choice = max(1, min(choice, decks[turn] - 1))
        decks = list(decks)
        decks[turn] = choice if np.random.randint(decks[turn]) <= choice else decks[turn] - choice
        return decks

    def guesswho_play_game(self, policy1, policy2, init=24):
        if policy1 is None and policy2 is None:
            return 4
        remaining_cards = [init, init]
        if np.random.uniform(0.0, 1.0) > 0.5:
            remaining_cards = self.guesswho_play(policy1, remaining_cards, 0)
            if min(remaining_cards) <= 1:
                return 1
        while True:
            remaining_cards = self.guesswho_play(policy2, remaining_cards, 1)
            if min(remaining_cards) <= 1:
                return 2
            remaining_cards = self.guesswho_play(policy1, remaining_cards, 0)
            if min(remaining_cards) <= 1:
                return 1

    def flip_play_game(self, policy1, policy2):
        if policy1 is None and policy2 is None:
            return 57 * 57
        if np.random.uniform(0.0, 1.0) > 0.5:
            r = self.flip_play_game_nosym(policy1=policy2, policy2=policy1)
            return 1 if r == 2 else 2 if r == 1 else 0
        return self.flip_play_game_nosym(policy1, policy2)

    def flip_match(self, a, b):
        return abs((a - b) % 13) <= 1

    def flip_play_game_nosym(self, policy1, policy2):
        # pylint: disable=too-many-branches,too-many-statements,too-many-locals
        # TODO: refactor?
        cards = [i // 4 for i in range(32)]
        np.random.shuffle(cards)
        stack = sorted(cards[:2])
        visible1 = sorted(cards[2:7])
        visible2 = sorted(cards[7:12])
        cards1 = [cards[i] for i in range(12, 22)]
        cards2 = [cards[i] for i in range(22, 32)]
        nan = float("nan")
        something_moves = True
        while something_moves:
            if self.verbose:
                print("==========")
                print(visible1 + [nan] + [len(visible1) + len(cards1)])
                print(stack)
                print(visible2 + [nan] + [len(visible2) + len(cards2)])
                print("==========")
            something_moves = False

            bestvalue = self.flip_value(
                visible1, visible2, len(visible1) + len(cards1), len(visible2) + len(cards2), stack, policy1
            )
            we_play = False
            for i in range(len(visible1)):  # pylint: disable=consider-using-enumerate
                for location in range(2):
                    # print("testing ", visible1[i], " on ", stack[location])
                    if self.flip_match(visible1[i], stack[location]):
                        # print("player 1 can play ", visible1[i], " on ", stack[location])
                        candidate_visible1 = visible1[:i] + visible1[i + 1 :]
                        candidate_stack = sorted([visible1[i], stack[1 - location]])
                        value = self.flip_value(
                            candidate_visible1,
                            visible2,
                            len(cards1) - 1 + len(visible1),
                            len(cards2) + len(visible2),
                            candidate_stack,
                            policy1,
                        )
                        if value < bestvalue:
                            # print("lgtm")
                            next_visible1 = candidate_visible1
                            bestvalue = value
                            next_stack = candidate_stack
                            we_play = True
            if we_play:
                something_moves = True
                visible1 = sorted(next_visible1 + ([cards1[0]] if cards1 else []))
                stack = sorted(next_stack)
                if cards1:
                    del cards1[0]
                if not visible1:
                    return 1
            bestvalue = self.flip_value(
                visible2, visible1, len(cards2) + len(visible2), len(cards1) + len(visible1), stack, policy2
            )
            we_play = False
            for i in range(len(visible2)):  # pylint: disable=consider-using-enumerate
                for location in range(2):
                    # print("testing ", visible2[i], " on ", stack[location])
                    if self.flip_match(visible2[i], stack[location]):
                        # print("player 2 can play ", visible2[i], " on ", stack[location])
                        candidate_visible2 = visible2[:i] + visible2[i + 1 :]
                        candidate_stack = sorted([visible2[i], stack[1 - location]])
                        value = self.flip_value(
                            candidate_visible2,
                            visible1,
                            len(visible2) + len(cards2) - 1,
                            len(visible1) + len(cards1),
                            candidate_stack,
                            policy2,
                        )
                        if value < bestvalue:
                            # print("lgtm")
                            next_visible2 = candidate_visible2
                            bestvalue = value
                            next_stack = candidate_stack
                            we_play = True
            if we_play:
                something_moves = True
                visible2 = sorted(next_visible2 + ([cards2[0]] if cards2 else []))
                stack = sorted(next_stack)
                if cards2:
                    del cards2[0]
                    if not visible2:
                        return 2
            if not something_moves and cards1 and cards2:
                stack = [cards1[0], cards2[0]]
                del cards1[0]
                del cards2[0]
                something_moves = True
        #    print "=========="
        #    print visible1 + [nan] + [len(visible1) + len(cards1)]
        #    print stack
        #    print visible2 + [nan] + [len(visible2) + len(cards2)]
        #    print "=========="
        return 1 if len(visible1) < len(visible2) else 2 if len(visible2) < len(visible1) else 0

    def flip_value(self, visible1, visible2, l1, l2, stack, policy1):
        # pylint: disable=too-many-arguments
        if policy1 is None:
            return l1
        state = [len(visible1), len(visible2), l1, l2]
        state += [stack[1] - stack[0] % 13]
        for i in range(13):
            state += [sum(1 for s in visible1 if (s - stack[0]) % 13 == i)]
            state += [sum(1 for s in visible1 if (s - stack[1]) % 13 == i)]
            state += [sum(1 for s in visible2 if (s - stack[0]) % 13 == i)]
            state += [sum(1 for s in visible2 if (s - stack[1]) % 13 == i)]
        # state has length 52+5=57
        value = 0.01 * l1
        for i in range(57):
            for j in range(57):
                value += policy1[i * 57 + j] * state[i] * state[j]
        return value

    # TODO remove if not planned to be used
    # def phantomgo_choose(self, policy, history):
    #     if policy is not None and history < len(policy):
    #         state = np.random.RandomState(hash(policy[history]))
    #         result = state.randint(board.NN)
    #         return result
    #     else:
    #         result = np.random.randint(board.NN)
    #         return result

    # def phantomgo_play_game(self, policy1, policy2, size=7):
    #     if policy1 is None and policy2 is None:
    #         return 20000
    #     if np.random.uniform(0., 1.) < .5:
    #         result = self.internal_phantomgo_play_game(policy2, policy1, size)
    #         return 1 if result == 2 else 2 if result == 1 else 0

    #     return self.internal_phantomgo_play_game(policy1, policy2, size)

    # def internal_phantomgo_play_game(self, policy1, policy2, size):
    #     # pylint: disable=too-many-locals
    #     # Empty board.
    #     if size == 7:
    #         EMPTY_BOARD = EMPTY_BOARD7
    #         Position = Position7
    #     if size == 9:
    #         EMPTY_BOARD = EMPTY_BOARD9
    #         Position = Position9
    #     if size == 19:
    #         EMPTY_BOARD = EMPTY_BOARD19
    #         Position = Position19

    #     p = Position(board=EMPTY_BOARD, ko=None)

    #     mixing = 3  # We mix 3 policies. Please note that we also apply a 8-fold random rotation/symmetry.
    #     if policy1 is not None:
    #         init = np.random.randint(mixing)
    #         policy1 = [policy1[i] for i in range(init, len(policy1), mixing)]
    #     if policy2 is not None:
    #         init = np.random.randint(mixing)
    #         policy2 = [policy2[i] for i in range(init, len(policy2), mixing)]

    #     history1 = 1  # Player 1 starts.
    #     history2 = 2

    #     random_rot = np.random.randint(8)

    #     def transformation(x):
    #         a = x // size
    #         b = x % size
    #         if random_rot == 0:
    #             a = size - 1 - a
    #             b = size - 1 - b
    #         elif random_rot == 1:
    #             b = size - 1 - b
    #         elif random_rot == 2:
    #             a = size - 1 - a
    #             b = size - 1 - b
    #         elif random_rot == 3:
    #             b = size - 1 - b
    #         elif random_rot == 4:
    #             c = b
    #             b = a
    #             a = c
    #             a = size - 1 - a
    #         elif random_rot == 5:
    #             c = b
    #             b = a
    #             a = c
    #         elif random_rot == 6:
    #             c = b
    #             b = a
    #             a = c
    #             a = size - 1 - a
    #         elif random_rot == 7:
    #             c = b
    #             b = a
    #             a = c
    #         return a * size + b

    #     # pylint: disable=broad-except
    #     for _ in range(2 * size * size - size - 1):
    #         # print("move " + str(idx))
    #         # print("=================")
    #         # print(str(p))
    #         # print("=================")
    #         for _ in range((size * size - 9) // 2):
    #             try:
    #                 move1 = transformation(self.phantomgo_choose(policy1, history1))
    #                 # print("player1 trying ", move1)
    #                 p = p.play_move(move1, board.BLACK)
    #                 history1 = 2 * history1 + 1  # legal move.
    #                 break
    #             except Exception:
    #                 # print("failed!" + str(e))
    #                 history1 = 2 * history1 + 2  # illegal move.
    #         # print("=================")
    #         # print(p)
    #         # print("=================")
    #         for _ in range((size * size - 9) // 2):
    #             try:
    #                 move2 = self.phantomgo_choose(policy2, history2)
    #                 # print("player 2 trying ", move2)
    #                 p = p.play_move(move2, board.WHITE)
    #                 history2 = 2 * history2 + 1  # legal move.
    #                 break
    #             except Exception:
    #                 # print("failed!" + str(e))
    #                 history2 = 2 * history2 + 2  # illegal move.

    #     return 1 if p.score() > 0 else 2

    def war_play_game(self, policy1, policy2, batawaf=False):
        # pylint: disable=too-many-return-statements
        self.batawaf = batawaf
        if policy1 is None and policy2 is None:
            if batawaf:
                return 10 * 18 * 6
            return 10 * 26 * 13
        cards = [i // 4 for i in range(52)] if not batawaf else [i // 6 for i in range(36)]
        shuffled_cards = list(np.random.choice(cards, size=len(cards), replace=False))
        if batawaf:
            cards1 = shuffled_cards[:18]
            cards2 = shuffled_cards[18:]
        else:
            cards1 = shuffled_cards[:26]
            cards2 = shuffled_cards[26:]
        assert len(cards1) == len(cards2)
        stack = []
        for _ in range(500):  # A game of length 5000 is a draw.
            # print cards1, " vs ", cards2

            if not (cards1 or cards2):
                return 0
            elif not cards1:
                return 2
            elif not cards2:
                return 1

            # Pick up first card.
            card1 = cards1[0]
            card2 = cards2[0]
            del cards1[0]
            del cards2[0]
            stack += [card1, card2]

            # Standard case.
            if card1 > card2:
                cards1 += self.war_decide(policy1, len(cards1), sorted(stack, reverse=True))
                stack = []
                continue
            if card2 > card1:
                cards2 += self.war_decide(policy2, len(cards2), sorted(stack, reverse=True))
                stack = []
                continue

            # Battle!
            if len(cards1) < 2 or len(cards2) < 2:
                return 1 if len(cards2) < len(cards1) else 2 if len(cards1) < len(cards2) else 0

            stack += [cards1[0], cards2[0]]
            del cards1[0]
            del cards2[0]
        return 0

    def war_decide(self, policy, num_cards, list_of_cards):
        cards = sorted(list_of_cards, reverse=True)
        if policy is None:
            return cards
        assert len(cards) % 2 == 0
        a = min(num_cards, 9)  # at most 9
        b = len(cards) // 2  # at most 26
        c = cards[0]  # at most 13
        # print(a, b, c, a*26*13+b*13+c, len(policy))
        if self.batawaf:
            seed = policy[a * 18 * 6 + b * 6 + c]  # type: ignore
        else:
            seed = policy[a * 26 * 13 + b * 13 + c]  # type: ignore
        if seed == 0.0:
            return cards
        state = np.random.RandomState(hash(seed) % (2 ** 32))
        state.shuffle(cards)
        return list(cards)


# Real life is more complicated! This is a very simple model.
class Game(ExperimentFunction):
    """
    Parameters
    ----------
    nint intaum_stocks: number of stocks to be managed
    depth: number of layers in the neural networks
    width: number of neurons per hidden layer
    """

    def __init__(self, game: str = "war") -> None:
        self.game = game
        self.game_object = _Game()
        dimension = (
            self.game_object.play_game(self.game) * 2
        )  # times 2 because we consider both players separately.
        super().__init__(self._simulate_game, p.Array(shape=(dimension,)))
        self.parametrization.function.deterministic = False
        self.parametrization.function.metrizable = game not in ["war", "batawaf"]

    def _simulate_game(self, x: np.ndarray) -> float:
        # FIXME: an adaptive opponent, e.g. bandit, would be better.
        # We play a game as player 1.
        p1 = x[: (self.dimension // 2)]
        p2 = np.random.normal(size=self.dimension // 2)
        r = self.game_object.play_game(self.game, p1, p2)
        result = 0.0 if r == 1 else 0.5 if r == 0 else 1.0
        # We play a game as player 2.
        p1 = np.random.normal(size=self.dimension // 2)
        p2 = x[(self.dimension // 2) :]
        r = self.game_object.play_game(self.game, p1, p2)
        return (result + (0.0 if r == 2 else 0.5 if r == 0 else 1.0)) / 2

    def evaluation_function(self, *recommendations: p.Parameter) -> float:
        assert len(recommendations) == 1, "Should not be a pareto set for a singleobjective function"
        x = recommendations[0].value
        # pylint: disable=not-callable
        loss = sum([self.function(x) for _ in range(42)]) / 42.0
        assert isinstance(loss, float)
        return loss
