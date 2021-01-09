# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import random
import copy


class Six:
    def __init__(self, num_players=5):
        self.num_players = num_players
        self.cards = list(np.random.permutation(list(range(1, 105))))
        self.current = [[self.cards.pop()] for _ in range(4)]
        self.players = [{self.cards.pop() for _ in range(10)} for p in range(num_players)]
        self.past = []
        self.losses = [0 for _ in range(num_players)]
        self.ai_coefficients = np.zeros(self.dimension())
        self.policies = [self.ai_play, self.random_play, self.min_play, self.max_play, self.naive_play]
        self.consistency()

    def consistency(self, verbose: bool = False) -> None:
        if verbose:
            print("Remaining:", self.cards, "(", len(self.cards), ")")
            for i in range(self.num_players):
                print(f"Player {i}:")
                print(self.players[i], "(", len(self.players[i]), ")")
            for c in self.current:
                print(c)
            print("Past: ", self.past, "(", len(self.past), ")")
        assert (
            sum(len(p) for p in self.players)
            + len(self.cards)
            + sum(len(c) for c in self.current)
            + len(self.past)
            == 104
        )
        for i in range(1, 105):
            total = 1 if i in self.past else 0
            total += i in self.cards
            for p in self.players:
                total += i in p
            for c in self.current:
                total += i in c
            assert total == 1, f"{i} is present in {total} stacks."

    def cost(self, i: int) -> int:
        return 7 if (i % 11 == 0) else 3 if (i % 10 == 0) else 2 if (i % 5 == 0) else 1

    def get_representation(self, player: int) -> float:
        my_desk = [1 if i in self.players[player] else 0 for i in range(1, 105)]
        past = [1 if i in self.past else 0 for i in range(1, 105)]
        current = [i in self.current[j] for j in range(4) for i in range(1, 105)]
        x = np.asarray(my_desk + past + current)
        return np.matmul(x.reshape(-1, 1), x.reshape(1, -1)).ravel()

    def dimension(self) -> int:
        return len(self.get_representation(0))

    def value_function(self, player: int):
        state = self.get_representation(player)
        coefficients = self.ai_coefficients
        assert state.shape == coefficients.shape
        return sum(state * coefficients)

    def ai_play(self, player: int) -> int:
        move = self.naive_play(player=player, use_ai=True)
        assert move in self.players[player]
        return move

    def random_play(self, player: int) -> int:
        return random.sample(self.players[player], 1)[0]

    def min_play(self, player: int) -> int:
        return min(self.players[player])

    def max_play(self, player: int) -> int:
        return max(self.players[player])

    def play_moves(self, moves: list, preserve: bool = False, valued_function: int = -1) -> list:
        if preserve:
            backup_loss = copy.deepcopy(self.losses)
            backup_current = copy.deepcopy(self.current)
            backup_players = copy.deepcopy(self.players)

        costs = np.array([0.0 for _ in range(self.num_players)])
        # Players play from lowest to highest.
        ordered_players = sorted(range(len(moves)), key=lambda i: moves[i])
        for player in ordered_players:
            m = moves[player]
            # print(f"{player} plays {m} where (s)he has {self.players[player]}.")
            if not preserve:
                self.players[player].remove(m)

            # Find the right stack.
            legit_stacks_indices = [i for i in range(4) if m > max(self.current[i])]
            if len(legit_stacks_indices) == 0:  # Then we play the first stack with minimum weight.
                weights = [sum(self.cost(i) for i in self.current[i]) for i in range(4)]
                stack_index = weights.index(min(weights))
                self.losses[player] += weights[stack_index]
                costs[player] += weights[stack_index]
                if not preserve:
                    self.past += self.current[stack_index]
                self.current[stack_index] = [m]
                continue

            # Check the smallest tower.
            legit_stacks_maxima = [max(self.current[i]) for i in legit_stacks_indices]
            stack_index = legit_stacks_indices[legit_stacks_maxima.index(min(legit_stacks_maxima))]
            weights = [sum(self.current[i]) for i in range(4)]
            if len(self.current[stack_index]) == 5:
                self.losses[player] += weights[stack_index]
                costs[player] += weights[stack_index]
                weights[stack_index] = self.cost(m)
                if not preserve:
                    self.past += self.current[stack_index]
                self.current[stack_index] = [m]
            else:
                self.current[stack_index] += [m]

        if valued_function >= 0:
            costs[player] += self.value_function(player)

        if preserve:
            self.losses = backup_loss
            self.current = backup_current
            self.players = backup_players
        return costs

    def naive_play(self, player: int, use_ai: bool = False) -> int:
        remaining_cards = {i for i in range(1, 105)}
        # print("WE SIMULATE A RANDOM STEP =============================================")
        # print("All cards:", remaining_cards)
        self.consistency()
        for c in self.past:
            # print(f"{c} is from the past...")
            remaining_cards.remove(c)
        for c in self.players[player]:
            # print(f"{c} is in my cards.")
            remaining_cards.remove(c)
        for c in [c for current in self.current for c in current]:
            # print(f"{c} is on the board.")
            remaining_cards.remove(c)
        played_by_others = []
        for _ in range(self.num_players - 1):
            c = random.sample(remaining_cards, 1)[0]
            remaining_cards.remove(c)
            played_by_others += [c]
        assert len(played_by_others) == self.num_players - 1
        my_cards = sorted(self.players[player])
        best_cost = float("inf")
        # print(f"SIMULTATION WITH OTHER PLAYING {played_by_others}")
        for c in my_cards:
            # print(f"I try to play {c}...")
            moves = [
                c if i == player else played_by_others[(i - 1) if i > player else i]
                for i in range(self.num_players)
            ]
            cost = self.play_moves(moves, preserve=True, valued_function=player if use_ai else -1)[player]
            if cost < best_cost:
                best_cost = cost
                my_move = c
        # print(f"naive_play proposes {my_move}.")
        assert my_move in self.players[player]
        return my_move

    def next(self):
        moves = []
        for i in range(self.num_players):
            move = self.policies[i % len(self.policies)](i)
            # print(f"Player {i} chooses {move} chosen among {self.players[i]}.")
            assert move in self.players[i]
            moves.append(move)
        self.play_moves(moves)

    def play_game(self, policy: np.asarray) -> float:
        assert len(policy) == self.dimension()
        for _ in range(10):
            self.consistency()
            self.next()
        return self.losses


def dimension(num_players: int = 5):
    return Six(num_players=num_players).dimension()


def play_games(policy: np.array, num_players: int = 5, num_games: int = 1):
    results = np.zeros(num_players)
    for _ in range(num_games):
        game = Six(num_players)
        result = np.asarray(game.play_game(policy=policy))
        results += result
    return results / num_games
