import numpy as np
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p

class CrowdingDistance:
    """This class implements the calculation of crowding distance for NSGA-II.
    """

    def accumulate_distance_per_objective(self, front: tp.List[p.Parameter], i: int):
        is_multiobj: bool = isinstance(front[0].loss, np.ndarray)
        assert (not is_multiobj and (i == 0)) or is_multiobj

        # Sort the population by objective i
        if is_multiobj:
            front = sorted(front, key=lambda x: x.loss[i])
            objective_minn = front[0].loss[i]
            objective_maxn = front[-1].loss[i]
            assert objective_minn <= objective_maxn

            # Set the crowding distance
            front[0]._meta['crowding_distance'] = float('inf')
            front[-1]._meta['crowding_distance'] = float('inf')

            # All other intermediate solutions are assigned a distance value equal 
            # to the absolute normalized difference in the function values of two
            # adjacent solutions.
            for j in range(1, len(front) - 1):
                distance = front[j + 1].loss[i] - front[j - 1].loss[i]

                # Check if minimum and maximum are the same (in which case do nothing)
                if objective_maxn - objective_minn == 0:
                    pass #undefined
                else:
                    distance = distance / float(objective_maxn - objective_minn)
                print(f"front[j]: {front[j].uid} distance: {distance}")
                # The overall crowding-distance value is calculated as the sum of
                # individual distance values corresponding to each objective.
                front[j]._meta['crowding_distance'] += distance
        else:
            front = sorted(front, key=lambda x: x.loss)
            objective_minn = front[0].loss
            objective_maxn = front[-1].loss
            assert objective_minn <= objective_maxn

            # Set the crowding distance
            front[0]._meta['crowding_distance'] = float('inf')
            front[-1]._meta['crowding_distance'] = float('inf')

            # All other intermediate solutions are assigned a distance value equal 
            # to the absolute normalized difference in the function values of two
            # adjacent solutions.
            for j in range(1, len(front) - 1):
                distance = front[j + 1].loss - front[j - 1].loss

                # Check if minimum and maximum are the same (in which case do nothing)
                if objective_maxn - objective_minn == 0:
                    pass #undefined
                else:
                    distance = distance / float(objective_maxn - objective_minn)
                # The overall crowding-distance value is calculated as the sum of
                # individual distance values corresponding to each objective.
                front[j]._meta['crowding_distance'] += distance


    def compute_distance(self, front: tp.List[p.Parameter]):
        """This function assigns the crowding distance to the solutions.
        :param front: The list of solutions.
        """
        size = len(front)

        if size == 0:
            return
        # The boundary solutions (solutions with smallest and largest function values)
        # are set to an infinite (maximum) distance value
        if size == 1:
            front[0]._meta['crowding_distance'] = float("inf")
            return
        if size == 2:
            front[0]._meta['crowding_distance'] = float("inf")
            front[1]._meta['crowding_distance'] = float("inf")
            return

        for i in range(len(front)):
            front[i]._meta['crowding_distance'] = 0.0

        if isinstance(front[0].loss, np.ndarray):
            number_of_objectives = len(front[0].loss)
        else:
            number_of_objectives = 1

        for i in range(number_of_objectives):
            self.accumulate_distance_per_objective(front, i)

    def sort(self, candidates: tp.List[p.Parameter], in_place: bool = True) -> tp.List[p.Parameter]:
        if in_place:
            candidates.sort(key=lambda elem: elem._meta['crowding_distance'])
        return sorted(candidates, key=lambda elem: elem._meta['crowding_distance'])


class FastNonDominatedRanking:
    """ Non-dominated ranking of NSGA-II proposed by Deb et al., see [Deb2002] """

    def __init__(self):
        super(FastNonDominatedRanking, self).__init__()


    def compare(self, candidate1: p.Parameter, candidate2: p.Parameter) -> int:
        """ Compare the domainance relation of two candidates.

        :param candidate1: Candidate.
        :param candidate2: Candidate.
        """
        one_wins = np.sum(candidate1.losses < candidate2.losses)
        two_wins = np.sum(candidate2.losses < candidate1.losses)
        if one_wins > two_wins:
            return -1
        if two_wins > one_wins:
            return 1
        return 0


    def compute_ranking(self, candidates: tp.Dict[str, p.Parameter], k: int = None) -> tp.List[tp.List[p.Parameter]]:
        """ Compute ranking of candidates.

        :param candidates: Dict of candidates.
        :param k: Number of individuals.
        """
        n_cand: int = len(candidates)
        # dominated_by_cnt[i]: number of candidates dominating ith candidate
        dominated_by_cnt = [0] * n_cand #[0 for _ in range(len(candidates))]

        # candidates_dominated[i]: List of candidates dominated by ith candidate
        candidates_dominated = [[] for _ in range(n_cand)]

        # front[i] contains the list of solutions belonging to front i
        front = [[] for _ in range(n_cand + 1)]

        uids = list(candidates)
        for c1 in range(n_cand - 1):
            uid1 = uids[c1]
            for c2 in range(c1 + 1, n_cand):
                uid2 = uids[c2]
                dominance_test_result = self.compare(candidates[uid1], candidates[uid2])
                #self.number_of_comparisons += 1
                if dominance_test_result == -1:
                    # c1 wins
                    candidates_dominated[c1].append(c2)
                    dominated_by_cnt[c2] += 1
                elif dominance_test_result == 1:
                    # c2 wins
                    candidates_dominated[c2].append(c1)
                    dominated_by_cnt[c1] += 1

        # Formation of front[0], i.e. candidates that do not dominated by others
        front[0] = [c1 for c1 in range(n_cand) if dominated_by_cnt[c1] == 0]
        last_fronts = 0
        while len(front[last_fronts]) != 0:
            last_fronts += 1
            # Number of candidates in a frontier <= Number of candidates that dominate at least 1 candidate
            assert len(front[last_fronts - 1]) <= len(candidates_dominated)
            for c1 in front[last_fronts - 1]:
                for c2 in candidates_dominated[c1]:
                    dominated_by_cnt[c2] -= 1
                    if dominated_by_cnt[c2] == 0:
                        front[last_fronts].append(c2)

        # Convert index to uid
        # Trim to frontiers that contain the k candidates of interest
        ranked_sublists = []
        count = 0
        for front_i in range(last_fronts):
            count += len(front[front_i])
            if (k is not None) and (count >= k):
                ranked_sublists.append([candidates[uids[i]] for i in front[front_i]])
                break
            ranked_sublists.append([candidates[uids[i]] for i in front[front_i]])

        return ranked_sublists
        