# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import scipy
import scipy.signal
import scipy.stats
import time
import copy
import numpy as np
import itertools
from joblib import Parallel, delayed  # type: ignore

# import matplotlib as mpl
# import matplotlib.pyplot as plt
from collections import defaultdict
import nevergrad as ng

# pylint: skip-file

default_budget = 3000  # centiseconds
methods = {}
metrics = {}

# A few helper functions.
def normalize(x):
    for i in range(len(x)):
        normalization = np.sqrt(np.sum(x[i] ** 2.0))
        if normalization > 0.0:
            x[i] = x[i] / normalization
        else:
            x[i] = np.random.randn(np.prod(x[i].shape)).reshape(x[i].shape)
            x[i] = x[i] / np.sqrt(np.sum(x[i] ** 2.0))
    return x


def convo(x, k):
    if k is None:
        return x
    return scipy.ndimage.gaussian_filter(x, sigma=list(k) + [0.0] * (len(x.shape) - len(k)))


# Our well distributed point configurations.
def pure_random(n, shape, conv=None):
    return normalize([np.random.randn(*shape) for i in range(n)])


def antithetic_pm(n, shape, conv=None):
    m = n // 2
    x = [np.random.randn(*shape) for i in range(m)]
    x = normalize(x)
    x = x + [-xi for xi in x]
    if len(x) < n:
        x = x + [np.random.randn(*shape)]
    return np.array(x)


def antithetic_order(n, shape, axis=-1, also_sym=False, conv=None):
    x = []
    s = shape[axis]
    indices = [slice(0, s, 1) for s in shape]
    indices_sym = [slice(0, s, 1) for s in shape]
    while len(x) < n:
        icx = normalize([np.random.randn(*shape)])[0]
        for p in itertools.permutations(range(s)):
            if len(x) < n:
                indices[axis] = p
                cx = copy.deepcopy(icx)[tuple(indices)]
                x = x + [cx]
                if also_sym:
                    order = list(itertools.product((False, True), repeat=s))
                    np.random.shuffle(order)
                    for ordering in order:  # Ordering is a list of bool
                        if any(ordering) and len(x) < n:
                            scx = copy.deepcopy(cx)
                            for o in [i for i, o in enumerate(ordering) if o]:  # we must symetrize o
                                indices_sym[axis] = o
                                scx[tuple(indices_sym)] = -scx[tuple(indices_sym)]
                            x = x + [scx]
    return x


def antithetic_order_and_sign(n, shape, axis=-1, conv=None):
    return antithetic_order(n, shape, axis, also_sym=True)


def greedy_dispersion(n, shape, budget=default_budget, conv=None):
    x = normalize([np.random.randn(*shape)])

    for i in range(n - 1):
        bigdist = -1
        t0 = time.time()
        while time.time() < t0 + 0.01 * budget / n:
            y = normalize([np.random.randn(*shape)])[0]
            dist = min(np.linalg.norm(convo(y, conv) - convo(x[i], conv)) for i in range(len(x)))
            if dist > bigdist:
                bigdist = dist
                newy = y
        x += [newy]
    return x


def dispersion(n, shape, budget=default_budget, conv=None):
    x = greedy_dispersion(n, shape, budget / 2, conv=conv)
    t0 = time.time()
    num = n
    while time.time() < t0 + 0.01 * budget / 2:
        num = num + 1
        for j in range(len(x)):
            bigdist = -1
            for idx in range(2 * num):
                if idx > 0:
                    y = normalize([np.random.randn(*shape)])[0]
                else:
                    y = x[j]
                convoy = convo(y, conv)
                dist = min(np.linalg.norm(convoy - convo(x[i], conv)) for i in range(len(x)) if i != j)
                if dist > bigdist:
                    x[j] = y
                    bigdist = dist
    return x


def dispersion_with_conv(n, shape, budget=default_budget):
    return dispersion(n, shape, budget=budget, conv=[8, 8])


def greedy_dispersion_with_conv(n, shape, budget=default_budget):
    return greedy_dispersion(n, shape, budget=budget, conv=[8, 8])


def dispersion_with_big_conv(n, shape, budget=default_budget):
    return dispersion(n, shape, budget=budget, conv=[24, 24])


def greedy_dispersion_with_big_conv(n, shape, budget=default_budget):
    return greedy_dispersion(n, shape, budget=budget, conv=[24, 24])


def dispersion_with_mini_conv(n, shape, budget=default_budget):
    return dispersion(n, shape, budget=budget, conv=[2, 2])


def greedy_dispersion_with_mini_conv(n, shape, budget=default_budget):
    return greedy_dispersion(n, shape, budget=budget, conv=[2, 2])


def block_symmetry(n, shape, num_blocks=None):
    x = []
    if num_blocks is None:
        num_blocks = [4, 4]
    for pindex in range(n):
        # print(f"block symmetry {pindex}/{n}")
        newx = normalize([np.random.randn(*shape)])[0]
        s = np.prod(num_blocks)
        num_blocks = num_blocks + ([1] * (len(shape) - len(num_blocks)))
        order = list(itertools.product((False, True), repeat=s))
        np.random.shuffle(order)
        ranges = [list(range(n)) for n in num_blocks]
        for o in order:  # e.g. o is a list of 64 bool if num_blocks is 8x8
            # print(f"We have the boolean vector {o}")
            tentativex = copy.deepcopy(newx)
            for i, multi_index in enumerate(itertools.product(*ranges)):
                if o[i]:  # This multi-index corresponds to a symmetry.
                    # print("our multi-index is ", multi_index)
                    slices = [[]] * len(shape)
                    for c, p in enumerate(multi_index):
                        assert p >= 0
                        assert p < num_blocks[c]
                        a = p * shape[c] // num_blocks[c]
                        b = min((p + 1) * shape[c] // num_blocks[c], shape[c])
                        slices[c] = slice(a, b)
                    slices = tuple(slices)
                    # print(slices)
                    tentativex[slices] = -tentativex[slices]
            if len(x) >= n:
                return x
            x += [tentativex]


def big_block_symmetry(n, shape):
    return block_symmetry(n, shape, num_blocks=[2, 2])


def covering(n, shape, budget=default_budget, conv=None):
    x = greedy_dispersion_with_conv(n, shape, budget / 2, conv)
    mindists = []
    c = 0.01
    previous_score = float("inf")
    num = 0
    t0 = time.time()
    while time.time() < t0 + 0.01 * budget / 2:
        num = num + 1
        t = normalize([np.random.randn(*shape)])[0]
        convt = convo(t, conv)
        mindist = float("inf")
        for k in range(len(x)):
            dist = np.linalg.norm(convt - convo(x[k], conv))
            if dist < mindist:
                mindist = dist
                index = k
        mindists += [mindist]
        if len(mindists) % 2000 == 0:
            score = np.sum(mindists[-2000:]) / len(mindists[-2000:])
            c *= 2 if score < previous_score else 0.5
            previous_score = score
            # print(score, c)
        x[index] = normalize([x[index] + (c / (35 + n + np.sqrt(num))) * (t - x[index])])[0]
    # print("covering:", scipy.ndimage.gaussian_filter(mindists, budget / 3, mode='reflect')[::len(mindists) // 7])
    return x


def covering_conv(n, shape, budget=default_budget):
    return covering(n, shape, budget, conv=[8, 8])


def covering_mini_conv(n, shape, budget=default_budget):
    return covering(n, shape, budget, conv=[2, 2])


def get_class(x, num_blocks, just_max):
    shape = x.shape
    split_volume = len(num_blocks)
    num_blocks = num_blocks + [1] * (len(shape) - len(num_blocks))
    ranges = [list(range(n)) for n in num_blocks]
    result = []
    for _, multi_index in enumerate(itertools.product(*ranges)):
        slices = [[]] * len(shape)
        for c, p in enumerate(multi_index):
            assert p >= 0
            assert p < num_blocks[c]
            a = p * shape[c] // num_blocks[c]
            b = min((p + 1) * shape[c] // num_blocks[c], shape[c])
            slices[c] = slice(a, b)
        slices = tuple(slices)
        if just_max:
            result = result + [
                list(np.argsort((np.sum(x[slices], tuple(range(split_volume)))).flatten()))[-1]
            ]
        else:
            result = result + list(np.argsort((np.sum(x[slices], tuple(range(split_volume)))).flatten()))
    return hash(str(result))


def jittered(n, shape, num_blocks=None, just_max=False):
    if num_blocks is None:
        num_blocs = [2, 2]
    hash_to_set = defaultdict(list)
    for i in range(int(np.sqrt(n)) * n):
        x = normalize([np.random.randn(*shape)])[0]
        hash_to_set[get_class(x, num_blocks, just_max)] += [x]
    min_num = 10000000
    max_num = -1
    while True:
        for k in hash_to_set.keys():
            min_num = min(min_num, len(hash_to_set[k]))
            max_num = max(max_num, len(hash_to_set[k]))
        # print(f"min={min_num}, max={max_num}")
        if min_num < n / len(hash_to_set.keys()):
            x = normalize([np.random.randn(*shape)])[0]
            hash_to_set[get_class(x, num_blocks, just_max)] += [x]
        else:
            break
    x = []
    while len(x) < n:
        num = max(1, (n - len(x)) // len(hash_to_set))
        for k in hash_to_set.keys():
            if len(x) < n:
                # print(f"Adding {num} in {k}...")
                x += hash_to_set[k][:num]
            hash_to_set[k] = hash_to_set[k][num:]
    assert len(x) == n
    # print(f"Jittered used {len(hash_to_set)} classes")
    return x


def reduced_jittered(n, shape):
    return jittered(n, shape, [2, 2], just_max=True)


def covering_big_conv(n, shape, budget=default_budget):
    return covering(n, shape, budget, [24, 24])


def lhs(n, shape):
    num = np.prod(shape)
    x = np.zeros([n, num])
    for i in range(num):
        xb = (1.0 / n) * np.random.rand(n)
        xplus = np.linspace(0, n - 1, n) / n
        np.random.shuffle(xplus)
        x[:, i] = scipy.stats.norm.ppf(xb + xplus)
    thex = []
    for i in range(n):
        thex += normalize([x[i].reshape(*shape)])
    assert len(thex) == n
    assert thex[0].shape == tuple(shape), f" we get {x[0].shape} instead of {tuple(shape)}"
    return thex


# list_for_drawing = [
#     "lhs",
#     "reduced_jittered",
#     "jittered",
#     "big_block_symmetry",
#     "block_symmetry",
#     "greedy_dispersion",
#     "dispersion",
#     "pure_random",
#     "antithetic_pm",
#     "dispersion",
#     "antithetic_order",
#     "antithetic_order_and_sign",
#     "covering_conv",
#     "covering",
# ]
#
#
# def show_points(x, k):
#     plt.clf()
#     # fig = plt.figure()
#     ##ax = fig.gca(projection='3d')
#     # ax = fig.add_subplot(projection='3d')
#     # ax.set_aspect("equal")
#     # ax.view_init(elev=0., azim=0.)
#
#     def make_ball(a, b, c, r, col="r"):
#         # u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
#         # x = 0.5 + 0.5 * (a+r*np.cos(u)*np.sin(v))
#         # y = 0.5 + 0.5 * (b+r*np.sin(u)*np.sin(v))
#         # z = 0.5 + 0.5 * (c+r*np.cos(v))
#         # ax.plot_wireframe(x, y, z, color=col, zorder=z)
#         u = np.linspace(0, 2 * 3.14159, 40)
#         x = 0.5 + 0.5 * (a + r * np.cos(u))
#         y = 0.5 + 0.5 * (b + r * np.sin(u))
#         plt.plot(x, y, color=col)
#         plt.plot(
#             [0.5, 0.5 + 0.5 * a], [0.5, 0.5 + 0.5 * b], color=col, linestyle=("dashed" if c > 0 else "solid")
#         )
#
#     for i in range(len(x)):
#         if x[i][2] > 0:
#             make_ball(x[i][0], x[i][1], x[i][2], 0.05 * (3 / (3 + x[i][2])), "g")
#     make_ball(0, 0, 0, 1, "r")
#     for i in range(len(x)):
#         if x[i][2] <= 0:
#             make_ball(x[i][0], x[i][1], x[i][2], 0.05 * (3 / (3 + x[i][2])), "b")
#     # plt.show()
#     plt.tight_layout()
#     plt.savefig(f"sphere3d_{k}.png")


# Let us play with metrics.
def metric_half(x, budget=default_budget, conv=None):
    shape = x[0].shape
    t0 = time.time()
    xconv = np.array([convo(x_, conv).flatten() for x_ in x])
    scores = []
    while time.time() < t0 + 0.01 * budget:
        y = convo(normalize([np.random.randn(*shape)])[0], conv).flatten()
        scores += [np.average(np.matmul(xconv, y) > 0.0)]
    return np.average((np.array(scores) - 0.5) ** 2)


def metric_half_conv(x, budget=default_budget):
    return metric_half(x, budget, conv=[8, 8])


def metric_cap(x, budget=default_budget, conv=None):
    shape = x[0].shape
    t0 = time.time()
    c = 1.0 / np.sqrt(len(x[0].flatten()))
    xconv = np.array(normalize([convo(x_, conv).flatten() for x_ in x]))
    scores = []
    while time.time() < t0 + 0.01 * budget:
        y = convo(normalize([np.random.randn(*shape)])[0], conv).flatten()
        scores += [np.average(np.matmul(xconv, y) > c)]
        scores += [np.average(np.matmul(xconv, y) < -c)]
    return np.std(np.array(scores))


def metric_cap_conv(x, budget=default_budget):
    return metric_cap(x, budget, conv=[8, 8])


def metric_pack_avg(x, budget=default_budget, conv=None):
    shape = x[0].shape
    xconv = np.array(normalize([convo(x_, conv).flatten() for x_ in x]))
    scores = np.matmul(xconv, xconv.transpose())
    for i in range(len(scores)):
        assert 0.99 < scores[i, i] < 1.01
        scores[i, i] = 0
    scores = scores.flatten()
    assert len(scores) == len(x) ** 2
    return np.average(scores)


def metric_pack_avg_conv(x, budget=default_budget):
    return metric_pack_avg(x, budget=default_budget, conv=[8, 8])


def metric_pack(x, budget=default_budget, conv=None):
    shape = x[0].shape
    xconv = np.array(normalize([convo(x_, conv).flatten() for x_ in x]))
    scores = np.matmul(xconv, xconv.transpose())
    for i in range(len(scores)):
        assert 0.99 < scores[i, i] < 1.01, "we get score " + str(scores[i, i])
        scores[i, i] = 0
    scores = scores.flatten()
    assert len(scores) == len(x) ** 2
    return max(scores)


def metric_pack_conv(x, budget=default_budget):
    return metric_pack(x, budget=default_budget, conv=[8, 8])


list_of_methods = [
    "rs_ng_TwoPointsDE",
    "rs_ng_DE",
    "rs_ng_PSO",
    "rs_ng_OnePlusOne",
    "rs_ng_DiagonalCMA",
    "lhs",
    "reduced_jittered",
    "jittered",
    "big_block_symmetry",
    "block_symmetry",
    "greedy_dispersion",
    "dispersion",
    "pure_random",
    "antithetic_pm",
    "dispersion",
    "antithetic_order",
    "antithetic_order_and_sign",
    "dispersion_with_conv",
    "dispersion_with_big_conv",
    "greedy_dispersion_with_big_conv",
    "dispersion_with_mini_conv",
    "greedy_dispersion_with_mini_conv",
    "covering",
    "covering_conv",
    "covering_mini_conv",
    "rs_metric",
    "rs_metric_mhc",
    "rs_metric_pack",
    "rs_metric_pa",
    "rs_metric_pc",
    "rs_metric_pac",
    "rs_metric_cap",
    "rs_metric_cc",
    "rs_metric_all",
]
list_metrics = [
    "metric_half",
    "metric_half_conv",
    "metric_pack",
    "metric_pack_conv",
    "metric_pack_avg",
    "metric_pack_avg_conv",
    "metric_cap",
    "metric_cap_conv",
]
for u in list_metrics:
    metrics[u] = eval(u)


def rs_metric(n, shape, budget=default_budget, k="metric_half", ngtool=None):
    t0 = time.time()
    bestm = float("inf")
    if ngtool is not None:
        opt = ng.optimizers.registry[ngtool](
            ng.p.Array(shape=tuple([n] + list(shape))), budget=10000000000000
        )
    while time.time() < t0 + 0.01 * budget:
        if ngtool is None:
            x = pure_random(n, shape)
        else:
            candidate = opt.ask()
            x = list(candidate.value)
            assert len(x) == n
            x = normalize(x)
        # m = eval(f"{k}(x, budget={budget/max(10,np.sqrt(budget/100))})")
        if k == "all":
            m = np.sum(
                [
                    metrics[k2](x, (budget / len(list_metrics)) / max(10, np.sqrt(budget / 100)))
                    for k2 in list_metrics
                ]
            )
        else:
            m = metrics[k](x, budget / max(10, np.sqrt(budget / 100)))
        if ngtool is not None:
            print("ng gets ", m)
            opt.tell(candidate, m)
        if m < bestm:
            bestm = m
            bestx = x
    return bestx


def rs_metric_mhc(n, shape, budget=default_budget):
    return rs_metric(n, shape, budget, k="metric_half_conv")


def rs_metric_cap(n, shape, budget=default_budget):
    return rs_metric(n, shape, budget, k="metric_cap")


def rs_metric_cc(n, shape, budget=default_budget):
    return rs_metric(n, shape, budget, k="metric_cap_conv")


def rs_metric_pack(n, shape, budget=default_budget):
    return rs_metric(n, shape, budget, k="metric_pack")


def rs_metric_pa(n, shape, budget=default_budget):
    return rs_metric(n, shape, budget, k="metric_pack_avg")


def rs_metric_pc(n, shape, budget=default_budget):
    return rs_metric(n, shape, budget, k="metric_pack_conv")


def rs_metric_pac(n, shape, budget=default_budget):
    return rs_metric(n, shape, budget, k="metric_pack_avg_conv")


def rs_metric_all(n, shape, budget=default_budget):
    return rs_metric(n, shape, budget, k="all")


def rs_ng_TwoPointsDE(n, shape, budget=default_budget):
    return rs_metric(n, shape, budget, k="all", ngtool="TwoPointsDE")


def rs_ng_DE(n, shape, budget=default_budget):
    return rs_metric(n, shape, budget, k="all", ngtool="DE")


def rs_ng_PSO(n, shape, budget=default_budget):
    return rs_metric(n, shape, budget, k="all", ngtool="PSO")


def rs_ng_OnePlusOne(n, shape, budget=default_budget):
    return rs_metric(n, shape, budget, k="all", ngtool="OnePlusOne")


def rs_ng_DiagonalCMA(n, shape, budget=default_budget):
    return rs_metric(n, shape, budget, k="all", ngtool="DiagonalCMA")


data = defaultdict(lambda: defaultdict(list))  # type: ignore


# def do_plot(tit, values):
#     plt.clf()
#     plt.title(tit.replace("_", " "))
#     x = np.cos(np.linspace(0.0, 2 * 3.14159, 20))
#     y = np.sin(np.linspace(0.0, 2 * 3.14159, 20))
#     for i, v in enumerate(sorted(values.keys(), key=lambda k: np.average(values[k]))):
#         print(f"context {tit}, {v} ==> {values[v]}")
#         plt.plot([i + r for r in x], [np.average(values[v]) + r * np.std(values[v]) for r in y])
#         plt.text(i, np.average(values[v]) + np.std(values[v]), f"{v}", rotation=30)
#         if i > 0:
#             plt.savefig(
#                 f"comparison_{tit}_time{default_budget}.png".replace(" ", "_")
#                 .replace("]", "_")
#                 .replace("[", "_")
#             )
#
#
# def heatmap(y, x, table, name):
#     for cn in ["viridis", "plasma", "inferno", "magma", "cividis"]:
#         print(f"Creating a heatmap with name {name}: {table}")
#         plt.clf()
#         fig, ax = plt.subplots()
#         tab = copy.deepcopy(table)
#
#         for j in range(len(tab[0, :])):
#             for i in range(len(tab)):
#                 tab[i, j] = np.average(table[:, j] < table[i, j])
#         print(tab)
#         im = ax.imshow(tab, aspect="auto", cmap=mpl.colormaps[cn])
#
#         # Show all ticks and label them with the respective list entries
#         ax.set_xticks(np.arange(len(x)), labels=x)
#         ax.set_yticks(np.arange(len(y)), labels=y)
#
#         # Rotate the tick labels and set their alignment.
#         plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
#         # Loop over data dimensions and create text annotations.
#         for i in range(len(y)):
#             for j in range(len(x)):
#                 text = ax.text(j, i, str(tab[i, j])[:4], ha="center", va="center", color="k")
#
#         fig.tight_layout()
#         plt.savefig(f"TIME{default_budget}_cmap{cn}" + name)
#
#
# def create_statistics(n, shape, list_of_methods, list_of_metrics, num=1):
#     for _ in range(num):
#         for idx, method in enumerate(list_of_methods):
#             print(f" {idx}/{len(list_of_methods)}: {method}")
#             x = eval(f"{method}(n, shape)")
#             for k in list_of_metrics:
#                 m = eval(f"{k}(x)")
#                 # xr = pure_random(n, shape)
#                 # mr = eval(f"{k}(xr)")
#                 print(f"{k} --> {m}")  # , vs   {mr} for random")
#                 data[k][method] += [m]
#                 if len(data[k]) > 1:
#                     do_plot(k + f"_number{n}_shape{shape}", data[k])
#         # Now let's do a heatmap
#         tab = np.array([[np.average(data[k][method]) for k in list_of_metrics] for method in list_of_methods])
#         print("we have ", tab)
#         heatmap(
#             list_of_methods,
#             list_of_metrics,
#             tab,
#             str(shape) + "_" + str(n) + "_" + str(np.random.randint(50000)) + "_bigartifcompa.png",
#         )


for u in list_of_methods:
    methods[u] = eval(u)


def parallel_create_statistics(n, shape, list_of_methods, list_of_metrics, num=1):
    shape = [int(s) for s in list(shape)]
    for _ in range(num):

        def deal_with_method(method):
            print(f"{method}")
            # x = eval(f"methods[method](n, shape)")
            x = methods[method](n, shape)
            np.array(x).tofile(
                f"pointset_{n}_{shape}_{method}_{default_budget}_{np.random.randint(50000)}.dat".replace(
                    " ", "_"
                )
                .replace("[", " ")
                .replace("]", " ")
            )
            print(f"{method}({n}, {shape}) created in time {default_budget}")
            metrics_values = []
            for k in list_of_metrics:
                # m = eval(f"{k}(x)")
                m = metrics[k](x)
                # print(f"{k} --> {m}")   #, vs   {mr} for random")
                metrics_values += [m]
                print(f"{method}({n}, {shape}) evaluated for {k}")
            return metrics_values

        # for method in list_of_methods:
        results = Parallel(n_jobs=70)(delayed(deal_with_method)(method) for method in list_of_methods)
        for i, method in enumerate(list_of_methods):
            for j, k in enumerate(list_of_metrics):
                data[k][method] += [results[i][j]]

        for k in list_of_metrics:
            if len(data[k]) > 1:
                do_plot(k + f"_number{n}_shape{shape}", data[k])
        # Now let's do a heatmap
        tab = np.array([[np.average(data[k][method]) for k in list_of_metrics] for method in list_of_methods])
        print("we have ", tab)
        # heatmap(list_of_methods, list_of_metrics, tab, "bigartifcompa.png")
        heatmap(
            list_of_methods,
            list_of_metrics,
            tab,
            str(shape) + "_" + str(n) + "_" + str(np.random.randint(50000)) + "_bigartifcompa.png",
        )
        # heatmap(list_of_methods, list_of_metrics, tab, str(shape) + "_" + str(n) + "_bigartifcompa.png")


# size=int(np.random.choice([1,1,1,64,32,16]))
# size=1
# channels = int(np.random.choice([2,3,4]))
# if size == 1:
# channels = np.random.choice([128, 16])
# numpoints = int(np.random.choice([12, 24, 48, 96, 192, 384]))
# print(f"Size={size}, Channels={channels}, numpoints={numpoints}")
# parallel_create_statistics(numpoints, [size,size,channels], num=1, list_of_metrics=list_metrics, list_of_methods=list_of_methods)
##create_statistics(40, [16,16,2])
# quit()
## First we play in dimension two.
# n=30
# for k in list_for_drawing:
#    print(f"Testing {k}")
#    plt.clf()
#    x = eval(f"{k}(n, [1, 2])")
#    assert len(x) == n, f"n={n}, x={x}"
#    for x_ in x:
#        assert 0.99 < np.linalg.norm(x_.flatten()) < 1.01, f"{k} fails with norm {np.linalg.norm(x_.flatten())}"
#    xx = np.array([x[i][0,0] for i in range(len(x))])
#    yy = np.array([x[i][0,1] for i in range(len(x))])
#    indices = np.argsort(xx)
#    xx = xx[indices]
#    yy = yy[indices]
#    nxx = []
#    nyy = []
#    for i in range(len(xx)):
#        nxx += [xx[i], 0]
#        nyy += [yy[i], 0]
#    plt.plot(nxx, nyy, label=k)
#    plt.savefig("sphere_" + k + ".png")
## Then dim 3.
# for k in list_for_drawing:
#    print(f"Testing {k} in dim 3")
#    x = eval(f"{k}(n, [3, 1])")
#    show_points(x, k)


def bigcheck():
    # Now let us go!
    n = 20  # Let us consider batch size 50
    shape = (8, 8, 3)  # Maybe this is what we need for now ?
    for k in [
        "lhs",
        "reduced_jittered",
        "jittered",
        "big_block_symmetry",
        "block_symmetry",
        "greedy_dispersion",
        "dispersion",
        "pure_random",
        "antithetic_pm",
        "dispersion",
        "antithetic_order",
        "antithetic_order_and_sign",
        "dispersion_with_conv",
        "dispersion_with_big_conv",
        "greedy_dispersion_with_big_conv",
        "dispersion_with_mini_conv",
        "greedy_dispersion_with_mini_conv",
        "covering",
        "covering_conv",
        "covering_mini_conv",
    ]:
        print("Starting to play with ", k)
        eval(f"{k}(n, shape)")
        print(f" {k} has been used for generating a batch of {n} points with shape {shape}")


# bigcheck()


def get_a_point_set(n, shape, method=None):
    k = np.random.choice(list_of_methods)
    if method is not None:
        assert method in list_of_methods
        k = method
    print("Working with ", k)
    x = eval(f"{k}({n}, {shape})")
    for i in range(len(x)):
        assert 0.999 < np.linalg.norm(x[i]) < 1.001, "we have norm " + str(np.linalg.norm(x[i]))
    np.array(x).tofile(
        f"pointset_{n}_{shape}_{method}_{default_budget}_{np.random.randint(50000)}.dat".replace(" ", "_")
        .replace("[", " ")
        .replace("]", " ")
    )
    return k, x


# k, x = get_a_point_set(50, (64, 64, 4))


def quasi_randomize(pointset, method):
    n = len(pointset)
    shape = [int(i) for i in list(pointset[0].shape)]
    norms = [np.linalg.norm(pointset[i]) for i in range(n)]
    if method == "none":
        if len(shape) > 1 and shape[0] > 5:
            x = dispersion(n, shape, conv=[int(s / 3) for s in list(shape)[:-1]])
        else:
            x = rs_ng_DiagonalCMA(n, shape)
    else:
        x = get_a_point_set(n, shape, method)
    x = normalize(x)
    for i in range(n):
        x[i] = norms[i] * x[i]
    return x
