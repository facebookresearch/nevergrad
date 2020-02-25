# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The source code in this module was created by Antoine Moreau
# from University Clermont Auvergne, CNRS, SIGMA Clermont, Institut Pascal.
#
# Publication:
# - Aliou Barry, Mamadou; Berthier, Vincent; Wilts, Bodo D.; Cambourieux, Marie-Claire; Pollès, Rémi;
#   Teytaud, Olivier; Centeno, Emmanuel; Biais, Nicolas; Moreau, Antoine (2018)
#   Evolutionary algorithms converge towards evolved biological photonic structures,
#   https://arxiv.org/abs/1808.04689
# - Defrance, J., Lemaître, C., Ajib, R., Benedicto, J., Mallet, E., Pollès, R., Plumey, J.-P.,
#   Mihailovic, M., Centeno, E., Ciracì, C., Smith, D.R. and Moreau, A., 2016.
#   Moosh: A Numerical Swiss Army Knife for the Optics of Multilayers in Octave/Matlab. Journal of Open Research Software, 4(1), p.e13.

import typing as tp
import numpy as np
from scipy.linalg import toeplitz
# pylint: disable=blacklisted-name,too-many-locals,too-many-arguments


def bragg(X: np.ndarray) -> float:
    lam = 600
    bar = int(np.size(X) / 2)
    n = np.concatenate(([1], np.sqrt(X[0:bar]), [1.7320508075688772]))
    Type = np.arange(0, bar + 2)
    hauteur = np.concatenate(([0], X[bar : 2 * bar], [0]))
    tmp = np.tan(2 * np.pi * n[Type] * hauteur / lam)
    # Specific to this substrate.
    Z = n[-1]
    for k in range(np.size(Type) - 1, 0, -1):
        Z = (Z - 1j * n[Type[k]] * tmp[k]) / (1 - 1j * tmp[k] * Z / n[Type[k]])
    # Specific to air.
    r = (1 - Z) / (1 + Z)
    c = np.real(1 - r * np.conj(r))
    return float(c)


def chirped(X: np.ndarray) -> float:
    lam = np.linspace(500, 800, 50)
    n = np.array([1, 1.4142135623730951, 1.7320508075688772])
    Type = np.concatenate(([0], np.tile([2, 1], int(np.size(X) / 2)), [2]))
    hauteur = np.concatenate(([0], X, [0]))
    r = np.zeros(np.size(lam)) + 0j
    for m in range(0, np.size(lam)):
        # Specific to this substrate.
        tmp = np.tan(2 * np.pi * n[Type] * hauteur / lam[m])
        Z = 1.7320508075688772
        for k in range(np.size(Type) - 1, 0, -1):
            Z = (Z - 1j * n[Type[k]] * tmp[k]) / (1 - 1j * tmp[k] * Z / n[Type[k]])
        # Specific to air.
        r[m] = (1 - Z) / (1 + Z)
    # c=1-np.mean(abs(r)**2)
    c = 1 - np.real(np.sum(r * np.conj(r)) / np.size(lam))
    return float(c)


def cascade(T: np.ndarray, U: np.ndarray) -> np.ndarray:
    n = int(T.shape[1] / 2)
    J = np.linalg.inv(np.eye(n) - np.matmul(U[0:n, 0:n], T[n : 2 * n, n : 2 * n]))
    K = np.linalg.inv(np.eye(n) - np.matmul(T[n : 2 * n, n : 2 * n], U[0:n, 0:n]))
    S = np.block(
        [
            [
                T[0:n, 0:n]
                + np.matmul(
                    np.matmul(np.matmul(T[0:n, n : 2 * n], J), U[0:n, 0:n]),
                    T[n : 2 * n, 0:n],
                ),
                np.matmul(np.matmul(T[0:n, n : 2 * n], J), U[0:n, n : 2 * n]),
            ],
            [
                np.matmul(np.matmul(U[n : 2 * n, 0:n], K), T[n : 2 * n, 0:n]),
                U[n : 2 * n, n : 2 * n]
                + np.matmul(
                    np.matmul(np.matmul(U[n : 2 * n, 0:n], K), T[n : 2 * n, n : 2 * n]),
                    U[0:n, n : 2 * n],
                ),
            ],
        ]
    )
    return S  # type: ignore


def c_bas(A: np.ndarray, V: np.ndarray, h: float) -> np.ndarray:
    n = int(A.shape[1] / 2)
    D = np.diag(np.exp(1j * V * h))
    S = np.block(
        [
            [A[0:n, 0:n], np.matmul(A[0:n, n : 2 * n], D)],
            [
                np.matmul(D, A[n : 2 * n, 0:n]),
                np.matmul(np.matmul(D, A[n : 2 * n, n : 2 * n]), D),
            ],
        ]
    )
    return S  # type: ignore


def marche(a: float, b: float, p: float, n: int, x: float) -> np.ndarray:
    l = np.zeros(n, dtype=np.complex)  # noqa
    m = np.zeros(n, dtype=np.complex)
    tmp = (
        1
        / (2 * np.pi * np.arange(1, n))
        * (np.exp(-2 * 1j * np.pi * p * np.arange(1, n)) - 1)
        * np.exp(-2 * 1j * np.pi * np.arange(1, n) * x)
    )
    l[1:n] = 1j * (a - b) * tmp
    l[0] = p * a + (1 - p) * b
    m[0] = l[0]
    m[1:n] = 1j * (b - a) * np.conj(tmp)
    T = toeplitz(l, m)
    return T  # type: ignore


def creneau(k0: float, a0: float, pol: float, e1: float, e2: float, a: float, n: int, x0: float) -> tp.Tuple[np.ndarray, np.ndarray]:
    nmod = int(n / 2)
    alpha = np.diag(a0 + 2 * np.pi * np.arange(-nmod, nmod + 1))
    if pol == 0:
        M = alpha * alpha - k0 * k0 * marche(e1, e2, a, n, x0)
        L, E = np.linalg.eig(M)
        L = np.sqrt(-L + 0j)
        L = (1 - 2 * (np.imag(L) < -1e-15)) * L
        P = np.block([[E], [np.matmul(E, np.diag(L))]])
    else:
        U = marche(1 / e1, 1 / e2, a, n, x0)
        T = np.linalg.inv(U)
        M = (
            np.matmul(
                np.matmul(np.matmul(T, alpha), np.linalg.inv(marche(e1, e2, a, n, x0))),
                alpha,
            )
            - k0 * k0 * T
        )
        L, E = np.linalg.eig(M)
        L = np.sqrt(-L + 0j)
        L = (1 - 2 * (np.imag(L) < -1e-15)) * L
        P = np.block([[E], [np.matmul(np.matmul(U, E), np.diag(L))]])
    return P, L


def homogene(k0: float, a0: float, pol: float, epsilon: float, n: int) -> tp.Tuple[np.ndarray, np.ndarray]:
    nmod = int(n / 2)
    valp = np.sqrt(
        epsilon * k0 * k0 - (a0 + 2 * np.pi * np.arange(-nmod, nmod + 1)) ** 2 + 0j
    )
    valp = valp * (1 - 2 * (valp < 0)) * (pol / epsilon + (1 - pol))
    P = np.block([[np.eye(n)], [np.diag(valp)]])
    return P, valp


def interface(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    n = int(P.shape[1])
    S = np.matmul(
        np.linalg.inv(
            np.block(
                [[P[0:n, 0:n], -Q[0:n, 0:n]], [P[n : 2 * n, 0:n], Q[n : 2 * n, 0:n]]]
            )
        ),
        np.block([[-P[0:n, 0:n], Q[0:n, 0:n]], [P[n : 2 * n, 0:n], Q[n : 2 * n, 0:n]]]),
    )
    return S  # type: ignore


def morpho(X: np.ndarray) -> float:
    lam = 449.5897
    pol = 1.0
    d = 600.521475
    nmod = 25
    # nmod=1
    e2 = 2.4336
    n = 2 * nmod + 1
    n_motifs = int(X.size / 4)
    X = X / d
    h = X[0:n_motifs]
    x0 = X[n_motifs : 2 * n_motifs]
    a = X[2 * n_motifs : 3 * n_motifs]
    spacers = X[3 * n_motifs : 4 * n_motifs]
    l = lam / d  # noqa
    k0 = 2 * np.pi / l
    P, V = homogene(k0, 0, pol, 1, n)
    S = np.block(
        [[np.zeros([n, n]), np.eye(n, dtype=np.complex)], [np.eye(n), np.zeros([n, n])]]
    )
    for j in range(0, n_motifs):
        Pc, Vc = creneau(k0, 0, pol, e2, 1, a[j], n, x0[j])
        S = cascade(S, interface(P, Pc))
        S = c_bas(S, Vc, h[j])
        S = cascade(S, interface(Pc, P))
        S = c_bas(S, V, spacers[j])
    Pc, Vc = homogene(k0, 0, pol, e2, n)
    S = cascade(S, interface(P, Pc))
    R = np.zeros(3, dtype=np.float)
    for j in range(-1, 2):
        R[j] = abs(S[j + nmod, nmod]) ** 2 * np.real(V[j + nmod]) / k0
    cost: float = 1 - (R[-1] + R[1]) / 2 + R[0] / 2

    lams = (np.array([400, 500, 600, 700, 800]) + 0.24587) / d
    bar = 0
    for lo in lams:
        k0 = 2 * np.pi / lo
        P, V = homogene(k0, 0, pol, 1, n)
        S = np.block(
            [
                [np.zeros([n, n], dtype=np.complex), np.eye(n)],
                [np.eye(n), np.zeros([n, n])],
            ]
        )
        for j in range(0, n_motifs):
            Pc, Vc = creneau(k0, 0, pol, e2, 1, a[j], n, x0[j])
            S = cascade(S, interface(P, Pc))
            S = c_bas(S, Vc, h[j])
            S = cascade(S, interface(Pc, P))
            S = c_bas(S, V, spacers[j])
        Pc, Vc = homogene(k0, 0, pol, e2, n)
        S = cascade(S, interface(P, Pc))
        bar += abs(S[nmod, nmod]) ** 2 * np.real(V[nmod]) / k0
    cost += bar / lams.size
    return cost
