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
from pathlib import Path
import numpy as np
from scipy.linalg import toeplitz

# pylint: disable=blacklisted-name,too-many-locals,too-many-arguments


def bragg(X: np.ndarray) -> float:
    """
    Cost function for the Bragg mirror problem: maximizing the reflection
    when the refractive index are given for all the layers.
    Input: a vector whose components represent each the thickness of each
    layer.
    https://hal.archives-ouvertes.fr/hal-02613161
    """
    lam = 600
    bar = int(np.size(X) / 2)
    n = np.concatenate(([1], np.sqrt(X[0:bar]), [1.7320508075688772]))
    type_ = np.arange(0, bar + 2)
    hauteur = np.concatenate(([0], X[bar : 2 * bar], [0]))
    tmp = np.tan(2 * np.pi * n[type_] * hauteur / lam)
    # Specific to this substrate.
    Z = n[-1]
    for k in range(np.size(type_) - 1, 0, -1):
        Z = (Z - 1j * n[type_[k]] * tmp[k]) / (1 - 1j * tmp[k] * Z / n[type_[k]])
    # Specific to air.
    r = (1 - Z) / (1 + Z)
    c = np.real(1 - r * np.conj(r))
    return float(c)


def chirped(X: np.ndarray) -> float:
    lam = np.linspace(500, 800, 50)
    n = np.array([1, 1.4142135623730951, 1.7320508075688772])
    type_ = np.concatenate(([0], np.tile([2, 1], int(np.size(X) / 2)), [2]))
    hauteur = np.concatenate(([0], X, [0]))
    r = np.zeros(np.size(lam)) + 0j
    for m in range(0, np.size(lam)):
        # Specific to this substrate.
        tmp = np.tan(2 * np.pi * n[type_] * hauteur / lam[m])
        Z = 1.7320508075688772
        for k in range(np.size(type_) - 1, 0, -1):
            Z = (Z - 1j * n[type_[k]] * tmp[k]) / (1 - 1j * tmp[k] * Z / n[type_[k]])
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


def creneau(
    k0: float, a0: float, pol: float, e1: float, e2: float, a: float, n: int, x0: float
) -> tp.Tuple[np.ndarray, np.ndarray]:
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
    valp = np.sqrt(epsilon * k0 * k0 - (a0 + 2 * np.pi * np.arange(-nmod, nmod + 1)) ** 2 + 0j)
    valp = valp * (1 - 2 * (valp < 0)) * (pol / epsilon + (1 - pol))
    P = np.block([[np.eye(n)], [np.diag(valp)]])
    return P, valp


def interface(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    n = int(P.shape[1])
    S = np.matmul(
        np.linalg.inv(np.block([[P[0:n, 0:n], -Q[0:n, 0:n]], [P[n : 2 * n, 0:n], Q[n : 2 * n, 0:n]]])),
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
    S = np.block([[np.zeros([n, n]), np.eye(n, dtype=np.complex)], [np.eye(n), np.zeros([n, n])]])
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


i = complex(0, 1)


def epscSi(lam: np.ndarray) -> np.ndarray:
    a = np.arange(250, 1500, 5)
    e = np.load(
        Path(__file__).with_name("epsilon_epscSi.npy")
    )  # saved with np.save(filename, e) and dumped in this folder
    y = np.argmin(np.sign(lam - a))
    y = y - 1
    epsilon = (e[y + 1] - e[y]) / (a[y + 1] - a[y]) * (lam - a[y]) + e[y]
    return epsilon  # type: ignore


def cascade2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    This function takes two 2x2 matrices A and B, that are assumed to be scattering matrices
    and combines them assuming A is the "upper" one, and B the "lower" one, physically.
    The result is a 2x2 scattering matrix.
    """
    t = 1 / (1 - B[0, 0] * A[1, 1])
    S = np.zeros((2, 2), dtype=complex)
    S[0, 0] = A[0, 0] + A[0, 1] * B[0, 0] * A[1, 0] * t
    S[0, 1] = A[0, 1] * B[0, 1] * t
    S[1, 0] = B[1, 0] * A[1, 0] * t
    S[1, 1] = B[1, 1] + A[1, 1] * B[0, 1] * B[1, 0] * t
    return S


def solar(lam: np.ndarray) -> np.ndarray:
    # saved with np.save(filename, e) and dumped in this folder
    a = np.load(Path(__file__).with_name("wavelength_solar.npy"))
    e = np.load(Path(__file__).with_name("epsilon_solar.npy"))
    jsc = np.interp(lam, a, e)
    return jsc  # type: ignore


def absorption(
    lam: float,
    epsilon: np.ndarray,
    mu: np.ndarray,
    type_: np.ndarray,
    hauteur: np.ndarray,
    pol: int,
    theta: float,
) -> np.ndarray:
    f = mu if not pol else epsilon
    k0 = 2 * np.pi / lam
    g = type_.size
    alpha = np.sqrt(epsilon[type_[0]] * mu[type_[0]]) * k0 * np.sin(theta)
    gamma = np.sqrt(epsilon[type_] * mu[type_] * k0 ** 2 - np.ones(g) * alpha ** 2)
    if np.real(epsilon[type_[0]]) < 0 and np.real(mu[type_[0]]) < 0:
        gamma[0] = -gamma[0]
    if g > 2:
        gamma[1 : g - 2] = gamma[1 : g - 2] * (1 - 2 * (np.imag(gamma[1 : g - 2]) < 0))
    if (
        np.real(epsilon[type_[g - 1]]) < 0
        and np.real(mu[type_[g - 1]]) < 0
        and np.real(np.sqrt(epsilon[type_[g - 1]] * mu[type_[g - 1]] * k0 ** 2 - alpha ** 2)) != 0
    ):
        gamma[g - 1] = -np.sqrt(epsilon[type_[g - 1]] * mu[type_[g - 1]] * k0 ** 2 - alpha ** 2)
    else:
        gamma[g - 1] = np.sqrt(epsilon[type_[g - 1]] * mu[type_[g - 1]] * k0 ** 2 - alpha ** 2)
    T = np.zeros(((2 * g, 2, 2)), dtype=complex)
    T[0] = [[0, 1], [1, 0]]
    for k2 in range(g - 1):
        t = np.exp(i * gamma[k2] * hauteur[k2])
        T[2 * k2 + 1] = [[0, t], [t, 0]]
        # Interface scattering matrix
        b1 = gamma[k2] / f[type_[k2]]
        b2 = gamma[k2 + 1] / f[type_[k2 + 1]]
        T[2 * k2 + 2] = [
            [(b1 - b2) / (b1 + b2), 2 * b2 / (b1 + b2)],
            [2 * b1 / (b1 + b2), (b2 - b1) / (b1 + b2)],
        ]
        t = np.exp(i * gamma[g - 1] * hauteur[g - 1])
        T[2 * g - 1] = [[0, t], [t, 0]]
    H = np.zeros(((2 * g - 1, 2, 2)), dtype=complex)
    A = np.zeros(((2 * g - 1, 2, 2)), dtype=complex)
    H[0] = T[2 * g - 1]
    A[0] = T[0]
    for j in range(2 * g - 2):
        A[j + 1] = cascade2(A[j], T[j + 1])
        H[j + 1] = cascade2(T[2 * g - 2 - j], H[j])
    # r = A[len(A) - 1][0, 0]  # TODO: unused
    t = A[len(A) - 1][1, 0]
    I = np.zeros(((2 * g, 2, 2)), dtype=complex)  # noqa
    for j in range(len(T) - 1):
        I[j][0, 0] = A[j][1, 0] / (1 - A[j][1, 1] * H[len(T) - 2 - j][0, 0])
        I[j][0, 1] = A[j][1, 1] * H[len(T) - 2 - j][0, 1] / (1 - A[j][1, 1] * H[len(T) - 2 - j][0, 0])
        I[j][1, 0] = A[j][1, 0] * H[len(T) - 2 - j][0, 0] / (1 - A[j][1, 1] * H[len(T) - 2 - j][0, 0])
        I[j][1, 1] = H[len(T) - 2 - j][0, 1] / (1 - A[j][1, 1] * H[len(T) - 2 - j][0, 0])
    I[2 * g - 1][0, 0] = I[2 * g - 2][0, 0] * np.exp(i * gamma[g - 1] * hauteur[g - 1])
    I[2 * g - 1][0, 1] = I[2 * g - 2][0, 1] * np.exp(i * gamma[g - 1] * hauteur[g - 1])
    I[2 * g - 1][1, 0] = 0
    I[2 * g - 1][1, 1] = 0
    w = 0
    poynting = np.zeros(2 * g, dtype=complex)
    if pol == 0:  # TE
        for j in range(2 * g):
            poynting[j] = np.real(
                (I[j][0, 0] + I[j][1, 0])
                * np.conj((I[j][0, 0] - I[j][1, 0]) * gamma[w] / mu[type_[w]])
                * mu[type_[0]]
                / (gamma[0])
            )
            w = w + 1 - np.mod(j + 1, 2)
    else:  # TM
        for j in range(2 * g):
            poynting[j] = np.real(
                (I[j][0, 0] - I[j][1, 0])
                * np.conj((I[j][0, 0] + I[j][1, 0]) * gamma[w] / epsilon[type_[w]])
                * epsilon[type_[0]]
                / (gamma[0])
            )
            w = w + 1 - np.mod(j + 1, 2)
    tmp = abs(-np.diff(poynting))
    absorb = tmp[np.arange(0, 2 * g, 2)]
    return absorb  # type: ignore


def cf_photosic_reference(X: np.ndarray) -> float:
    """vector X is only the thicknesses of each layers, because the materials (so the epislon)
    are imposed by the function. This is similar in the chirped function.
    """
    lam_min = 375
    lam_max = 750
    n_lam = 100
    theta = 0 * np.pi / 180
    vlam = np.linspace(lam_min, lam_max, n_lam)
    scc = np.zeros(n_lam)
    Ab = np.zeros(n_lam)
    for k in range(n_lam):
        lam = vlam[k]
        epsilon = np.array([1, 2, 3, epscSi(lam)], dtype=complex)
        mu = np.ones(epsilon.size)
        type_ = np.append(0, np.append(np.tile(np.array([1, 2]), int(X.size / 2)), 3))
        hauteur = np.append(0, np.append(X, 30000))
        pol = 0
        absorb = absorption(lam, epsilon, mu, type_, hauteur, pol, theta)
        scc[k] = solar(lam)
        Ab[k] = absorb[len(absorb) - 1]
    max_scc = np.trapz(scc, vlam)
    j_sc = np.trapz(scc * Ab, vlam)
    CE = j_sc / max_scc
    cost = 1 - CE
    return cost  # type: ignore


def cf_photosic_realistic(eps_and_d: np.ndarray) -> float:
    """eps_and_d is a vector composed in a first part with the epsilon values
    (the material used in each one of the layers), and in a second part with the
    thicknesses of each one of the layers, like in Bragg.
    Any number of layers can work. Basically I used between 4 and 50 layers,
    and the best results are generally obtained when the structure has between 10 and 20 layers.
    The epsilon values are generally comprised between 1.00 and 9.00.
    """
    dimension = int(eps_and_d.size / 2)
    eps = eps_and_d[0:dimension]
    d = eps_and_d[dimension : dimension * 2]
    epsd = np.array([eps, d])
    lam_min = 375
    lam_max = 750
    n_lam = 100
    theta = 0 * 180 / np.pi
    vlam = np.linspace(lam_min, lam_max, n_lam)
    scc = np.zeros(n_lam)
    Ab = np.zeros(n_lam)
    # spectrum=np.zeros(n_lam)
    for k in range(n_lam):
        # absorb=absorption(epsd,theta*pi/180,lam[k])
        lam = vlam[k]
        epsilon = np.append(1, np.append(epsd[0], epscSi(lam)))
        mu = np.ones(epsilon.size)
        type_ = np.arange(0, epsd[0].size + 2)
        hauteur = np.append(0, np.append(epsd[1], 30000))
        pol = 0
        absorb = absorption(lam, epsilon, mu, type_, hauteur, pol, theta)
        scc[k] = solar(lam)
        Ab[k] = absorb[len(absorb) - 1]
    max_scc = np.trapz(scc, vlam)
    j_sc = np.trapz(scc * Ab, vlam)
    CE = j_sc / max_scc
    cost = 1 - CE
    return cost  # type: ignore
