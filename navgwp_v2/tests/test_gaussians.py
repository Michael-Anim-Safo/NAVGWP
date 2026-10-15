"""Gaussian integrals against brute-force numerical integration.

These are the strongest tests in the suite: two independent routes to the
same well-defined number (closed form vs. dense grid quadrature).
"""
import numpy as np

import navgwp2 as ng
from navgwp2.compute import wavefunction


def _grid(D, n, half=9.0):
    axes = [np.linspace(-half, half, n)] * D
    G = np.meshgrid(*axes, indexing="ij")
    X = np.stack(G, axis=-1)
    dv = np.prod([a[1] - a[0] for a in axes])
    return X, dv


def test_overlap_and_moments_vs_bruteforce():
    """S, zbar and <x^2> vs grid integration, complex unequal widths, D=1,2,3."""
    rng = np.random.default_rng(5)
    for D, n in [(1, 241), (2, 161), (3, 81)]:
        z1 = rng.uniform(.6, 1.5, D) + 1j * rng.uniform(-.5, .5, D)
        z2 = rng.uniform(.6, 1.5, D) + 1j * rng.uniform(-.5, .5, D)
        q1, p1 = rng.uniform(-1, 1, D), rng.uniform(-2, 2, D)
        q2, p2 = rng.uniform(-1, 1, D), rng.uniform(-2, 2, D)
        X, dv = _grid(D, n)
        g1 = wavefunction(z1, q1, p1, X)
        g2 = wavefunction(z2, q2, p2, X)
        S_bf = np.sum(np.conj(g1) * g2) * dv
        S, zb, var = ng.overlap_moments(z1, q1, p1, z2, q2, p2)
        assert abs(S - S_bf) < 1e-12, (D, abs(S - S_bf))
        for k in range(D):
            zb_bf = np.sum(np.conj(g1) * X[..., k] * g2) * dv / S_bf
            x2_bf = np.sum(np.conj(g1) * X[..., k] ** 2 * g2) * dv / S_bf
            assert abs(zb[k] - zb_bf) < 1e-10
            assert abs(zb[k] ** 2 + var[k] - x2_bf) < 1e-10


def test_gaussian_coupling_vs_bruteforce():
    """The Gaussian-coupling variant of the moments."""
    rng = np.random.default_rng(11)
    for D, n in [(1, 241), (2, 161)]:
        z1 = rng.uniform(.7, 1.4, D) + 1j * rng.uniform(-.3, .3, D)
        z2 = rng.uniform(.7, 1.4, D) + 1j * rng.uniform(-.3, .3, D)
        q1, p1 = rng.uniform(-1, 1, D), rng.uniform(-2, 2, D)
        q2, p2 = rng.uniform(-1, 1, D), rng.uniform(-2, 2, D)
        beta, xc = rng.uniform(.3, .9, D), rng.uniform(-.5, .5, D)
        X, dv = _grid(D, n)
        g1 = wavefunction(z1, q1, p1, X)
        g2 = wavefunction(z2, q2, p2, X)
        w = np.exp(-np.sum(beta * (X - xc) ** 2, axis=-1))
        S_bf = np.sum(np.conj(g1) * w * g2) * dv
        S, zb, _ = ng.gaussian_coupling_moments(z1, q1, p1, z2, q2, p2, beta, xc)
        assert abs(S - S_bf) < 1e-12
        for k in range(D):
            zb_bf = np.sum(np.conj(g1) * X[..., k] * w * g2) * dv / S_bf
            assert abs(zb[k] - zb_bf) < 1e-10


def test_gauss_lin_coupling_vs_bruteforce():
    """gauss_lin (linear-times-Gaussian coupling, e.g. the Ferretti model's
    H12 = gamma*Y*exp(-alpha(X-X3)^2)*exp(-beta*Y^2)) against a brute-force
    2-D grid integral of both the coupling matrix element W_ij and the force
    term M_ij used in the equations of motion."""
    rng = np.random.default_rng(13)
    D = 2
    z1 = rng.uniform(.7, 1.4, D) + 1j * rng.uniform(-.3, .3, D)
    z2 = rng.uniform(.7, 1.4, D) + 1j * rng.uniform(-.3, .3, D)
    q1, p1 = rng.uniform(-1, 1, D), rng.uniform(-2, 2, D)
    q2, p2 = rng.uniform(-1, 1, D), rng.uniform(-2, 2, D)
    beta, xc = rng.uniform(.3, .9, D), rng.uniform(-.5, .5, D)
    A, L = 0.37, 1

    m = dict(N=2, D=D, J=np.zeros((2, 2)), lam=np.zeros((2, 2, D)),
            gauss=None,
            gauss_lin=dict(A=np.array([[0.0, A], [A, 0.0]]), beta=beta,
                          xc=xc, lin_dim=L))
    W, M = ng.coupling_terms(m, 0, 1, q1, p1, z1, q2, p2, z2)

    X, dv = _grid(D, 161)
    g1 = wavefunction(z1, q1, p1, X)
    g2 = wavefunction(z2, q2, p2, X)
    Vc = A * (X[..., L] - xc[L]) * np.exp(-np.sum(beta * (X - xc) ** 2, axis=-1))
    W_bf = np.sum(np.conj(g1) * Vc * g2) * dv
    assert abs(W - W_bf) < 1e-10, (W, W_bf)
    for k in range(D):
        M_bf = np.sum(np.conj(g1) * (X[..., k] - q1[k]) * Vc * g2) * dv
        assert abs(M[k] - M_bf) < 1e-10, (k, M[k], M_bf)


def test_quad_coupling_vs_gauss_analytic_and_bruteforce():
    """quad_coupling (numerical-quadrature coupling, e.g. Tully III's
    asymmetric ECR V12 which has no closed form) cross-checked two ways for
    a Gaussian V12 that DOES have a closed form: against the already-trusted
    analytic `gauss` path, and against an independent brute-force grid."""
    rng = np.random.default_rng(21)
    D = 1
    z1 = rng.uniform(.7, 1.4, D) + 1j * rng.uniform(-.3, .3, D)
    z2 = rng.uniform(.7, 1.4, D) + 1j * rng.uniform(-.3, .3, D)
    q1, p1 = rng.uniform(-1, 1, D), rng.uniform(-2, 2, D)
    q2, p2 = rng.uniform(-1, 1, D), rng.uniform(-2, 2, D)
    A, beta, xc = 0.37, 2.1, 0.15

    m = dict(N=2, D=D, J=np.zeros((2, 2)), lam=np.zeros((2, 2, D)),
            gauss=None, gauss_lin=None,
            quad_coupling=dict(i=0, j=1, v=lambda x: A * np.exp(-beta * (x - xc) ** 2),
                              npts=4000, margin=15.0))
    W_quad, M_quad = ng.coupling_terms(m, 0, 1, q1, p1, z1, q2, p2, z2)

    S, zb, _ = ng.gaussian_coupling_moments(z1, q1, p1, z2, q2, p2,
                                            np.array([beta]), np.array([xc]))
    W_analytic, M_analytic = A * S, A * S * (zb - q1)
    assert abs(W_quad - W_analytic) < 1e-9
    assert abs(M_quad[0] - M_analytic[0]) < 1e-9

    X, dv = _grid(D, 241)
    g1 = wavefunction(z1, q1, p1, X)
    g2 = wavefunction(z2, q2, p2, X)
    Vc = A * np.exp(-beta * (X[..., 0] - xc) ** 2)
    W_bf = np.sum(np.conj(g1) * Vc * g2) * dv
    assert abs(W_quad - W_bf) < 1e-6


def test_quad_coupling_is_hermitian():
    """W_ji = conj(W_ij) for an ASYMMETRIC coupling with no closed form
    (Tully III's actual V12), not just for the Gaussian test case above."""
    m = ng.tully3()
    rng = np.random.default_rng(4)
    q = rng.normal(0, 3, (2, 1)); p = rng.normal(0, 8, (2, 1))
    z = ng.model_widths(m)
    W01, _ = ng.coupling_terms(m, 0, 1, q[0], p[0], z[0], q[1], p[1], z[1])
    W10, _ = ng.coupling_terms(m, 1, 0, q[1], p[1], z[1], q[0], p[0], z[0])
    assert abs(W01 - np.conj(W10)) < 1e-12


def test_frozen_limit_of_overlap():
    """Real equal widths must reproduce the textbook frozen overlap."""
    rng = np.random.default_rng(2)
    D = 3
    a = rng.uniform(.7, 1.4, D)
    q1, p1 = rng.uniform(-1, 1, D), rng.uniform(-2, 2, D)
    q2, p2 = rng.uniform(-1, 1, D), rng.uniform(-2, 2, D)
    S, zb, var = ng.overlap_moments(a + 0j, q1, p1, a + 0j, q2, p2)
    S_ref = np.exp(np.sum(-0.25 * a * (q1 - q2) ** 2
                          - (p2 - p1) ** 2 / (4 * a)
                          + 0.5j * (p1 + p2) * (q1 - q2)))
    zb_ref = 0.5 * (q1 + q2) + 1j * (p2 - p1) / (2 * a)
    assert abs(S - S_ref) < 1e-12
    assert np.max(np.abs(zb - zb_ref)) < 1e-12
    assert np.max(np.abs(var - 1 / (2 * a))) < 1e-12


def test_self_overlap_is_one():
    rng = np.random.default_rng(3)
    D = 4
    z = rng.uniform(.5, 2.0, D) + 1j * rng.uniform(-.4, .4, D)
    q, p = rng.uniform(-2, 2, D), rng.uniform(-3, 3, D)
    S, zb, _ = ng.overlap_moments(z, q, p, z, q, p)
    assert abs(S - 1.0) < 1e-12
    assert np.max(np.abs(zb - q)) < 1e-12       # <x> = q for the same Gaussian


def test_kinetic_expectation_vs_bruteforce():
    """<T> for a complex width, against a finite-difference Laplacian."""
    D, m = 1, 1.7
    z = np.array([1.2 + 0.6j]); q = np.array([0.4]); p = np.array([0.9])
    x = np.linspace(-25, 25, 400001)[:, None]
    g = wavefunction(z, q, p, x)
    h = 1e-4
    d2 = (wavefunction(z, q, p, x + h) - 2 * g
          + wavefunction(z, q, p, x - h)) / h ** 2
    T_bf = np.real(np.sum(np.conj(g) * (-d2 / (2 * m))) * (x[1, 0] - x[0, 0]))
    T = ng.kinetic_expectation(z, p, np.array([m]))
    assert abs(T - T_bf) < 1e-6, (T, T_bf)


def test_coupling_matrix_is_hermitian():
    """W_ji = conj(W_ij) guarantees exact norm conservation."""
    rng = np.random.default_rng(7)
    m = ng.spin_boson(0.1, 0.8, 1.0, 2.0, nmodes=6)
    N, D = m["N"], m["D"]
    q = rng.normal(0, 1, (N, D)); p = rng.normal(0, 1, (N, D))
    z = ng.model_widths(m)
    W = np.zeros((N, N), complex)
    for i in range(N):
        for j in range(N):
            if i != j:
                W[i, j], _ = ng.coupling_terms(m, i, j, q[i], p[i], z[i],
                                               q[j], p[j], z[j])
    assert np.max(np.abs(W - W.conj().T)) == 0.0
