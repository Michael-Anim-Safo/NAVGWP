"""Full (non-diagonal) width matrices -- compute_matrix.py.

  * diagonal widths reproduce compute.run (frozen and thawed)
  * off-diagonal widths: zero for uncoupled separable surfaces, sourced by coupling
  * coherent_Z is stationary on a rotated well
  * rotated harmonic well: full Z is exact, diagonal Z is not
  * the two terms legacy=True drops are required by the exact TDSE, and
    compute.run / compute.rhs carry them
"""
import numpy as np

import navgwp2 as ng
from navgwp2 import compute_matrix as cm


def _random_lvc(rng, N=3, D=3):
    J = rng.normal(0, 0.005, (N, N)); J = (J + J.T) / 2
    lam = rng.normal(0, 0.002, (N, N, D)); lam = (lam + np.swapaxes(lam, 0, 1)) / 2
    return ng.lvc([0.0, 0.01, 0.02][:N], rng.normal(0, 0.02, (N, D)),
                  rng.uniform(0.005, 0.02, (N, D)), J, lam=lam,
                  mass=rng.uniform(1, 3, D),
                  gauss=dict(A=0.004 * (1 - np.eye(N)), beta=np.full(D, .3),
                             xc=np.full(D, .5)))


def test_diagonal_reproduces_compute_run():
    rng = np.random.default_rng(0)
    m = _random_lvc(rng)
    zeta0 = rng.uniform(0.05, 0.4, (3, 3)) + 0j          # squeezed: widths breathe
    kw = dict(q0=rng.normal(0, 1, (3, 3)), p0=rng.normal(0, .1, (3, 3)),
              dt=0.2, t_max=40.0, record_every=50)
    for mode in ("frozen", "thawed"):
        a = ng.run(m, [1, 0, 0], zeta0=zeta0, width_mode=mode, **kw)
        b = cm.run(m, [1, 0, 0], Z0=np.array([np.diag(z) for z in zeta0]),
                   width_mode=mode, width_shape="diagonal", **kw)
        assert np.max(np.abs(a["pops"] - b["pops"])) < 1e-12
        assert np.max(np.abs(a["q"] - b["q"])) < 1e-12
        zb = np.diagonal(b["Z"], axis1=-2, axis2=-1)
        assert np.max(np.abs(a["zeta"] - zb)) < 1e-12


def test_offdiagonal_widths_sources():
    """Z_kl (k != l) has two sources: the off-diagonal Hessian of V_ii, and
    the inter-state coupling term of the width equation (its bracket
    R_ij = <g_i|(x-q_i)(x-q_i)^T V_ij|g_j> has off-diagonal d_k d_l parts).
    Uncoupled separable surfaces have neither, so Z stays exactly diagonal;
    switching the coupling on makes it correlate."""
    rng = np.random.default_rng(1)
    m = _random_lvc(rng)
    kw = dict(q0=rng.normal(0, 1, (3, 3)), Z0=[0.1, 0.2, 0.3], dt=0.2,
              t_max=40.0, record_every=50)
    m0 = dict(m, J=np.zeros_like(m["J"]), lam=np.zeros_like(m["lam"]), gauss=None)
    f = cm.run(m0, [1, 0, 0], width_shape="full", **kw)
    d = cm.run(m0, [1, 0, 0], width_shape="diagonal", **kw)
    assert f["offdiag"].max() == 0.0
    assert np.max(np.abs(f["pops"] - d["pops"])) == 0.0
    f = cm.run(m, [1, 0, 0], width_shape="full", **kw)
    assert f["offdiag"].max() > 1e-6


def _rotated_well(theta_deg=35.0):
    th = np.deg2rad(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    K = R @ np.diag([1.0, 0.35 ** 2]) @ R.T
    return cm.lvc_full([0.0], [[0.0, 0.0]], K, [[0.0]], mass=np.array([1.0, 2.0]))


def test_coherent_Z_is_stationary_on_rotated_well():
    m = _rotated_well()
    r = cm.run(m, [1.0], dt=0.01, t_max=10.0, record_every=100, seed=None)
    assert np.max(np.abs(r["Z"] - r["Z"][0])) < 1e-10
    assert r["offdiag"][0] > 0.1                     # genuinely non-diagonal


def test_rotated_harmonic_well_full_exact_diagonal_not():
    m = _rotated_well()
    axes = [np.linspace(-9, 9, 96, endpoint=False)] * 2
    X = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    dt, T = 0.01, 6.0
    kw = dict(q0=[[2.0, 0.5]], Z0=np.eye(2), dt=dt, t_max=T,
              record_every=int(T / dt), seed=None)
    full = cm.run(m, [1.0], width_shape="full", **kw)
    diag = cm.run(m, [1.0], width_shape="diagonal", **kw)
    psi0 = cm.wavefunction_full(full["Z"][0, 0], full["q"][0, 0],
                                full["p"][0, 0], X)[None]
    ex = cm.propagate_exact_grid(m, psi0, axes, dt, int(T / dt))

    def fid(r):
        g = r["c"][-1, 0] * cm.wavefunction_full(r["Z"][-1, 0], r["q"][-1, 0],
                                                 r["p"][-1, 0], X)
        return abs(np.sum(np.conj(ex["psi"][0]) * g) * ex["dV"]) ** 2
    assert fid(full) > 0.9999
    assert fid(diag) < 0.9


def test_breathing_phase_term_is_required():
    """Uncoupled harmonic surfaces, squeezed widths, superposition: each
    component is an exact thawed Gaussian, so the inter-state coherence is
    exact iff the phase of c carries -1/4 tr(A^-1 Bdot)."""
    m = ng.lvc([0.0, 0.003], [[0.0], [0.02]], [[0.04], [0.09]],
               np.zeros((2, 2)), mass=np.array([1.0]))
    x = np.linspace(-25, 25, 1024, endpoint=False)
    c0 = np.array([1, 1]) / np.sqrt(2)
    g0 = cm.wavefunction_full(np.array([[0.05]]), np.array([1.0]),
                              np.zeros(1), x[:, None])
    dt, T = 0.01, 60.0
    ex = cm.propagate_exact_grid(m, np.array([c0[0] * g0, c0[1] * g0]), [x],
                                 dt, int(T / dt))
    coh_ex = np.sum(np.conj(ex["psi"][0]) * ex["psi"][1]) * (x[1] - x[0])
    err = {}
    for legacy in (True, False):
        r = cm.run(m, c0, q0=[[1.0], [1.0]], Z0=0.05, dt=dt, t_max=T,
                   record_every=int(T / dt), seed=None, width_shape="diagonal",
                   legacy=legacy)
        S, _, _ = cm.pair_moments(r["Z"][-1, 0], r["q"][-1, 0], r["p"][-1, 0],
                                  r["Z"][-1, 1], r["q"][-1, 1], r["p"][-1, 1])
        err[legacy] = abs(np.conj(r["c"][-1, 0]) * r["c"][-1, 1] * S - coh_ex)
    # compute.py itself (run() keeps only |c|, so step rk4 to keep the phase)
    y = ng.pack(np.ones((2, 1)), np.zeros((2, 1)), np.full((2, 1), 0.05 + 0j), c0)
    for _ in range(int(T / dt)):
        y = ng.rk4_step(y, dt, m, width_mode="thawed")
    q, p, z, c = ng.unpack(y, 2, 1)
    S = ng.overlap_moments(z[0], q[0], p[0], z[1], q[1], p[1])[0]
    err_ng = abs(np.conj(c[0]) * c[1] * S - coh_ex)
    assert err[False] < 1e-4
    assert err[True] > 0.1
    assert err_ng < 1e-4


def test_pdot_matches_exact_ehrenfest_with_chirped_width():
    """Instantaneous d<P>_1/dt from the exact TDSE on a grid vs the model's
    pdot, for a chirped (Im Z != 0) width and linear coupling."""
    m = ng.lvc([0.0, 0.01], [[0.0], [0.03]], [[0.05], [0.08]],
               [[0, 0.01], [0.01, 0]],
               lam=np.array([[[0.0], [0.02]], [[0.02], [0.0]]]),
               mass=np.array([1.0]))
    x = np.linspace(-30, 30, 4096, endpoint=False); dx = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(len(x), dx)
    q = np.array([[0.5], [-1.0]]); p = np.array([[0.3], [-0.2]])
    Z = np.array([[[0.4 + 0.3j]], [[0.25 - 0.2j]]]); c = np.array([0.8, 0.6j])
    psi = np.array([c[i] * cm.wavefunction_full(Z[i], q[i], p[i], x[:, None])
                    for i in range(2)])
    V = cm.potential_matrix(m, x[:, None])
    dpsi = -1j * (np.einsum("xij,jx->ix", V, psi)
                  + np.fft.ifft(k ** 2 / 2 * np.fft.fft(psi, axis=1), axis=1))
    P = lambda f: np.fft.ifft(k * np.fft.fft(f))
    n1 = np.sum(abs(psi[0]) ** 2) * dx
    P1 = np.sum(np.conj(psi[0]) * P(psi[0])).real * dx
    dn1 = 2 * np.sum(np.conj(psi[0]) * dpsi[0]).real * dx
    dP1 = 2 * np.sum(np.conj(psi[0]) * P(dpsi[0])).real * dx
    exact = (dP1 - (P1 / n1) * dn1) / n1
    y = cm.pack(q, p, Z, c)
    pd = cm.unpack(cm.rhs(y, m, eps=0.0), 2, 1)[1][0, 0]
    pd_legacy = cm.unpack(cm.rhs(y, m, eps=0.0, legacy=True), 2, 1)[1][0, 0]
    zeta = np.array([[Z[0, 0, 0]], [Z[1, 0, 0]]])
    pd_ng = ng.unpack(ng.rhs(ng.pack(q, p, zeta, c), m, eps=0.0,
                             width_mode="thawed"), 2, 1)[1][0, 0]
    assert abs(pd - exact) < 1e-10
    assert abs(pd_ng - exact) < 1e-10
    assert abs(pd_legacy - exact) > 1e-3
