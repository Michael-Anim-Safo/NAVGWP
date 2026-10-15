"""Exactly solvable limits.

These check the DYNAMICS against closed-form answers:
  * bare electronic limit  -> exp(-iHt)
  * free particle width    -> zeta0/(1 + i zeta0 t/m)
  * coherent width         -> stationary
  * squeezed width         -> breathing at 2 omega/sqrt(m), closed form
  * thawed at coherent width == frozen, bitwise
"""
import numpy as np

import navgwp2 as ng


def test_bare_electronic_limit_matches_matrix_exponential():
    """No nuclear dependence => must reduce to i cdot = H c exactly."""
    rng = np.random.default_rng(3)
    for N in (2, 3, 5):
        H = rng.normal(0, 0.02, (N, N)); H = (H + H.T) / 2
        E = np.diag(H).copy(); J = H.copy(); np.fill_diagonal(J, 0)
        D = 4
        m = ng.lvc(E, np.zeros((N, D)), np.zeros(D), J)
        c0 = np.zeros(N); c0[0] = 1.0
        r = ng.run(m, c0, dt=0.5, t_max=300.0, record_every=10, seed=None)
        ex = ng.exact_level_populations(H, c0, r["t"])
        assert np.max(np.abs(r["pops"] - ex)) < 1e-6
        assert r["dNmax"] < 1e-8


def test_free_particle_width_law():
    """Thawed width on a flat surface follows zeta0/(1+i zeta0 t/m)."""
    D = 1
    mass = np.array([1.7])
    m = ng.lvc([0.0, 10.0], np.zeros((2, D)), np.zeros(D),
               [[0, 0], [0, 0]], mass=mass)
    z0 = 0.8 + 0j
    r = ng.run(m, [1, 0], zeta0=z0, dt=0.01, t_max=5.0, record_every=10,
               width_mode="thawed")
    ex = ng.free_width_exact(z0, mass[0], r["t"])
    assert np.max(np.abs(r["zeta"][:, 0, 0] - ex)) < 1e-8
    # and the packet really does spread
    assert r["sigma"][-1, 0, 0] > 2.0 * r["sigma"][0, 0, 0]


def test_coherent_width_is_stationary():
    """zeta = sqrt(m) omega is the fixed point of the width equation."""
    D = 1
    mass = np.array([2.0]); w = np.array([1.3])
    m = ng.lvc([0.0, 10.0], np.zeros((2, D)), w ** 2, [[0, 0], [0, 0]],
               mass=mass)
    zc = complex(np.sqrt(mass[0]) * w[0])
    r = ng.run(m, [1, 0], zeta0=zc, dt=0.01, t_max=20.0, record_every=20,
               width_mode="thawed")
    assert np.max(np.abs(r["zeta"][:, 0, 0] - zc)) < 1e-11


def test_breathing_matches_closed_form():
    """A squeezed width follows the analytic Riccati solution and breathes
    at 2 omega / sqrt(m)."""
    D = 1
    mass = np.array([1.0]); w = np.array([1.0])
    m = ng.lvc([0.0, 10.0], np.zeros((2, D)), w ** 2, [[0, 0], [0, 0]],
               mass=mass)
    z0 = 2.5 + 0j
    r = ng.run(m, [1, 0], zeta0=z0, dt=0.005, t_max=20.0, record_every=10,
               width_mode="thawed")
    ex = ng.breathing_width_exact(z0, mass[0], w[0], r["t"])
    assert np.max(np.abs(r["zeta"][:, 0, 0] - ex)) < 1e-6
    # measured breathing frequency
    s = r["sigma"][:, 0, 0]; t = r["t"]
    pk = [k for k in range(1, len(s) - 1) if s[k] > s[k - 1] and s[k] > s[k + 1]]
    meas = 2 * np.pi / np.mean(np.diff(t[pk]))
    assert abs(meas - 2 * w[0] / np.sqrt(mass[0])) / (2 * w[0]) < 5e-3


def test_thawed_equals_frozen_at_coherent_width():
    """Without inter-state coupling the coherent width is an exact fixed
    point of the width equation, so thawed and frozen agree bitwise.

    With coupling they do NOT: the TDVP width equation contains the
    coupling term -(2i a/|c|^2) sum c^* c [W - 2a R], which makes the width
    respond to population transfer.  That case is checked below."""
    D = 1
    mass = np.array([1.0]); w = np.array([1.0])
    m = ng.lvc([0.0, 0.05], [[0.02], [-0.02]], w ** 2,
               [[0, 0.0], [0.0, 0]], mass=mass)
    zc = ng.model_widths(m)
    kw = dict(zeta0=zc, dt=0.01, t_max=30.0, record_every=50)
    a = ng.run(m, [1, 0], width_mode="thawed", **kw)
    b = ng.run(m, [1, 0], width_mode="frozen", **kw)
    assert np.max(np.abs(a["pops"] - b["pops"])) == 0.0
    assert np.max(np.abs(a["zeta"] - b["zeta"])) == 0.0

    # coupling on: the width now breathes and thawed departs from frozen
    mc = ng.lvc([0.0, 0.05], [[0.02], [-0.02]], w ** 2,
                [[0, 0.01], [0.01, 0]], mass=mass)
    a = ng.run(mc, [1, 0], width_mode="thawed", **kw)
    b = ng.run(mc, [1, 0], width_mode="frozen", **kw)
    assert np.max(np.abs(a["zeta"] - zc)) > 1e-6
    assert np.max(np.abs(a["pops"] - b["pops"])) > 0.0


def test_free_particle_centre_moves_classically():
    """With no potential the centre must satisfy q(t) = q0 + p t/m."""
    D = 1
    mass = np.array([3.0])
    m = ng.lvc([0.0, 10.0], np.zeros((2, D)), np.zeros(D),
               [[0, 0], [0, 0]], mass=mass)
    r = ng.run(m, [1, 0], q0=[[0.0], [0.0]], p0=[[1.5], [1.5]],
               zeta0=1.0 + 0j, dt=0.01, t_max=4.0, record_every=10)
    expected = 1.5 * r["t"] / mass[0]
    assert np.max(np.abs(r["q"][:, 0, 0] - expected)) < 1e-9
