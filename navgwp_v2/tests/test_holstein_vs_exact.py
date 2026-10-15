"""Holstein model (reference Fortran parameters) vs numerically exact
split-operator grid propagation.

Reference parameters (model 3):  am=1, ak1=ak2=10, aka=1, qs1=0, qs2=1,
qc=2, v0=15, beta=sqrt(ak1*am)/2, i.e.

    V_11 = 5 x^2 ,   V_22 = 5 (x-1)^2 + 15 ,
    V_12 = 1.0 exp[-1.5811 (x-2)^2]

This is the only test file that checks the ACCURACY OF THE ANSATZ against
true quantum dynamics; the tolerances encode the method's real accuracy.
"""
import numpy as np

import navgwp2 as ng

K1 = K2 = 10.0
X0, Y0, C3, MASS = 1.0, 15.0, 2.0, 1.0
C2 = 0.5 * np.sqrt(K1 * MASS)
OMEGA = np.sqrt(K1 / MASS)
ZETA0 = np.sqrt(MASS) * OMEGA


def _exact(c1, q0, t_max, dt=0.001, n=2048):
    x, k, _ = ng.grid(-8.0, 8.0, n)
    v1 = 0.5 * K1 * x ** 2
    v2 = 0.5 * K2 * (x - X0) ** 2 + Y0
    vc = c1 * np.exp(-C2 * (x - C3) ** 2)
    psi1 = ng.gaussian_on_grid(x, ZETA0, q0, 0.0)
    return ng.propagate_exact(psi1, np.zeros_like(psi1), x, k, MASS,
                              v1, v2, vc, dt, int(round(t_max / dt)),
                              record_every=100)


def _vgwp(c1, q0, t_max, dt=0.002, q2=None):
    m = ng.holstein_ref(k1=K1, k2=K2, x0=X0, y0=Y0, c1=c1, c2=C2, c3=C3,
                        mass=MASS)
    q2 = q0 if q2 is None else q2
    return ng.run(m, c0=[1, 0], q0=[[q0], [q2]], p0=[[0.0], [0.0]],
                  zeta0=ZETA0 + 0j, dt=dt, t_max=t_max, record_every=20)


def _compare(c1, q0, t_max):
    ex = _exact(c1, q0, t_max)
    r = _vgwp(c1, q0, t_max)
    P = np.array([np.interp(ex["t"], r["t"], r["pops"][:, i])
                  for i in (0, 1)]).T
    return ex, r, P, np.max(np.abs(P - ex["pops"]))


# ------------------------------------------------------- model structure --
def test_reference_parameters_reproduce_intended_potentials():
    m = ng.holstein_ref()
    x = np.array([-2.0, 0.0, 1.0, 2.0])
    for i, want in enumerate([0.5 * K1 * x ** 2,
                              0.5 * K2 * (x - X0) ** 2 + Y0]):
        got = m["E"][i] + m["kappa"][i, 0] * x + 0.5 * m["omega2"][i, 0] * x ** 2
        assert np.allclose(got, want), (i, got, want)


def test_crossing_coincides_with_coupling_centre():
    """The reference parameters put the diabatic crossing exactly at qc."""
    m = ng.holstein_ref()
    f = lambda x: (m["E"][0] + m["kappa"][0, 0] * x + 0.5 * m["omega2"][0, 0] * x ** 2) \
        - (m["E"][1] + m["kappa"][1, 0] * x + 0.5 * m["omega2"][1, 0] * x ** 2)
    # linear in x since the curvatures are equal -> single root
    root = -(m["E"][0] - m["E"][1]) / (m["kappa"][0, 0] - m["kappa"][1, 0])
    assert abs(root - C3) < 1e-12
    assert abs(f(root)) < 1e-10
    assert abs(m["gauss"]["beta"][0] - 0.5 * np.sqrt(K1 * MASS)) < 1e-12


def test_release_point_is_the_classical_turning_point():
    """V_11(-2) equals the crossing energy V_11(2)."""
    assert abs(0.5 * K1 * (-2.0) ** 2 - 0.5 * K1 * C3 ** 2) < 1e-12


# ---------------------------------------------------------- the reference --
def test_exact_propagator_conserves_norm():
    ex = _exact(1.0, -2.0, 4.0)
    assert np.max(np.abs(ex["norm"] - 1.0)) < 1e-9
    assert np.max(np.abs(ex["pops"].sum(axis=1) - 1.0)) < 1e-12


# ------------------------------------------------------------- accuracy ---
def test_reference_case_transferred_population():
    """At the reference parameters the final transferred population must
    agree with the exact result to a few per cent."""
    ex, r, P, err = _compare(1.0, -2.0, 10.0)
    assert abs(P[-1, 1] - ex["pops"][-1, 1]) < 0.05, (P[-1, 1], ex["pops"][-1, 1])
    assert err < 0.12, err


def test_weak_coupling_is_accurate():
    _, _, _, err = _compare(0.25, -2.0, 6.0)
    assert err < 0.03, err


def test_error_grows_with_coupling():
    _, _, _, e_small = _compare(0.25, -2.0, 6.0)
    _, _, _, e_large = _compare(2.0, -2.0, 6.0)
    assert e_small < e_large
    assert e_small < 0.03 and e_large > 0.05


def test_nuclear_motion_matches_exact():
    """q1 must follow the exact <x> through several oscillations, and turn
    around at the crossing x = 2."""
    ex, r, _, _ = _compare(1.0, -2.0, 6.0)
    q1 = np.interp(ex["t"], r["t"], r["q"][:, 0, 0])
    assert np.max(np.abs(q1 - ex["xmean"][:, 0])) < 0.25
    assert 1.8 < np.max(q1) < 2.3


def test_holstein_ref_conserves():
    r = _vgwp(1.0, -2.0, 10.0)
    assert r["dNmax"] < 1e-6
    assert r["dEmax"] < 1e-6


def test_thawed_close_to_frozen_here():
    """Both surfaces have curvature k1 = k2 and we start at the coherent
    width zeta = sqrt(m k1).  Without coupling that width would be a fixed
    point; the TDVP coupling term in the width equation makes it breathe
    once population transfers, so thawed departs from frozen -- but only
    slightly here, and it stays as accurate against the exact result
    (the error at this coupling is set by the one-Gaussian-per-state ansatz,
    not by the width)."""
    m = ng.holstein_ref()
    kw = dict(c0=[1, 0], q0=[[-2.0], [0.0]], p0=[[0.0], [0.0]],
              zeta0=ZETA0 + 0j, dt=0.002, t_max=6.0, record_every=20)
    a = ng.run(m, width_mode="frozen", **kw)
    b = ng.run(m, width_mode="thawed", **kw)
    assert np.max(np.abs(b["zeta"] - ZETA0)) > 1e-6
    assert np.max(np.abs(a["pops"] - b["pops"])) < 0.03
    ex = _exact(1.0, -2.0, 6.0)
    P = np.interp(ex["t"], b["t"], b["pops"][:, 1])
    assert np.max(np.abs(P - ex["pops"][:, 1])) < 0.05
    assert b["dEmax"] < 1e-6
