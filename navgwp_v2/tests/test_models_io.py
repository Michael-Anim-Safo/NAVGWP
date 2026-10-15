"""Model builders, the separable-diagonal (Tully) path, and file output."""
import os
import tempfile

import numpy as np

import navgwp2 as ng


# ---------------------------------------------------------------- models --
def test_model_shapes():
    m = ng.fmo(nsites=3, nmodes_per_site=5)
    assert m["N"] == 3 and m["D"] == 15
    assert m["kappa"].shape == (3, 15)
    assert m["omega2"].shape == (3, 15)
    assert m["J"].shape == (3, 3)
    assert np.allclose(np.diag(m["J"]), 0.0)
    # each site couples only to its own bath block
    for n in range(3):
        blk = m["kappa"][n].reshape(3, 5)
        assert np.all(blk[n] != 0)
        assert np.all(np.delete(blk, n, axis=0) == 0)


def test_fmo_energies_match_reference_matrix():
    m = ng.fmo(nsites=3, nmodes_per_site=2)
    assert np.allclose(m["E"] / ng.CM1, [200.0, 320.0, 0.0])
    assert np.isclose(m["J"][0, 1] / ng.CM1, -87.7)


def test_debye_bath_reorganisation_energy():
    """sum_j kappa_j^2/(2 w_j^2) must reproduce lambda."""
    lam = 0.037
    w, k = ng.debye_bath(4000, lam, 0.1)
    assert abs(np.sum(k ** 2 / (2 * w ** 2)) - lam) / lam < 1e-12


def test_separable_diagonal_matches_analytic_gaussian_integral():
    """The Gauss-Hermite diagonal path vs a case with a known answer:
    a harmonic v(x)=w^2 x^2/2 has <V> = w^2(q^2 + 1/(2a))/2."""
    w2 = 0.7
    vf = [lambda k, x: 0.5 * w2 * np.asarray(x, float) ** 2] * 2
    df = [lambda k, x: w2 * np.asarray(x, float)] * 2
    m = ng.separable(vf, df, N=2, D=1, J=[[0, 0.01], [0.01, 0]])
    q = np.array([0.37]); z = np.array([1.3 + 0j])
    V, F, d2 = ng.diagonal_terms(m, 0, q, z)
    a = z.real[0]
    assert abs(V - 0.5 * w2 * (q[0] ** 2 + 1 / (2 * a))) < 1e-10
    assert abs(F[0] - w2 * q[0]) < 1e-10
    assert abs(d2[0] - w2) < 1e-5


def test_tully_models_build_and_run():
    for build in (ng.tully1, ng.tully2):
        m = build()
        assert m["N"] == 2 and m["D"] == 1
        r = ng.run(m, [1, 0], q0=[[-8.0], [-8.0]], p0=[[20.0], [20.0]],
                   dt=0.5, t_max=400.0, record_every=50)
        assert np.all(np.isfinite(r["pops"]))
        assert abs(r["pops"].sum(axis=1) - 1).max() < 1e-12
        assert r["dNmax"] < 1e-4


def test_tully1_transfers_population():
    """A packet driven through the crossing must transfer some population."""
    m = ng.tully1()
    r = ng.run(m, [1, 0], q0=[[-8.0], [-8.0]], p0=[[25.0], [25.0]],
               dt=0.2, t_max=1200.0, record_every=100)
    assert r["pops"][-1, 1] > 0.05


def test_ferretti_model_shape_and_couples_only_off_y0():
    """The Ferretti conical-intersection model: N=2, D=2 (X, Y), and its
    coupling is ODD in Y -- it must vanish exactly along Y=0 (a true, not
    avoided, crossing) and be nonzero elsewhere."""
    m = ng.ferretti()
    assert m["N"] == 2 and m["D"] == 2
    z = ng.model_widths(m)
    # at Y=0 (q[:,1]=0) the coupling matrix element must vanish exactly
    p0 = np.zeros(2)
    q_on_axis = np.array([[3.0, 0.0], [3.0, 0.0]])
    W, M = ng.coupling_terms(m, 0, 1, q_on_axis[0], p0, z[0],
                             q_on_axis[1], p0, z[1])
    assert abs(W) < 1e-12
    # off-axis (Y != 0) it should not vanish
    q_off_axis = np.array([[3.0, 0.5], [3.0, 0.5]])
    W2, _ = ng.coupling_terms(m, 0, 1, q_off_axis[0], p0, z[0],
                              q_off_axis[1], p0, z[1])
    assert abs(W2) > 1e-6


def test_ferretti_conserves_energy_and_norm():
    m = ng.ferretti()
    r = ng.run(m, [1, 0], q0=[[4.0, 0.3], [4.0, 0.3]], p0=np.zeros((2, 2)),
               dt=2.0, t_max=400.0, record_every=20)
    assert np.all(np.isfinite(r["pops"]))
    assert abs(r["pops"].sum(axis=1) - 1).max() < 1e-10
    assert r["dNmax"] < 1e-6
    assert r["dEmax"] / abs(r["E0"]) < 1e-4


def test_tully2_builds_and_runs():
    m = ng.tully2()
    assert m["N"] == 2 and m["D"] == 1
    r = ng.run(m, [1, 0], q0=[[-8.0], [-8.0]], p0=[[20.0], [20.0]],
               dt=0.5, t_max=400.0, record_every=50)
    assert np.all(np.isfinite(r["pops"]))
    assert abs(r["pops"].sum(axis=1) - 1).max() < 1e-12
    assert r["dNmax"] < 1e-4


def test_tully3_conserves_and_transfers_population():
    """Tully III (extended coupling region): the quad_coupling path, with
    suggest_dt's automatic extra safety margin for this coupling type."""
    m = ng.tully3()
    assert m["N"] == 2 and m["D"] == 1
    dt = ng.suggest_dt(m)
    r = ng.run(m, c0=[0, 1], q0=[[-15.0], [-15.0]], p0=[[20.0], [20.0]],
               dt=dt, t_max=800.0, record_every=40)
    assert np.all(np.isfinite(r["pops"]))
    assert abs(r["pops"].sum(axis=1) - 1).max() < 1e-10
    assert r["dNmax"] < 1e-8
    assert r["dEmax"] / abs(r["E0"]) < 1e-8
    # the wavepacket hasn't reached the coupling region by t_max=800
    # (v=p/mass=0.01, distance covered=8 < 15), so it should still be
    # almost entirely on the state it started on
    assert r["pops"][-1, 1] > 0.99


def test_suggest_dt_positive():
    for m in (ng.spin_boson(0, 1, 2, 2.5, 10), ng.fmo(3, 3), ng.tully1()):
        assert ng.suggest_dt(m) > 0


def test_boltzmann_sample_statistics():
    """<p^2>/m ~ kT and <w^2 q^2> ~ kT for the sampled bath."""
    m = ng.spin_boson(0.0, 1.0, 1.0, 1.0, nmodes=4000)
    kT = 0.5
    q, p = ng.boltzmann_sample(m, kT, rng=np.random.default_rng(0))
    w2 = m["omega2"].mean(axis=0)
    assert abs(np.mean(p ** 2 / m["mass"]) - kT) / kT < 0.1
    assert abs(np.mean(w2 * q ** 2) - kT) / kT < 0.1


# -------------------------------------------------------------------- io --
def test_write_columns_creates_files_and_normalised_snapshots():
    m = ng.tully1()
    r = ng.run(m, [1, 0], q0=[[-6.0], [-6.0]], p0=[[20.0], [20.0]],
               dt=0.5, t_max=300.0, record_every=10)
    with tempfile.TemporaryDirectory() as d:
        ng.write_columns(r, outdir=d, n_snapshots=6)
        for f in ("p.txt", "q.txt", "a.txt", "coeff.txt", "wfc.txt"):
            assert os.path.exists(os.path.join(d, f)), f
        # coeff.txt rows must sum to 1 (to the %.8e precision written)
        c = np.loadtxt(os.path.join(d, "coeff.txt"))
        assert np.max(np.abs(c[:, 1:].sum(axis=1) - 1.0)) < 1e-7
        # each wfc block must integrate to the norm
        blocks = open(os.path.join(d, "wfc.txt")).read().strip().split("\n\n")
        assert len(blocks) == 6
        for b in blocks:
            rows = [l for l in b.splitlines() if not l.startswith("#")]
            arr = np.array([[float(v) for v in r_.split()] for r_ in rows])
            integral = np.trapezoid(arr[:, -1], arr[:, 0])
            assert abs(integral - 1.0) < 1e-3, integral


def test_pickle_roundtrip():
    m = ng.lvc([0, 0.05], [[0.02], [-0.02]], [1.0], [[0, .01], [.01, 0]])
    r = ng.run(m, [1, 0], dt=0.05, t_max=10.0, record_every=10)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "r.pkl")
        ng.save_pickle(r, p)
        r2 = ng.load_pickle(p)
    assert np.allclose(r["pops"], r2["pops"])


def test_reconstruct_density_is_normalised():
    m = ng.tully1()
    r = ng.run(m, [1, 0], q0=[[-5.0], [-5.0]], p0=[[15.0], [15.0]],
               dt=0.5, t_max=200.0, record_every=20)
    x = np.linspace(-25, 25, 4001)
    d = ng.density(r, -1, x)
    assert abs(np.trapezoid(d, x) - 1.0) < 1e-4
