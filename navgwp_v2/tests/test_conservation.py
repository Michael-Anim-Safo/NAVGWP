"""Conservation laws, scaling with D, and the c -> 0 singularity handling."""
import numpy as np

import navgwp2 as ng


def test_norm_and_energy_conserved_spin_boson():
    for nm in (5, 50, 200):
        m = ng.spin_boson(0.0, 1.0, 2.0, 2.5, nmodes=nm)
        q0, p0 = ng.boltzmann_sample(m, 1.0, rng=np.random.default_rng(7))
        r = ng.run(m, [1, 0], q0=q0, p0=p0, t_max=3.0, record_every=25)
        assert r["dNmax"] < 1e-6, (nm, r["dNmax"])
        assert r["dEmax"] < 1e-2, (nm, r["dEmax"])


def test_conservation_independent_of_D():
    """Norm conservation must not degrade as D grows (the gauge fix)."""
    drifts = []
    for nm in (10, 100, 400):
        m = ng.spin_boson(0.0, 1.0, 2.0, 2.5, nmodes=nm)
        q0, p0 = ng.boltzmann_sample(m, 1.0, rng=np.random.default_rng(1))
        r = ng.run(m, [1, 0], q0=q0, p0=p0, t_max=2.0, record_every=50)
        drifts.append(r["dNmax"])
    assert max(drifts) < 1e-6, drifts


def test_seeding_fixes_empty_state_singularity():
    """Starting a state exactly empty is stiff; seeding must cure it."""
    m = ng.lvc([0.0, 0.1], [[0.02, 0.01], [-0.02, -0.01]],
               [0.5 ** 2, 0.9 ** 2], [[0, 0.005], [0.005, 0]])
    kw = dict(dt=ng.suggest_dt(m), t_max=400.0, record_every=100)
    bad = ng.run(m, [1, 0], seed=None, **kw)
    good = ng.run(m, [1, 0], seed=1e-2, **kw)
    assert good["dNmax"] < 1e-8
    assert good["dNmax"] < bad["dNmax"]          # seeding strictly helps


def test_thawed_conserves_too():
    m = ng.lvc([0.0, 0.05], [[0.02], [-0.02]], [1.0],
               [[0, 0.01], [0.01, 0]])
    r = ng.run(m, [1, 0], dt=0.01, t_max=40.0, record_every=50,
               width_mode="thawed")
    assert r["dNmax"] < 1e-8
    assert r["dEmax"] < 1e-6


def test_populations_sum_to_one():
    m = ng.fmo(nsites=3, nmodes_per_site=4)
    r = ng.run(m, [1, 0, 0], t_max=2000.0, record_every=100)
    s = r["pops"].sum(axis=1)
    assert np.max(np.abs(s - 1.0)) < 1e-12


def test_gauge_shift_does_not_change_observables():
    """Adding a constant to every diagonal energy is a pure global phase."""
    E = [0.0, 0.05]
    kw = dict(kappa=[[0.02], [-0.02]], omega2=[1.0], J=[[0, 0.01], [0.01, 0]])
    m1 = ng.lvc(E, **kw)
    m2 = ng.lvc([e + 3.7 for e in E], **kw)      # shifted by a constant
    r1 = ng.run(m1, [1, 0], dt=0.01, t_max=30.0, record_every=50)
    r2 = ng.run(m2, [1, 0], dt=0.01, t_max=30.0, record_every=50)
    assert np.max(np.abs(r1["pops"] - r2["pops"])) < 1e-10
