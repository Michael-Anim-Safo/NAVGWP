"""model.py -- case definitions, run helpers and text output for the
na-VGWP example notebook (navgwp2_examples.ipynb).

Nothing here re-implements any physics.  The models live in
navgwp2/initialize.py and the dynamics in navgwp2/compute.py -- the one
engine, in which every Gaussian carries a full D x D complex symmetric width
matrix.  This module only holds

    * the numeric builder ids and build_model(), which dispatch a case to the
      matching initialize.* builder;
    * model_params(), dyn_params(), plot_params(): the per-case parameter
      dicts the notebook used to define inline;
    * exact_steps(), the step-count cap for the split-operator reference;
    * write_table() / write_observables() / read_observables(), plain-text
      output of every recorded observable, and reading it back;
    * density_2d() and plot_density_2d_full(), a 2-D density plot that keeps
      the off-diagonal of the width matrix, so a tilted packet is drawn
      tilted.

Import it as `mdl`, not as `model`: the notebook uses the name `model` for
the built model dict.

    import model as mdl
    model_params = mdl.model_params(case)
    model        = mdl.build_model(model_params)
    res          = mdl.run_case(case, model, mdl.dyn_params(case, model_params))

Every case runs on the same engine with width_shape="full"; compute.py
supports separable diagonals and quad_coupling as well as the quadratic
kinds, so Tully I/II/III are not excluded.  For the D=1 cases that matrix is
1x1 and identical to a per-mode width; only case 5 (Ferretti, D=2) tilts.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from navgwp2 import initialize, compute, save

__all__ = [
    "BUILDER_TULLY1", "BUILDER_FMO", "BUILDER_LVC", "BUILDER_HOLSTEIN_REF",
    "BUILDER_FERRETTI", "BUILDER_TULLY2", "BUILDER_TULLY3", "BUILDER_CORONADO",
    "BUILDER_VILLASECO", "BUILDER_CORONADO_HARM",
    "CASES", "MAX_EXACT_STEPS",
    "build_model", "model_params", "dyn_params", "plot_params",
    "outdir_for", "exact_steps", "run_case",
    "write_table", "write_observables", "read_observables",
    "density_2d", "density_axis_range", "plot_density_2d_full",
]


# model ids dispatched by build_model() -- mirrors a compute_model(q,
# model_params, full_id) style selector, just routing to a whole-PES builder
# in initialize.py instead of a per-point potential function.
BUILDER_TULLY1 = 0
BUILDER_FMO = 1
BUILDER_LVC = 2
BUILDER_HOLSTEIN_REF = 3
BUILDER_FERRETTI = 4
BUILDER_TULLY2 = 5
BUILDER_TULLY3 = 6
BUILDER_CORONADO = 7
BUILDER_VILLASECO = 8
BUILDER_CORONADO_HARM = 9

CASES = (0, 1, 2, 21, 3, 4, 5, 6, 7, 8, 9, 10)

MAX_EXACT_STEPS = 200000    # safety cap on the split-operator reference (cases 3, 4)


# ==========================================================================
#  Models
# ==========================================================================
def build_model(params):
    """Build the model dict for one case's `params` (from model_params()).

    Dispatches on params["model"], one of the BUILDER_* ids, to the matching
    navgwp2.initialize builder, passing params["kwargs"] straight through.

    If params carries a "D", it is checked against the D the builder actually
    produced (kappa.shape[1] inside lvc(), etc).  A mismatch means some kwarg
    -- an extra column in kappa or omega2, say -- does not match what the case
    intended, and raises ValueError rather than propagating silently.
    """
    which = params["model"]
    if which == BUILDER_TULLY1:
        res = initialize.tully1(**params["kwargs"])
    elif which == BUILDER_FMO:
        res = initialize.fmo(**params["kwargs"])
    elif which == BUILDER_LVC:
        res = initialize.lvc(**params["kwargs"])
    elif which == BUILDER_HOLSTEIN_REF:
        res = initialize.holstein_ref(**params["kwargs"])
    elif which == BUILDER_FERRETTI:
        res = initialize.ferretti(**params["kwargs"])
    elif which == BUILDER_TULLY2:
        res = initialize.tully2(**params["kwargs"])
    elif which == BUILDER_TULLY3:
        res = initialize.tully3(**params["kwargs"])
    elif which == BUILDER_CORONADO:
        res = initialize.coronado(**params["kwargs"])
    elif which == BUILDER_VILLASECO:
        res = initialize.villaseco(**params["kwargs"])
    elif which == BUILDER_CORONADO_HARM:
        res = initialize.coronado_harmonic(**params["kwargs"])
    else:
        raise ValueError("Unrecognized model id: %r" % (which,))

    if "D" in params and res["D"] != params["D"]:
        raise ValueError("D mismatch: model_params declared D=%d but the "
                         "built model has D=%d" % (params["D"], res["D"]))
    return res


def outdir_for(case):
    """Directory name this case writes its text output into."""
    names = {0: "tully1_output", 1: "fmo_output", 2: "lvcA_output",
             21: "lvcB_output", 3: "holstein_output", 4: "holstein_scan_output",
             5: "ferretti_output", 6: "tully2_output", 7: "tully3_output",
             8: "coronado_output", 9: "villaseco_output", 10: "coronado_harm_output"}
    return names[case]


def model_params(case):
    """The model parameters for one case: dict(model=<id>, D=<modes>, kwargs=...).

    `D` is the number of nuclear modes the case intends, checked by
    build_model().  Note that the Tully builders take their own `D` inside
    `kwargs` -- that one is the Gaussian coupling width, unrelated to the
    number of degrees of freedom.
    """
    if case == 0:
        return dict(model=BUILDER_TULLY1, D=1,
                    kwargs=dict(A=0.01, B=1.147, C=0.005, D=1.0))

    if case == 1:
        nsites, nmodes_per_site = 3, 20
        return dict(model=BUILDER_FMO, D=nsites * nmodes_per_site,
                    kwargs=dict(nsites=nsites, nmodes_per_site=nmodes_per_site))

    if case == 2:
        return dict(model=BUILDER_LVC, D=1,
                    kwargs=dict(E=[0.050, 0.0455], kappa=[[0.0], [-0.0006]],
                                omega2=[0.010 ** 2],
                                J=[[0, 0.0008], [0.0008, 0]],
                                mass=[1.0], name="A: bound -> bound"))

    if case == 21:
        return dict(model=BUILDER_LVC, D=1,
                    kwargs=dict(E=[0.050, 0.050], kappa=[[0.0], [-0.0007]],
                                omega2=[[0.010 ** 2], [0.0]],
                                J=[[0, 0.0008], [0.0008, 0]],
                                mass=[1.0], name="B: bound -> dissociative"))

    if case == 3:
        k1 = k2 = 10.0
        x0, y0, c1, c3, mass = 1.0, 15.0, 1.0, 2.0, 1.0
        c2 = 0.5 * np.sqrt(k1 * mass)          # beta = sqrt(ak1*am)/2
        return dict(model=BUILDER_HOLSTEIN_REF, D=1,
                    kwargs=dict(k1=k1, k2=k2, x0=x0, y0=y0, c1=c1, c2=c2,
                                c3=c3, mass=mass))

    if case == 4:
        k1 = k2 = 10.0
        x0, y0, c3, mass = 1.0, 15.0, 2.0, 1.0
        c2 = 0.5 * np.sqrt(k1 * mass)
        return dict(model=BUILDER_HOLSTEIN_REF, D=1,
                    kwargs=dict(k1=k1, k2=k2, x0=x0, y0=y0, c2=c2, c3=c3,
                                mass=mass),
                    c1_values=[0.25, 1.0, 2.0])

    if case == 5:
        # Ferretti, Granucci, Lami, Persico & Villani,
        # J. Chem. Phys. 104, 5517 (1996):
        #   H11(X,Y) = (Kx/2)(X-X1)^2 + (Ky/2)Y^2
        #   H22(X,Y) = (Kx/2)(X-X2)^2 + (Ky/2)Y^2 + Delta
        #   H12(X,Y) = gamma*Y*exp[-alpha(X-X3)^2]*exp[-beta*Y^2]
        # D=2: the symmetric-stretch X and antisymmetric-stretch Y coordinates
        # (see initialize.ferretti's docstring for why a true conical
        # intersection needs both).
        return dict(model=BUILDER_FERRETTI, D=2,
                    kwargs=dict(Kx=0.02, Ky=0.10, Delta=0.01,
                                X1=4.0, X2=3.0, X3=3.0,
                                gamma=0.01, alpha=3.0, beta=1.5,
                                Mx=20000.0, My=6667.0))

    if case == 6:
        return dict(model=BUILDER_TULLY2, D=1,
                    kwargs=dict(A=0.10, B=0.28, C=0.015, D=0.06))

    if case == 7:
        # Tully model C -- extended coupling with reflection (ECR),
        # Tully, J. Chem. Phys. 93, 1061 (1990):
        #   V11 = A, V22 = -A
        #   V12 = B*exp(C*x)          x < 0
        #   V12 = B*(2 - exp(-C*x))   x >= 0
        # V12 does NOT vanish as x -> +inf (it saturates to 2B) -- not a
        # Gaussian or Gaussian*linear shape, so this is the one case that
        # needs quad_coupling (numerical quadrature) rather than a closed-form
        # coupling.
        return dict(model=BUILDER_TULLY3, D=1,
                    kwargs=dict(A=0.0006, B=0.10, C=0.90, npts=150,
                                margin=15.0))

    if case == 10:
        # A simplified stand-in for the Coronado Morse model (case 8): states
        # 1 and 2 are harmonic fits of the first two Morse wells and state 3
        # is a Morse made less steep (b3_scale).  fit='lsq' fits each parabola
        # over the window the packet visits, which keeps both diabatic
        # crossings; fit='minimum' (curvature at the Morse minimum) is more
        # faithful at the bottom of each well but loses the 1-2 crossing
        # entirely.  b3_scale=1.0 leaves state 3 exactly as in case 8.
        return dict(model=BUILDER_CORONADO_HARM, D=1,
                    kwargs=dict(variant=1, b3_scale=0.5, fit="taylor",
                                mass=20000.0, omega_ground=5e-3))

    if case == 9:
        # Pieroni & Agostini, J. Chem. Theory Comput. 17, 5969 (2021), Table 1,
        # Models I-V: two shifted harmonic diabats of equal curvature k with a
        # Gaussian coupling.  `variant` selects the model (1..5); variant 1 is
        # also the model of Villaseco Arribas, Maitra & Agostini, J. Chem.
        # Phys. 160, 054102 (2024).  The packet starts in ONE electronic state
        # at that variant's R0 (initialize.villaseco_init), not in a coherent
        # superposition.
        return dict(model=BUILDER_VILLASECO, D=1,
                    kwargs=dict(variant=2, k=0.02, a=3.0, R1=6.0, R2=2.0,
                                mass=20000.0))

    if case == 8:
        # Coronado, Xing & Miller, Chem. Phys. Lett. 349, 521 (2001):
        # three Morse diabats, Gaussian couplings at the crossing points
        # (Tables 1-3).  variant = 1, 2, 3 selects model I, II, III.
        return dict(model=BUILDER_CORONADO, D=1,
                    kwargs=dict(variant=1, mass=20000.0, omega_ground=5e-3))

    raise ValueError("Unsupported case: %r" % (case,))


def dyn_params(case, mp):
    """Keyword arguments for compute.run, for one case.

    `mp` is that case's model_params(), needed by the cases that derive a time
    step or an initial width from the model's own parameters.

    Time grid: "dt" is the time step of the dynamics, the run ends at "t_max"
    (= dt * number of steps), and a frame is saved every "dt_out" (a whole
    multiple of dt; None -> every "record_every" steps).

    Every case carries "Z0" and "width_shape"; Z0 accepts None (each state's
    coherent width matrix), a scalar, a (D,) diagonal, or a full
    (D,D)/(N,D,D).  Cases 1 and 4 also carry keys the propagator does not take
    ("ntraj", "T_K", and case 4's "q0"); the notebook pops those out before
    calling.
    """
    dyn = {"c0": None, "q0": None, "p0": None, "Z0": None,
           "dt": None, "t_max": None, "record_every": 1, "dt_out": None,
           "width_mode": None, "width_shape": "full",
           "eps": 1e-10, "seed": 1e-2}

    if case == 0:
        # Tully I: QTAG Table 1 (SAC/DAC) packet, x0=-5, p0=10, alpha0=2,
        # s0=0.  QTAG writes psi0 ~ exp[-alpha0 (x-x0)^2 + i p0 (x-x0)];
        # here g ~ exp[-1/2 Z (x-q)^2 + ...], so Z0 = 2 alpha0 = 4.
        dyn.update({
            "c0": [1, 0], "q0": [[-5.0], [-5.0]], "p0": [[10.0], [10.0]],
            "Z0": 2.0,
            "dt": 0.01, "t_max": 2000.0, "record_every": 80,
            "width_mode": "frozen", "seed": 0.05, "eps": 1e-10,
        })

    elif case == 1:
        # FMO: q0/p0 are Boltzmann-sampled PER TRAJECTORY in the run step
        # (initialize.boltzmann_sample), and dt is computed there too via
        # initialize.suggest_dt(model) -- all three stay None here on purpose.
        # nsites/nmodes_per_site live in model_params()["kwargs"] (they define
        # the MODEL's D, not the dynamics).
        # Z0=None gives each state its coherent width matrix -- diagonal here,
        # since FMO's Hessians are.  NOTE width_mode="frozen": a frozen width
        # never evolves, so Z stays at Z0 and cannot tilt.  At D=60 the full
        # matrix costs about 9x the per-mode engine for a bit-identical
        # answer; the run cell prints max|Z_kl|, which stays at exactly zero.
        dyn.update({
            "c0": [1, 0, 0], "q0": None, "p0": None, "Z0": None, "dt": None,
            "t_max": 1000.0 / initialize.FS, "record_every": 40,
            # energy + norm at EVERY step (energy.txt), not just every 40th
            "energy_every": 1,
            "ntraj": 8, "T_K": 77.0,
            "width_mode": "thawed", "width_shape": "full",
            "seed": 1e-2, "eps": 1e-10,
        })

    elif case in (2, 21):
        # width_mode is set for completeness but is moot here -- the run step
        # ignores it and explicitly runs BOTH "frozen" and "thawed".
        # D=1, so the "full" width matrix is 1x1: no off-diagonal to fill, and
        # mathematically identical to a per-mode width.
        dyn.update({
            "c0": [1, 0],
            "Z0": np.full((2, 1, 1), np.sqrt(1.0) * 0.010, dtype=complex),
            "p0": np.zeros((2, 1)),
            "width_mode": "frozen", "width_shape": "full",
            "seed": 1e-2, "eps": 1e-10,
        })
        if case == 2:
            dyn.update({"q0": np.array([[-40.0], [-40.0]]),
                        "dt": 1.0, "t_max": 3000.0, "record_every": 20})
        else:
            dyn.update({"q0": np.zeros((2, 1)),
                        "dt": 0.2, "t_max": 1200.0, "record_every": 100})

    elif case in (3, 4):
        k1 = mp["kwargs"]["k1"]
        mass = mp["kwargs"]["mass"]
        omega = np.sqrt(k1 / mass)
        zeta0 = np.sqrt(mass) * omega       # coherent width at the reference minimum
        # D=1 again.  A scalar Z0 is broadcast to Z0 * I for every state.
        dyn.update({
            "c0": [1, 0], "p0": [[0.0], [0.0]], "Z0": zeta0 + 0j,
            "width_mode": "thawed", "width_shape": "full",
            "seed": 1e-2, "eps": 1e-10,
        })
        if case == 3:
            # the exact-grid reference in the run step reads t_max from here,
            # so the two windows always match.
            dyn.update({"q0": [[-2.0], [-2.0]], "dt": 0.001, "t_max": 8.0,
                        "record_every": 20})
        else:
            # shared starting point for the c1-scan; q0 is popped out and
            # reused per c1 in the run step, not passed to the propagator.
            dyn.update({"q0": -2.0, "dt": 0.002, "t_max": 8.0,
                        "record_every": 20})

    elif case == 5:
        # init_params_2D = {"init_placement":0, "wfc_q0":[2.0,0.0],
        #                   "wfc_p0":[0.0,0.0], "wfc_a0":[22.22,12.88],
        #                   "wfc_s0":[0.0,0.0]}
        # a Gaussian on diabat 0, placed at X0=2.0 (displaced from state 0's
        # own minimum X1=4, on the far side of the coupling centre X3=3).
        # Z0: derived from the paper's OWN wavefunction formula (eq. 13),
        # chi(X,Y;0) = (pi*dX*dY)^-1/2 exp[-(X-X0)^2/(2 dX^2)] exp[-Y^2/(2 dY^2)],
        # with dX=0.150, dY=0.197 bohr -- matching our exp[-zeta/2 (x-q)^2]
        # gives zeta = 1/dX^2, 1/dY^2 = [44.444, 25.767] (exactly 2x wfc_a0,
        # confirming Libra's "a" is defined via exp[-a(x-q)^2], i.e. a = zeta/2).
        # PX0=5.0: nonzero momentum along X (the reference IC was at rest);
        # this drives the packet through the coupling centre X3=3 rather than
        # relying on the restoring force from X0=2 back to X1=4.
        #
        # This is the ONE case where the full width matrix changes an answer.
        # Off-diagonal Z has two sources: a non-diagonal Hessian of a state's
        # own surface, and the coupling bracket
        # <g_i|(x-q_i)(x-q_i)^T V_ij|g_j>.  Ferretti's diabats are
        # (Kx/2)(X-X1)^2 + (Ky/2)Y^2, Hessian exactly diagonal, so the first is
        # zero -- but H12 mixes X and Y, so the second is not.  Measured:
        #   state 0 (the populated one): max |Z_01| = 2.79 at t=1154, against
        #     diagonals 17.9 / 40.8 -- a real but modest tilt;
        #   state 1: max |Z_01| = 18.6 at t=279, but it holds almost no
        #     population then, and a near-empty state's width is weakly
        #     constrained (the 1/|c_i|^2 factor, regularised by eps and seed).
        # Transferred population at t_max: 0.1194 (diagonal) -> 0.1153 (full).
        # Set width_shape="diagonal" to recover the per-mode ansatz exactly.
        dyn.update({
            "c0": [1, 0], "q0": [[2.0, 0.0], [2.629151053, -0.3405737567]],
            "p0": [[-5.0, -5.0], [-5.0, -5.0]],
            "Z0": np.array([22.22, 12.88], dtype=complex),
            "dt": 0.25, "t_max": 2000.0, "record_every": 5,
            "width_mode": "thawed", "width_shape": "full",
            "seed": 0.0001, "eps": 1e-10,
        })

    elif case == 6:
        # Tully II: incoming packet from the left; k=20 lands in the strongest,
        # most oscillatory part of the DAC transfer-probability curve.
        dyn.update({
            "c0": [1, 0], "q0": [[-15.0], [-15.0]], "p0": [[20.0], [20.0]],
            "Z0": None,
            "dt": 0.2, "t_max": 1500.0, "record_every": 20,
            "width_mode": "frozen", "seed": 1e-1, "eps": 1e-10,
        })

    elif case == 7:
        # Tully III (ECR): starts on the LOWER diabat (state 1, V22=-A).  dt
        # comes from suggest_dt, which applies its automatic extra safety
        # margin for quad_coupling models -- the generic rate-based estimate
        # alone is ~15-20x too coarse here and silently gives WRONG
        # (non-converged) transfer probabilities, or blows up outright for
        # strong transfer, even though dE/dN look fine early on.
        dyn.update({
            "c0": [0, 1], "q0": [[-15.0], [-15.0]], "p0": [[20.0], [20.0]],
            "Z0": None,
            "dt": initialize.suggest_dt(initialize.tully3(**mp["kwargs"])),
            "t_max": 1700.0, "record_every": 40,
            "width_mode": "frozen", "seed": 1e-2, "eps": 1e-10,
        })

    elif case == 8:
        # Coronado three-state model: the ground-state packet (width m*omega
        # = 100) is put on state 1 at the ground-state Re (2.9, 3.3, 2.1 for
        # models I, II, III), at rest.  t_max = 3000 a.u. (~73 fs) covers the
        # paper's 70 fs.  dt = 0.5 is converged (dt 2, 1, 0.5 give the same
        # populations; model III needs <= 0.5 for a small norm drift).
        x0 = initialize.coronado_x0(mp["kwargs"]["variant"])
        dyn.update({
            "c0": [1, 0, 0], "q0": [[x0], [x0], [x0]],
            "p0": [[0.0], [0.0], [0.0]], "Z0": None,
            "dt": 0.125, "t_max": 3000.0, "record_every": 20,
            "width_mode": "thawed", "seed": 3e-3, "eps": 1e-10,
            # track-then-release: the empty states' Gaussians follow the state
            # that feeds them until population arrives (release at
            # track_factor x the seeded population); without it the packets
            # seeded at t=0 on the steep Morse walls fly off and states 2/3
            # are never populated.
            "seed_mode": "uniform", "seed_track": True, "track_factor": 3.0,
        })
    elif case == 10:
        # Same initial condition and time grid as case 8: the ground-state
        # packet (width m*omega_ground = 100) on state 1 at the ground-state
        # Re, at rest.  seed = 3e-3 (not case 8's 1e-5): it keeps the norm
        # when a tracked state is released, see the case-8 notes.
        x0 = initialize.coronado_x0(mp["kwargs"].get("variant", 1))
        dyn.update({
            "c0": [1, 0, 0], "q0": [[x0], [x0], [x0]],
            "p0": [[0.0], [0.0], [0.0]], "Z0": None,
            "dt": 0.25, "t_max": 3000.0, "dt_out": 5.0,
            "width_mode": "thawed", "seed": 3e-3, "eps": 1e-10,
            "seed_mode": "uniform", "seed_track": False,
        })
        # seed_track is NOT needed here, unlike case 8: the harmonic fit puts
        # state 2's minimum away from x0, so its seeded Gaussian slides into
        # the coupling region by itself.  Leaving it off also keeps the run
        # exactly conserving (dE 2e-12, dN 2e-9); switching it on costs a
        # norm drift of ~0.9 and, at dt=0.25, drives Re Z negative and the
        # run fails outright.

    elif case == 9:
        # The packet starts wholly in one electronic state at R0; the other
        # diabat is empty and seeded as usual.  Z0=None gives both states the
        # coherent width sqrt(m k) = 20, i.e. the papers' sigma = 0.223.
        # t_max = 8300 a.u. ~ 200 fs; dt = 0.25 is converged and conserves E
        # to ~1e-15, N to ~1e-8.
        # seed = 1e-5 matters here, and is NOT a cosmetic choice.  The two
        # diabats are displaced wells: the seeded Gaussian sits at its own
        # minimum while the occupied packet travels to the crossing alone, so
        # <g2|V12|g1> ~ 1e-6 and nothing transfers.  At a small enough seed the
        # 1/|c_j|^2 factor in the coupling corrections to qdot/pdot/Zdot (the
        # singular variational metric) drags the seeded Gaussian into the
        # coupling region by itself, and max|W21| rises to ~8e-3.  The result
        # converges as seed -> 0 (identical for 1e-6, 3e-6, 1e-5) and in dt
        # (identical for dt 0.5 ... 0.05), with dE ~ 1e-10, dN ~ 1e-7.  Above
        # a threshold near 3e-5 the dragging fails and the transfer vanishes.
        # seed_track is left False: it is not needed once the seed is small,
        # and with it on the norm breaks (dN up to 1e13) or the run crashes.
        m9 = initialize.villaseco(**mp["kwargs"])
        R0, c0 = initialize.villaseco_init(m9, mp["kwargs"].get("variant", 1))
        dyn.update({
            "c0": c0, "q0": [[R0], [R0]], "p0": [[0.0], [0.0]], "Z0": None,
            "dt": 0.25, "t_max": 8300.0, "dt_out": 5.0,
            "width_mode": "thawed", "seed": 1e-5, "seed_mode": "uniform",
            "seed_track": False, "eps": 1e-10,
        })

    else:
        raise ValueError("Unsupported case: %r" % (case,))

    return dyn


def plot_params(case):
    """Figure labels and options for one case, passed to the plot helpers."""
    if case == 0:
        return dict(names=["diabatic 1", "diabatic 2"],
                    title="Tully I -- diabatic populations")
    if case == 1:
        return dict(names=["site 1", "site 2", "site 3"],
                    t_scale=initialize.FS, t_label="time (fs)",
                    title="FMO 3-site + bath, thermal ensemble")
    if case in (2, 21):
        return dict(state=1, mode=0, t_label="time (a.u.)")
    if case == 3:
        return dict(names=["state 1", "state 2"],
                    title="Holstein: VGWP vs. exact grid propagation")
    if case == 4:
        return dict(names=["state 1", "state 2"])
    if case == 5:
        return dict(names=["state 0 (ground)", "state 1 (excited)"],
                    title="Ferretti conical intersection -- populations")
    if case == 6:
        return dict(names=["diabatic 1", "diabatic 2"],
                    title="Tully II -- dual avoided crossing populations")
    if case == 7:
        return dict(names=["state 0", "state 1"],
                    title="Tully III -- extended coupling region populations")
    if case == 8:
        return dict(names=["state 1", "state 2", "state 3"],
                    title="Coronado three-state model -- populations")
    if case == 10:
        return dict(names=["state 1", "state 2", "state 3"],
                    title="Coronado model I approximated (2 harmonic + 1 Morse)")
    if case == 9:
        return dict(names=["diabat 1", "diabat 2"],
                    title="Villaseco-Maitra-Agostini model -- diabatic populations")
    raise ValueError("Unsupported case: %r" % (case,))


# ==========================================================================
#  Run helpers
# ==========================================================================
def exact_steps(t_max_exact, dt_exact, max_steps=MAX_EXACT_STEPS):
    """(dt_exact, nsteps) for a split-operator reference covering t_max_exact.

    Caps nsteps at max_steps by COARSENING dt_exact, so that a large t_max
    cannot blow up the 2048-point grid propagation.  Prints a note when it
    does, because the reference then resolves the dynamics less finely than
    asked for.
    """
    nsteps = int(round(t_max_exact / dt_exact))
    if nsteps > max_steps:
        print("note: exact-grid reference would need %d steps (t_max=%.1f, "
              "dt_exact=%.4f) -- coarsening dt_exact to cap it at %d steps."
              % (nsteps, t_max_exact, dt_exact, max_steps))
        dt_exact = t_max_exact / max_steps
        nsteps = max_steps
    return dt_exact, nsteps


def run_case(case, model, dyn, **override):
    """Propagate one trajectory.  `case` is kept for call-site symmetry only.

    There is one engine, and compute.run already returns the per-mode marginal
    widths ("zeta"/"sigma") alongside the full "Z", so nothing needs attaching
    afterwards.  `override` replaces individual keys of `dyn` for this call
    (the FMO ensemble uses it for q0/p0/dt, the Holstein scan for q0).

    Returns the result dict.
    """
    kwargs = dict(dyn)
    kwargs.update(override)
    return compute.run(model, **kwargs)


# ==========================================================================
#  Text output
# ==========================================================================
def _header(names):
    return "  ".join("%-16s" % n for n in names)


def write_table(path, names, cols, title=""):
    """Write `cols` as a text table with a commented header line of `names`.

    Every column must have the same length.  Values are written as %.8e,
    which round-trips to about 1e-7 relative precision.  Returns `path`.
    """
    if len(names) != len(cols):
        raise ValueError("%d names but %d columns" % (len(names), len(cols)))
    head = title + "\n" + _header(names) if title else _header(names)
    np.savetxt(path, np.column_stack(cols), fmt="%.8e", header=head)
    return path


def _every(res):
    """Step spacing of the energy trace, as a string."""
    t = res["t_E"]
    if len(t) < 2:
        return "?"
    return "%d" % int(round((t[1] - t[0]) / res["dt"]))


def write_observables(res, outdir="output", modes=None, n_snapshots=25,
                      columns=True, max_full_z_columns=2000, note=""):
    """Write every recorded observable of one run to plain text in `outdir`.

    Files
    -----
    summary.txt      scalars: model name, N, D, dt, t_max, frames, width_mode,
                     width_shape, E0, dEmax, dNmax, final populations, and
                     max |Z_kl|.  Not a table -- one "name value" per line.
    observables.txt  one wide table, one row per recorded time:
                       t, energy, energy-E0, norm, |c_i|^2 per state,
                       <V_c> (coupling expectation value), Re/Im of
                       W_ij = <g_i|V_ij|g_j> for i < j, then
                       q, p, Re_z, Im_z, sigma for each (state, mode), then
                       max_offdiag_Z.
    energy.txt       t, energy, energy-E0, norm, norm-norm0 at EVERY step
                     run() checked (energy_every), which may be much finer
                     than the observables.txt frames.  dEmax/dNmax in
                     summary.txt are the maxima over this file.
    widths_full.txt  every element of every state's width matrix,
                     Re_Z<i>_<k><l> and Im_Z<i>_<k><l>, one row per recorded
                     time.  Skipped, with a note in summary.txt, when that
                     would need more than max_full_z_columns columns (FMO's
                     D=60 would need 21601).
    q.txt, p.txt, a.txt, coeff.txt, wfc.txt
                     the per-quantity files of save.write_columns, written
                     when `columns` is true.  wfc.txt (wavepacket snapshots)
                     only appears for D == 1.

    `note` is written verbatim into summary.txt as a "note" line and into
    observables.txt's title.  Use it whenever the result is not simply "the
    run": the FMO case passes "one trajectory of 8" there, because its q, p
    and widths belong to the last trajectory while the ensemble-averaged
    populations go to their own file.

    `modes` picks which nuclear modes go into observables.txt and
    widths_full.txt; the default is all of them.

    Returns the list of paths written.
    """
    os.makedirs(outdir, exist_ok=True)
    t = res["t"]
    N = res["N"]
    D = res["D"]
    if modes is None:
        modes = list(range(D))
    else:
        modes = list(modes)
    paths = []

    # ---------------------------------------------------------- summary.txt
    n_full_z = 1 + 2 * N * len(modes) * len(modes)
    skipped_z = "Z" in res and n_full_z > max_full_z_columns
    path = os.path.join(outdir, "summary.txt")
    with open(path, "w") as f:
        f.write("# na-VGWP run summary (atomic units)\n")
        f.write("model         %s\n" % res.get("model_name", ""))
        f.write("states N      %d\n" % N)
        f.write("modes  D      %d\n" % D)
        f.write("dt            %.8e\n" % res["dt"])
        f.write("t_max         %.8e\n" % t[-1])
        f.write("frames        %d\n" % len(t))
        f.write("width_mode    %s\n" % res.get("width_mode", ""))
        f.write("width_shape   %s\n" % res.get("width_shape", ""))
        f.write("width_scale   %s\n" % res.get("width_scale", 1.0))
        f.write("seed_mode     %s\n" % res.get("seed_mode", ""))
        f.write("drop_corr     %s\n" % res.get("drop_corrections", 0))
        f.write("E0            %.8e\n" % res["E0"])
        f.write("dEmax         %.8e\n" % res["dEmax"])
        f.write("dNmax         %.8e\n" % res["dNmax"])
        if "t_E" in res:
            f.write("energy steps  %d  (checked every %s steps)\n"
                    % (len(res["t_E"]), _every(res)))
        for i in range(N):
            f.write("final |c%d|^2  %.8e\n" % (i + 1, res["pops"][-1, i]))
        if "offdiag" in res:
            f.write("max |Z_kl|    %.8e\n" % res["offdiag"].max())
        if note:
            f.write("note          %s\n" % note)
        if skipped_z:
            f.write("note          widths_full.txt skipped: %d columns "
                    "exceeds max_full_z_columns=%d\n"
                    % (n_full_z, max_full_z_columns))
    paths.append(path)

    # ------------------------------------------------------ observables.txt
    names = ["t", "energy", "energy-E0", "norm"]
    cols = [t, res["energy"], res["energy"] - res["E0"], res["norm"]]
    for i in range(N):
        names.append("|c%d|^2" % (i + 1))
        cols.append(res["pops"][:, i])
    if "coupling" in res:
        names.append("<V_c>")
        cols.append(res["coupling"])
    if "W" in res:
        for i in range(N):
            for j in range(i + 1, N):
                names += ["Re_W%d%d" % (i + 1, j + 1), "Im_W%d%d" % (i + 1, j + 1)]
                cols += [res["W"][:, i, j].real, res["W"][:, i, j].imag]
    for i in range(N):
        for k in modes:
            names.append("q%d_%d" % (i + 1, k))
            cols.append(res["q"][:, i, k])
            names.append("p%d_%d" % (i + 1, k))
            cols.append(res["p"][:, i, k])
            names.append("Re_z%d_%d" % (i + 1, k))
            cols.append(np.real(res["zeta"][:, i, k]))
            names.append("Im_z%d_%d" % (i + 1, k))
            cols.append(np.imag(res["zeta"][:, i, k]))
            names.append("sigma%d_%d" % (i + 1, k))
            cols.append(res["sigma"][:, i, k])
    if "offdiag" in res:
        names.append("max_offdiag_Z")
        cols.append(res["offdiag"])
    title = "na-VGWP observables -- %s" % res.get("model_name", "")
    if note:
        title = title + "\n" + note
    paths.append(write_table(os.path.join(outdir, "observables.txt"),
                             names, cols, title=title))

    # ------------------------------------------------------- energy.txt
    if "t_E" in res:
        n0 = res["norm_steps"][0]
        paths.append(write_table(
            os.path.join(outdir, "energy.txt"),
            ["t", "energy", "energy-E0", "norm", "norm-norm0"],
            [res["t_E"], res["energy_steps"], res["energy_steps"] - res["E0"],
             res["norm_steps"], res["norm_steps"] - n0],
            title="energy and norm at every checked time step "
                  "(dEmax=%.3e, dNmax=%.3e)" % (res["dEmax"], res["dNmax"])))

    # ------------------------------------------------------ widths_full.txt
    if "Z" in res and not skipped_z:
        names = ["t"]
        cols = [t]
        for i in range(N):
            for k in modes:
                for l in modes:
                    names.append("Re_Z%d_%d%d" % (i + 1, k, l))
                    cols.append(np.real(res["Z"][:, i, k, l]))
                    names.append("Im_Z%d_%d%d" % (i + 1, k, l))
                    cols.append(np.imag(res["Z"][:, i, k, l]))
        paths.append(write_table(os.path.join(outdir, "widths_full.txt"),
                                 names, cols,
                                 title="full complex width matrix Z per state"))

    # ---------------------------------- the Libra-style per-quantity files
    if columns:
        save.write_columns(res, outdir=outdir, n_snapshots=n_snapshots,
                           mode=None if D > 1 else 0)
        for name in ("q.txt", "p.txt", "a.txt", "coeff.txt"):
            paths.append(os.path.join(outdir, name))
        if D == 1:
            paths.append(os.path.join(outdir, "wfc.txt"))

    return paths


def read_observables(path):
    """Read a table written by write_table/write_observables back in.

    Returns (names, data): `names` the column labels taken from the LAST
    comment line (so a multi-line title is fine), `data` a float array of
    shape (rows, len(names)).  Values went out as %.8e, so expect agreement
    to about 1e-7 relative, not exactly.
    """
    names = None
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            names = line
    if names is None:
        raise ValueError("no header line in %s" % path)
    names = names.lstrip("#").split()
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return names, data


# ==========================================================================
#  2-D density that keeps the width tilt
# ==========================================================================
def density_2d(res, step, state, modes, xs, ys):
    """One state's density on the (modes[0], modes[1]) plane, from the full Z.

    The density is a Gaussian with covariance Sigma = (2 Re Z)^-1.  Integrating
    out the modes that are not being plotted simply keeps the 2x2 sub-block of
    Sigma -- including its off-diagonal element, which is what tilts the
    ellipse.  The result is scaled so that its integral over the plane equals
    res["pops"][step, state] (checked numerically to 1e-15).

    Returns a real array of shape (len(xs), len(ys)).

    plot.plot_density_2d builds this same picture as an outer product of two
    1-D marginals, which assumes that off-diagonal element is zero.  That is
    right for an untilted packet and wrong for a tilted one, which is why this
    function exists.
    """
    cov = np.linalg.inv(2.0 * np.real(res["Z"][step, state]))
    sub = np.zeros((2, 2))
    for r in range(2):
        for c in range(2):
            sub[r, c] = cov[modes[r], modes[c]]
    prec = np.linalg.inv(sub)
    q0 = res["q"][step, state][modes[0]]
    q1 = res["q"][step, state][modes[1]]
    scale = res["pops"][step, state] / (2.0 * np.pi * np.sqrt(np.linalg.det(sub)))

    rho = np.zeros((len(xs), len(ys)))
    for ix in range(len(xs)):
        for iy in range(len(ys)):
            u0 = xs[ix] - q0
            u1 = ys[iy] - q1
            quad = (prec[0, 0] * u0 * u0 + 2.0 * prec[0, 1] * u0 * u1
                    + prec[1, 1] * u1 * u1)
            rho[ix, iy] = scale * np.exp(-0.5 * quad)
    return rho


def density_axis_range(res, snaps, states, mode, nsig=4.0):
    """(lo, hi) covering q +- nsig*sigma of every state and snapshot plotted.

    Needs res["sigma"], which compute.run always provides.
    """
    lo = None
    hi = None
    for step in snaps:
        for state in states:
            q = res["q"][step, state, mode]
            sig = res["sigma"][step, state, mode]
            if lo is None or q - nsig * sig < lo:
                lo = q - nsig * sig
            if hi is None or q + nsig * sig > hi:
                hi = q + nsig * sig
    return float(lo), float(hi)


def plot_density_2d_full(res, params):
    """Contour snapshots of each state's density over the plane of two modes.

    Rows are states, columns are snapshots, and each row shares one colour
    scale so that population transfer between panels is visible.  Panels come
    from density_2d(), so a tilted packet is drawn tilted; each panel title
    carries that snapshot's |Z_kl|, which is the tilt as a number.

    params keys, all optional except "snaps":
        prefix      figure title
        states      state indices, one row each (default: all)
        snaps       time-step indices, one column each
        modes       the two modes spanning the plane (default [0, 1])
        xlim, ylim  plot ranges (default: automatic, q +- 4 sigma)
        npoints     grid points per axis (default 150)
        panel_size  inches per panel (default 3.2)
        fname       optional path to save the figure

    Returns the matplotlib Figure.
    """
    states = list(params.get("states", range(res["N"])))
    snaps = list(params["snaps"])
    modes = params.get("modes", [0, 1])
    npoints = params.get("npoints", 150)

    xlim = params.get("xlim")
    if xlim is None:
        xlim = density_axis_range(res, snaps, states, modes[0])
    ylim = params.get("ylim")
    if ylim is None:
        ylim = density_axis_range(res, snaps, states, modes[1])
    xs = np.linspace(xlim[0], xlim[1], npoints)
    ys = np.linspace(ylim[0], ylim[1], npoints)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    size = params.get("panel_size", 3.2)
    fig, ax = plt.subplots(len(states), len(snaps), squeeze=False,
                           figsize=(size * len(snaps) + 1.0, size * len(states)),
                           constrained_layout=True)

    for r in range(len(states)):
        state = states[r]
        rho = []
        vmax = 0.0
        for step in snaps:
            panel = density_2d(res, step, state, modes, xs, ys)
            rho.append(panel)
            if panel.max() > vmax:
                vmax = float(panel.max())
        if vmax == 0.0:
            vmax = 1.0
        for c in range(len(snaps)):
            step = snaps[c]
            a = ax[r][c]
            cf = a.contourf(X, Y, rho[c], levels=np.linspace(0, vmax, 16),
                            cmap="viridis")
            off = abs(res["Z"][step, state][modes[0], modes[1]])
            a.set_title("state %d   t = %.1f   |Z_%d%d| = %.2f"
                        % (state, res["t"][step], modes[0], modes[1], off),
                        fontsize=9)
            a.set_xlabel("mode %d" % modes[0], fontsize=8)
            a.set_ylabel("mode %d" % modes[1], fontsize=8)
            a.tick_params(labelsize=7)
        fig.colorbar(cf, ax=ax[r].tolist(), shrink=.85,
                     label=r"$|\psi_%d|^2$" % state)

    if params.get("prefix"):
        fig.suptitle(params["prefix"])
    if params.get("fname"):
        fig.savefig(params["fname"], dpi=130, bbox_inches="tight")
    return fig
