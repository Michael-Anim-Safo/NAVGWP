# navgwp2 — Nonadiabatic Variational Gaussian Wavepacket dynamics

Arbitrary numbers of **electronic states** `N` and **nuclear degrees of freedom** `D`,
with **frozen or thawed** Gaussian widths handled by a *single* code path.
All matrix elements are closed form, so cost is `O(N²D)` per step — no
multidimensional quadrature, and hundreds of modes are routine.

```
navgwp_v2/
├── navgwp2/              the package
│   ├── gaussians.py      Gaussian integral toolbox (complex widths)
│   ├── models.py         lvc, separable, spin_boson, fmo, tully1/2, baths
│   ├── dynamics.py       matrix elements, EOMs, RK4, run()
│   ├── observables.py    norm, energy, reconstruction, analytic references
│   ├── exact.py          numerically exact 2-state grid propagator
│   ├── io.py             p/q/a/coeff/wfc text output, pickle, CSV
│   └── plotting.py       figures
├── examples/             01_tully, 02_fmo, 03_frozen_vs_thawed,
│                         04_holstein, 05_holstein_accuracy
└── run_tests.py          test runner (the tests/ suite it ran has been removed)
```

## Install / run

Only `numpy` is required (`scipy` for one reference test, `matplotlib` for plots).

```bash
python examples/01_tully.py
```

## Quick start

```python
import numpy as np, navgwp2 as ng

model = ng.spin_boson(eps=0.0, Delta=1.0, lam=2.0, omega_c=2.5, nmodes=200)
q0, p0 = ng.boltzmann_sample(model, kT=1.0)
res = ng.run(model, c0=[1, 0], q0=q0, p0=p0, t_max=5.0)

print(res["pops"][-1], res["dEmax"], res["dNmax"])
ng.plot_populations(res, fname="pops.png")
```

Result arrays: `t (T,)`, `q p (T,N,D)`, `zeta (T,N,D)` complex, `sigma (T,N,D)`,
`pops (T,N)`, `energy`, `norm`, plus `dEmax`, `dNmax`.

## The ansatz

One Gaussian per electronic state,

```
Psi(x,t) = sum_i c_i(t) g_i(x,t) |i>

g_i(x) = prod_k (Re zeta_ik/pi)^{1/4}
         exp[ -zeta_ik/2 (x_k-q_ik)^2 + i p_ik (x_k-q_ik) ]
```

`zeta = a + ib` is complex: `Re zeta` sets the width `sigma = 1/sqrt(2 Re zeta)`,
`Im zeta` is the chirp that lets the packet spread. **A frozen Gaussian is just
a thawed one whose width happens to be real and stationary** — hence one code path.

## Model classes

| builder | diagonal | coupling | exact? |
|---|---|---|---|
| `lvc(E, kappa, omega2, J, lam)` | `E_i + kappa_i·x + ½Σω²x²` | `J_ij + lam_ij·x` | yes, closed form |
| `separable(vfun, dvfun, ...)` | `Σ_k v_i(k,x_k)` (any shape) | as above (+ Gaussian) | yes, 1-D Gauss–Hermite |
| `spin_boson`, `fmo` | harmonic bath | constant | yes |
| `holstein` | 2-site dimer, 1 mode | constant | yes |
| `holstein_ref` | reference params (k1=k2=10, x0=1, y0=15) | Gaussian at qc=2 | yes |
| `tully1`, `tully2` | tanh (or piecewise exponential) / Gaussian | Gaussian | yes |

Optional Gaussian coupling `A_ij exp[-Σβ_k(x_k-x_ck)²]` is supported analytically.

## Frozen vs thawed

```python
ng.run(model, c0, width_mode="frozen")   # widths fixed (default)
ng.run(model, c0, width_mode="thawed")   # widths evolve
```

Widths follow Heller's equation `zetadot = -i(zeta²/m - V'')`, which for this
model class is **exact**. Three limits:

* `zeta = sqrt(m)·omega` → stationary (coherent state) → **thawed ≡ frozen, bitwise**
* other `zeta` → breathes at `2ω/√m`
* `omega = 0` → `zeta(t) = zeta0/(1 + i·zeta0·t/m)` → free spreading

**Use frozen** when all surfaces are harmonic with similar curvature and you start
at the coherent width (FMO, spin-boson, LVC baths) — identical answer, cheaper.
**Use thawed** for dissociation, free motion, or very different curvatures; see
`examples/03`, where frozen vs thawed give 0.0439 vs 0.0348 for the transferred
population.

## Numerical safeguards (both matter)

1. **Global-phase gauge.** The mean of the electronic diagonal is subtracted each
   step. It is identical for every state, so it is a pure global phase —
   populations, coherences, forces and energy are exactly unchanged — but it
   removes the `O(D)` common kinetic phase. Without it, norm drift at `D=1000`
   is `O(1)`; with it, `~1e-9`.
2. **Seeding.** A state starting exactly empty makes `1/|c_i|²` blow up. `run`
   rotates `theta=1e-2` onto empty states by default (`seed=None` disables),
   improving norm conservation from `2e-1` to `1e-10` in the standard test.

## Verification (historical record)

**The `tests/` suite was removed from the tree on 2026-09-23.** What follows
is the record of what it checked and the last result it gave (58 tests, 0
failures, 68 s) -- it documents how the code was validated, but there is
nothing left to re-run, and `run_tests.py` has no suite to collect.

| file | what it checks |
|---|---|
| `test_gaussians.py` | overlap, `zbar`, `<x²>`, Gaussian coupling and `<T>` **vs brute-force grid integration** in D=1,2,3 with complex unequal widths; frozen limit; `W` exactly Hermitian |
| `test_limits.py` | bare-electronic limit vs `expm(-iHt)`; free-particle spreading law; coherent width stationary; breathing vs closed form and at `2ω/√m`; thawed≡frozen bitwise; classical free motion |
| `test_conservation.py` | norm/energy for spin-boson at D=5…200; conservation independent of D; seeding cures the singularity; populations sum to 1; constant energy shift changes nothing |
| `test_holstein_vs_exact.py` | **accuracy of the ansatz** vs numerically exact split-operator grid propagation, at the reference Holstein parameters: potentials reproduce the intended forms, crossing coincides with the coupling centre, transferred population matches exact, error grows with coupling, `<x>` tracks the exact mean position and turns at the crossing |
| `test_models_io.py` | model shapes, FMO matches the reference matrix, Debye reorganisation energy, separable diagonal vs analytic, Tully builds/runs/transfers, file output with normalised `wfc` snapshots, pickle round-trip |

```
58 passed, 0 failed        (last run before removal)
```

## Scope and limitations

* Matrix elements are exact for the model classes above. A general
  *non*-separable anharmonic potential would need a local harmonic
  approximation (the moments are already available).
* **One Gaussian per state**: cannot describe branching or bifurcation on a
  single surface. Short-time coherent dynamics is good; full thermal relaxation
  needs ensemble averaging and is still approximate. The systematic fix is a
  multiconfigurational ansatz (several Gaussians per state, vMCG/G-MCTDH).
* Most tests validate the *implementation*. `test_holstein_vs_exact.py` is the
  exception: it checks the **accuracy of the ansatz** against exact grid
  propagation for the reference Holstein model
  (`am=1, ak1=ak2=10, aka=1, qs1=0, qs2=1, qc=2, v0=15, beta=sqrt(ak1*am)/2`):

  ```
  V_11 = 5 x^2      V_22 = 5(x-1)^2 + 15      V_12 = 1.0 exp[-1.5811 (x-2)^2]
  ```

  The parameters are internally consistent: the parabolas cross at `x = 2`,
  exactly the centre of the Gaussian coupling, and a coherent state released
  at rest from `q0 = -2` has energy 20 = the crossing energy, so it reaches
  the coupling region precisely at its classical turning point.

  Measured accuracy at `q0 = -2` (`examples/04`, `examples/05`):

  | quantity | exact | VGWP |
  |---|---|---|
  | final `abs(c2)^2` (t=10, seed B) | 0.2484 | 0.2502 |
  | final `abs(c2)^2` (t=10, seed C) | 0.2484 | 0.2635 |

  | coupling `c1` (aka) | max abs error |
  |---|---|
  | 0.25 | 0.0067 |
  | 1.00 (reference) | 0.0221 |
  | 2.00 | 0.264 |

  So at the reference coupling the transferred population is reproduced to
  under 1%, degrading as the coupling is increased and the exact wavefunction
  develops structure a single Gaussian cannot represent. For a dissipative
  many-mode problem such as FMO the analogous check would need HEOM.
