# navgwp2 — nonadiabatic variational Gaussian wavepackets

Nonadiabatic dynamics with one Gaussian wavepacket per diabatic electronic
state, for any number of states `N` and nuclear modes `D`, with frozen or
thawed widths. The parameters follow the **Dirac–Frenkel (TDVP) equations of
motion**, and every matrix element is evaluated in closed form (or by
quadrature where no closed form exists), so no multidimensional grid is ever
built.

```
Psi(x,t) = sum_i c_i(t) G_i(x,t) |i>

G_i(x) = (det Re Z_i / pi^D)^{1/4}
         exp[ -1/2 (x-q_i)^T Z_i (x-q_i) + i p_i.(x-q_i) ]
```

Each state carries its own centre `q_i`, momentum `p_i`, complex symmetric
width matrix `Z_i = A_i + i B_i` and amplitude `c_i`. `Re Z` sets the spatial
width, `Im Z` the chirp. A **frozen** Gaussian is simply one whose `Z` is real
and held fixed, so frozen and thawed share a single code path.

## Layout

```
navgwp_v2/
├── navgwp2/            the package (four modules, described below)
├── examples/
│   ├── navgwp2_examples.ipynb   one notebook, 12 cases, sweeps and analysis
│   ├── model.py                 case definitions and run helpers (`import model as mdl`)
│   ├── 0*.py                    standalone example scripts
│   ├── energy_rmse.py           RMS energy error of every run in a sweep
│   └── run_steps.py             RK4 step counts and propagated times
└── tests/                       see "Tests" below
```

### `navgwp2/compute.py` — the engine (everything that evolves or integrates)

The only module that knows the equations of motion. Four groups:

- **Gaussian toolbox.** `pair_moments` (the overlap `S`, complex centre and
  covariance of `conj(g_i) g_j`, with an optional Gaussian factor folded in),
  `kinetic_expectation`, `wavefunction`, `coherent_Z` / `coherent_zeta`,
  `marginal_widths`, `default_Z`, `state_hessian`, `lvc_full`,
  `suggest_dt_full`.
- **Matrix elements.** `diagonal_terms` returns `<V_ii>`, `<grad V_ii>` and
  `<Hess V_ii>` — closed form for quadratic diabats, exact closed form for
  Morse modes carried in `model["morse"]`, Gauss–Hermite otherwise.
  `coupling_terms` returns `W_ij`, `M_ij` and (for thawed widths) `R_ij`.
- **Propagation.** `rhs` (the equations of motion), `rk4_step`, `run`, with
  `pack`/`unpack` for the real state vector, and the seeding helpers
  `seed_uniform` and `seed_coupling`.
- **Observables and references.** `total_norm`, `total_energy`,
  `coupling_matrix`, `coupling_expectation`, `populations`, `reconstruct`,
  `density`; the closed-form checks `free_width_exact`,
  `breathing_width_exact`, `exact_level_populations`; and the split-operator
  references `propagate_exact` (N = 2, D = 1), `propagate_exact_grid` (any N,
  small D) with `grid`, `gaussian_on_grid`, `potential_matrix`.

### `navgwp2/initialize.py` — models and initial conditions

Data only, no dynamics. Two generic builders — `lvc` (quadratic diabats) and
`separable` (non-polynomial separable diabats) — plus the specific systems
listed under **Models** below, and the initial-condition helpers
`model_widths`, `suggest_dt`, `boltzmann_sample`, `coronado_x0`,
`villaseco_init`, and the unit constants `CM1`, `FS`, `KB`.

### `navgwp2/plot.py` — figures

Potentials and couplings (`plot_potentials`, `potential_curves`,
`plot_couplings`, `plot_surface_2d` for D = 2); dynamics
(`plot_populations`, `plot_widths`, `plot_real_width`,
`plot_coupling_element`, `plot_coupling_expectation`, `plot_conservation`,
`plot_summary`, `plot_energy_momentum_norm`); densities (`mode_density` —
the exact reduced density along one mode — `plot_density_grid`,
`plot_density_2d`, `plot_density`). Every function returns its figure.

### `navgwp2/save.py` — output

`write_columns` writes the `p/q/a/coeff/wfc` text files plus snapshots;
`save_pickle` / `load_pickle` / `save_csv` for everything else.

### `navgwp2/__init__.py`

Re-exports the public names of all four modules, so `import navgwp2 as ng`
gives `ng.run`, `ng.tully1`, `ng.plot_populations` and so on directly.

## Requirements

`numpy`; `matplotlib` for figures; `scipy` for a few reference tests.

## Quick start

```python
import numpy as np, navgwp2 as ng

model = ng.tully1()
res = ng.run(model, c0=[1, 0], q0=[[-10.0]] * 2, p0=[[20.0]] * 2,
             dt=0.2, t_max=3000.0, width_mode="frozen")

print(res["pops"][-1], res["dEmax"], res["dNmax"])
ng.plot_populations(res)
```

`run` returns a dict with `t`, `q`, `p` `(T,N,D)`, `Z` `(T,N,D,D)` complex,
`c` `(T,N)`, the per-mode marginal widths `zeta` and `sigma`, `pops` `(T,N)`,
`energy`, `norm`, the coupling energy `coupling`, the matrix `W` of
`<g_i|V_ij|g_j>`, and the conservation diagnostics `dEmax`, `dNmax`.

## Models

`initialize.py` provides `lvc` (quadratic diabats, the general
linear-vibronic/system–bath builder), `separable` (non-polynomial separable
diabats via Gauss–Hermite), and on top of them:

| builder | system |
|---|---|
| `tully1`, `tully2`, `tully3` | the three Tully scattering models |
| `holstein`, `holstein_ref` | Holstein two-state models |
| `spin_boson`, `debye_bath` | spin–boson with a discretised Debye bath |
| `fmo` | FMO exciton model, one bath per site (D = 60 by default) |
| `ferretti` | 2-mode conical intersection |
| `coronado` | three-state Morse photodissociation, models I–III |
| `coronado_harmonic` | the same with two wells replaced by harmonic fits |
| `villaseco` | two shifted harmonic diabats, models I–V |

Couplings may be constant, linear, Gaussian, Gaussian × linear, or supplied
numerically (`quad_coupling`, D = 1). Morse diabats carried in
`model["morse"]` use exact closed-form Gaussian averages.

Some models were taken from [Libra](https://github.com/Quantum-Dynamics-Hub/libra-code).

## Initialisation

A run is fixed by four things, all arguments of `run`:

**1. Amplitudes `c0`** — one complex number per state, e.g. `[1, 0, 0]`. They
are normalised, so `|c_i|^2` are the initial populations.

**2. Centres and momenta `q0`, `p0`** — shape `(N, D)`: one row per
*electronic state*, each row holding that state's `D` nuclear coordinates.
`[[2.9]] * 3` therefore means three states, one mode, all starting at
x = 2.9. `None` gives zeros.

**3. Widths `Z0` and `width_scale`** — `Z0=None` gives each state its
coherent width, `A = sqrt(K m)` from that state's own curvature `K` (for a
`separable` model, from `omega2_ref`, which exists only for this purpose).
`Z0` also accepts a scalar, a `(D,)` diagonal, a `(D,D)` matrix or a full
`(N,D,D)`. `width_scale=s` multiplies whatever results: `s > 1` squeezes the
packet, `s < 1` broadens it, and with `width_mode="thawed"` any `s != 1`
makes it breathe on its own surface. `width_mode` is `"frozen"` or
`"thawed"`; `width_shape` is `"full"` (general `Z`, packets may tilt) or
`"diagonal"` (one independent width per mode).

**4. Seeding of empty states.** The equations divide by `|c_j|^2`, so a state
with exactly zero population has no defined trajectory and must be given
some. `seed` is an angle `theta`: the occupied state is scaled by
`cos(theta)` and each empty state receives `sin(theta)`, so its population is
`sin^2(theta)` and the norm is preserved exactly. `seed_mode` decides where
its Gaussian goes:

| `seed_mode` | centre | width | phase |
|---|---|---|---|
| `"uniform"` | as supplied in `q0` | as supplied in `Z0` | real |
| `"coupling"` | fitted to `V_ji g_i` | fitted (`A_i + 2 beta`) | `-i` |
| `"shift"` | fitted to `V_ji g_i` | kept (its coherent width) | `-i` |

`"coupling"` is exact when `V_ji g_i` is itself a Gaussian (constant and
Gaussian couplings); otherwise the moments come from quadrature (D <= 3).

`seed_track=True` additionally holds each seeded state's `q, p, Z` equal to
those of the state feeding it — the most populated state it is directly
coupled to — until its population exceeds `track_factor * sin^2(seed)`,
then releases it (`result["release_t"]` records when). This is needed where a
seeded Gaussian would otherwise be accelerated out of the coupling region
before population arrives. It is an imposed constraint, not a variational
evolution: while it is active, energy and norm are conserved only to the
order of that state's population.

**Other `run` options:** `dt`, `t_max`, `dt_out` / `record_every`,
`energy_every` (set to 1 to check conservation at every step), `eps` (the
regulariser in `1/(|c|^2 + eps)`), `drop_corrections`, `legacy`, and `stop`
(a callable to end the run early).

## Exact references

`compute.propagate_exact` (two states, one mode, analytic 2×2 propagator) and
`compute.propagate_exact_grid` (any `N`, small `D`, eigendecomposition at each
grid point) perform split-operator propagation of the full Schrödinger
equation, started from the same initial wavefunction as the variational run.
Cost scales as `n^D`, so they are benchmarks for `D = 1–2`, not production
methods.

## Running the examples

### The notebook

`examples/navgwp2_examples.ipynb` runs one case at a time. The cells are in
dependency order, and the first four are all you normally touch:

1. **imports** — adds `..` and `.` to `sys.path`, imports the package and
   `model.py` as `mdl`.
2. **reload** — `importlib.reload` for the package and `model.py`, so edits
   to either take effect without restarting the kernel.
3. **`case = ...`** — the case to run, plus `variant`, `fit` and `b3_scale`
   for the cases that have them.
4. **`model_params`** — `mdl.model_params(case)`, with the knobs from cell 3
   applied; **`dyn_params`** — `mdl.dyn_params(case, model_params)`;
   **`plot_params`**.

Then: build the model and plot the surfaces, run the dynamics (with the exact
SOFT reference where available), plot populations, the four-panel summary,
and density snapshots. Each of those cells dispatches on `case`.

**Changing case or variant:** re-run from the `case` cell **downwards**.
`dyn_params` freezes the starting point `q0` when it runs, so changing the
variant without re-running it leaves the new surfaces with the old initial
condition. The `dyn_params` cell prints `case, variant -> q0` so you can see
what it used.

| case | model | | case | model |
|---|---|---|---|---|
| 0 | Tully I | | 5 | Ferretti (D = 2) |
| 1 | FMO, thermal ensemble | | 6, 7 | Tully II, Tully III |
| 2, 21 | bound→bound, bound→dissociative | | 8 | Coronado Morse, variants 1–3 |
| 3, 4 | Holstein vs exact, coupling scan | | 9 | Villaseco/Pieroni, variants 1–5 |
| | | | 10 | Coronado approximated, `fit`, `b3_scale` |

Later cells run seed sweeps (seed mode × ε × with/without coupling
corrections, each compared against SOFT) and the analysis helpers:
`energy_rmse()` (RMS energy error per run), `run_steps()` (RK4 steps and
propagated time) and `release_times()` (when each seeded state is released).
Each writes text files beside the sweep it analyses.

### From a script

`examples/model.py` is importable on its own, so the same cases run without
the notebook:

```python
import model as mdl

mp  = mdl.model_params(8)          # Coronado
mp["kwargs"]["variant"] = 2        # model II
m   = mdl.build_model(mp)
dyn = mdl.dyn_params(8, mp)        # read q0, dt, t_max, seeding from here
res = mdl.run_case(8, m, dyn)      # or override: mdl.run_case(8, m, dyn, dt=0.05)

mdl.write_observables(res, mdl.outdir_for(8))
```

The numbered scripts `examples/0*.py` are standalone and need no notebook:
`python examples/01_tully.py`.
