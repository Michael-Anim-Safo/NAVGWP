"""
navgwp2 -- Nonadiabatic Variational Gaussian Wavepacket dynamics.

Arbitrary numbers of electronic states (N) and nuclear degrees of freedom (D),
with FROZEN or THAWED Gaussian widths handled by a single code path, and a
full D x D complex width matrix per state, so the packets may tilt (a frozen
Gaussian is just a thawed one whose width is real and stationary; a per-mode
width is just a Z that stays diagonal).

Quick start
-----------
    import numpy as np, navgwp2 as ng

    model = ng.spin_boson(eps=0.0, Delta=1.0, lam=2.0, omega_c=2.5, nmodes=100)
    res   = ng.run(model, c0=[1, 0], t_max=5.0, width_mode="frozen")
    print(res["pops"][-1], res["dEmax"], res["dNmax"])

    ng.plot_populations(res)

Layout
------
    initialize.py   model builders (lvc, separable, spin_boson, fmo, tully1/2, ...)
                    and initial-condition helpers (model_widths, boltzmann_sample, ...)
    compute.py      THE engine: Gaussian integral toolbox, matrix elements,
                    equations of motion, RK4, run(), observables, and both
                    exact references.  Every Gaussian carries a full D x D
                    complex width matrix (width_shape='full' | 'diagonal').
    save.py         p/q/a/coeff/wfc text output, pickle, CSV
    plot.py         figures
"""

__version__ = "2.0.0"

from .initialize import (lvc, separable, spin_boson, fmo, tully1, tully2, tully3, coronado, coronado_x0, coronado_harmonic, villaseco, villaseco_init, holstein,
                         holstein_ref, ferretti,
                         debye_bath, suggest_dt, boltzmann_sample, model_widths,
                         CM1, FS, KB, FMO7_CM1)
from .compute import (pair_moments, kinetic_expectation, wavefunction,
                      coherent_Z, coherent_zeta, marginal_widths, state_hessian,
                      lvc_full, default_Z, suggest_dt_full,
                      pack, unpack, diagonal_terms, coupling_terms, rhs,
                      rk4_step, run, seed_uniform, seed_coupling,
                      total_norm, total_energy, coupling_expectation, coupling_matrix, populations, reconstruct,
                      density, exact_level_populations, free_width_exact,
                      breathing_width_exact,
                      grid, gaussian_on_grid, propagate_exact,
                      potential_matrix, propagate_exact_grid)
from . import compute

from .save import write_columns, save_pickle, load_pickle, save_csv
from .plot import (plot_potentials, plot_populations, plot_widths,
                   plot_conservation, plot_density, plot_density_grid,
                   plot_density_2d, mode_density,
                   plot_summary, plot_couplings, plot_surface_2d, potential_curves,
                   plot_real_width, plot_coupling_expectation,
                   plot_energy_momentum_norm, plot_coupling_element)

__all__ = [
    # initialize
    "lvc", "separable", "spin_boson", "fmo", "tully1", "tully2", "tully3", "coronado", "coronado_x0", "coronado_harmonic", "villaseco", "villaseco_init", "holstein", "holstein_ref",
    "ferretti",
    "debye_bath", "suggest_dt", "boltzmann_sample", "model_widths",
    "CM1", "FS", "KB", "FMO7_CM1",
    # compute -- gaussians
    "pair_moments", "kinetic_expectation", "wavefunction",
    "coherent_Z", "coherent_zeta", "marginal_widths", "state_hessian",
    # compute -- models
    "lvc_full", "default_Z", "suggest_dt_full",
    # compute -- dynamics
    "pack", "unpack", "diagonal_terms", "coupling_terms", "rhs", "rk4_step",
    "run", "seed_uniform", "seed_coupling",
    # compute -- observables
    "total_norm", "total_energy", "coupling_expectation", "coupling_matrix", "populations", "reconstruct", "density",
    "exact_level_populations", "free_width_exact", "breathing_width_exact",
    # compute -- exact references
    "grid", "gaussian_on_grid", "propagate_exact",
    "potential_matrix", "propagate_exact_grid",
    # save
    "write_columns", "save_pickle", "load_pickle", "save_csv",
    # plot
    "plot_potentials", "plot_populations", "plot_widths",
    "plot_conservation", "plot_density", "plot_density_grid",
    "plot_density_2d", "mode_density",
    "plot_summary", "plot_couplings", "plot_surface_2d", "potential_curves",
    "plot_real_width", "plot_coupling_expectation", "plot_energy_momentum_norm", "plot_coupling_element",
]
