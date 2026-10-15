"""Example 2b -- FMO exciton transfer: frozen vs thawed bath Gaussians.

Same model and thermal ensemble as 02_fmo.py (3 sites x 20 modes = 60, 77 K),
propagated three ways from IDENTICAL Boltzmann samples, so any difference is
due to the width treatment alone:

    frozen          Z held at its coherent value (what 02_fmo.py does)
    thawed-diag     Z evolves, but only its diagonal (per-mode widths, no tilt)
    thawed-full     Z evolves as a full 60x60 matrix (packets may tilt)

On its own surface each bath packet starts at the coherent width, so Heller's
term alone would keep it stationary; any width change comes from the
inter-state coupling bracket in Zdot (see compute.py / compute_derivations.pdf).
"""
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import navgwp2 as ng

NTRAJ = 8                       # raise for smoother curves
T_K = 77.0
VARIANTS = [                    # (label, width_mode, width_shape)
    ("frozen",      "frozen", "full"),
    ("thawed-diag", "thawed", "diagonal"),
    ("thawed-full", "thawed", "full"),
]

model = ng.fmo(nsites=3, nmodes_per_site=20)
N, D = model["N"], model["D"]
print("%s   (N=%d, D=%d)" % (model["name"], N, D))

dt = ng.suggest_dt(model)
t_max = 1000.0 / ng.FS
c0 = [1, 0, 0]                  # photoexcite site 1

# one set of thermal initial conditions, shared by every variant
samples = []
for it in range(NTRAJ):
    rng = np.random.default_rng(100 + it)
    samples.append(ng.boltzmann_sample(model, ng.KB * T_K, rng=rng))

out = {}
for label, mode, shape in VARIANTS:
    pops = sigma = offdiag = None
    dE = dN = 0.0
    E_drift, N_drift = [], []          # per trajectory, at EVERY time step
    t0 = time.time()
    for q0, p0 in samples:
        r = ng.run(model, c0, q0=q0, p0=p0, dt=dt, t_max=t_max,
                   width_mode=mode, width_shape=shape, record_every=40,
                   energy_every=1)
        pops = r["pops"] if pops is None else pops + r["pops"]
        sigma = r["sigma"] if sigma is None else sigma + r["sigma"]
        offdiag = r["offdiag"] if offdiag is None else np.maximum(offdiag, r["offdiag"])
        dE, dN = max(dE, r["dEmax"]), max(dN, r["dNmax"])
        E_drift.append(r["energy_steps"] - r["E0"])
        N_drift.append(r["norm_steps"] - r["norm_steps"][0])
    out[label] = dict(t=r["t"] * ng.FS, pops=pops / NTRAJ, sigma=sigma / NTRAJ,
                      offdiag=offdiag, dE=dE, dN=dN, t_E=r["t_E"] * ng.FS,
                      E_drift=np.array(E_drift), N_drift=np.array(N_drift))
    print("%-12s %d traj in %6.1f s ; max dE=%.1e dN=%.1e ; max|Z_kl|=%.2e ;"
          " final pops %s"
          % (label, NTRAJ, time.time() - t0, dE, dN, offdiag.max(),
             np.round(pops[-1] / NTRAJ, 4)))

# ---------------------------------------------------- energy conservation
# The variational equations conserve <H> exactly, so any drift is RK4 /
# round-off error.  Judge it against the smallest physical energy scale in
# the model: the weakest electronic coupling |J| between the sites.
J_min = np.min(np.abs(model["J"][np.nonzero(model["J"])]))
print("\nenergy conservation, checked at every one of %d steps "
      "(weakest coupling |J| = %.1f cm^-1):"
      % (len(out["frozen"]["t_E"]) - 1, J_min / ng.CM1))
for label, d in out.items():
    ok = d["dE"] < 1e-3 * J_min
    print("  %-12s max|E(t)-E0| = %.2e Eh = %.2e cm^-1 = %.1e |J|   "
          "max|N(t)-N0| = %.1e   %s"
          % (label, d["dE"], d["dE"] / ng.CM1, d["dE"] / J_min, d["dN"],
             "OK" if ok else "CHECK (drift > 1e-3 |J|: try a smaller dt)"))
    np.savetxt("example02b_energy_%s.txt" % label,
               np.column_stack([d["t_E"], d["E_drift"].T, d["N_drift"].T]),
               fmt="%.10e",
               header="t_fs  E(t)-E0 for traj 1..%d (Eh)  N(t)-N0 for traj 1..%d"
                      % (NTRAJ, NTRAJ))
print("wrote example02b_energy_<variant>.txt (every step, every trajectory)\n")

ref = out["frozen"]["pops"]
for label in ("thawed-diag", "thawed-full"):
    print("max |P_%s - P_frozen| = %.2e"
          % (label, np.max(np.abs(out[label]["pops"] - ref))))

np.savez("example02b_fmo_frozen_vs_thawed.npz",
         **{"%s_%s" % (lab, key): val for lab, d in out.items()
            for key, val in d.items()})

# ---------------------------------------------------------------- figure
styles = {"frozen": "-", "thawed-diag": "--", "thawed-full": ":"}
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

ax = axs[0, 0]
for label, d in out.items():
    for n in range(N):
        ax.plot(d["t"], d["pops"][:, n], styles[label], color=colors[n], lw=2,
                label="site %d, %s" % (n + 1, label))
ax.set_xlabel("time (fs)"); ax.set_ylabel("site population")
ax.set_title("Site populations (%d traj, %g K)" % (NTRAJ, T_K))
ax.set_ylim(-0.02, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=3)

ax = axs[0, 1]
for label in ("thawed-diag", "thawed-full"):
    d = out[label]
    for n in range(N):
        ax.plot(d["t"], d["pops"][:, n] - ref[:, n], styles[label],
                color=colors[n], lw=1.5, label="site %d, %s" % (n + 1, label))
ax.axhline(0, color="k", lw=.5)
ax.set_xlabel("time (fs)"); ax.set_ylabel("P(thawed) - P(frozen)")
ax.set_title("Effect of thawing on populations")
ax.grid(alpha=.3); ax.legend(fontsize=7)

ax = axs[1, 0]
# relative change of the marginal width sigma, averaged over modes, per state
for label in ("thawed-diag", "thawed-full"):
    d = out[label]
    rel = d["sigma"] / d["sigma"][0][None] - 1.0          # (T,N,D)
    for n in range(N):
        ax.plot(d["t"], np.mean(np.abs(rel[:, n, :]), axis=-1), styles[label],
                color=colors[n], lw=1.5, label="state %d, %s" % (n + 1, label))
ax.set_xlabel("time (fs)"); ax.set_ylabel(r"mean $|\sigma/\sigma_0 - 1|$ over modes")
ax.set_title("How much the bath widths move")
ax.grid(alpha=.3); ax.legend(fontsize=7)

ax = axs[1, 1]
for label in ("thawed-diag", "thawed-full"):
    ax.semilogy(out[label]["t"], np.maximum(out[label]["offdiag"], 1e-300),
                styles[label], color="k", lw=1.5, label=label)
ax.set_ylim(1e-12, None)
ax.set_xlabel("time (fs)"); ax.set_ylabel(r"max $|Z_{kl}|,\ k\neq l$")
ax.set_title("Packet tilt (0 for frozen and thawed-diag)")
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)

fig.suptitle("FMO 3-site + bath (D=%d): frozen vs thawed Gaussians" % D)
fig.tight_layout(); fig.savefig("example02b_fmo_frozen_vs_thawed.png", dpi=130)
print("wrote example02b_fmo_frozen_vs_thawed.png / .npz")

# ------------------------------------------- energy conservation figure
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
for label, d in out.items():
    for it in range(NTRAJ):
        axs[0].plot(d["t_E"], d["E_drift"][it] / ng.CM1, styles[label],
                    color=colors[list(out).index(label)], lw=.8,
                    label=label if it == 0 else None)
        axs[1].plot(d["t_E"], d["N_drift"][it], styles[label],
                    color=colors[list(out).index(label)], lw=.8,
                    label=label if it == 0 else None)
axs[0].set_ylabel(r"$E(t)-E_0$ (cm$^{-1}$)")
axs[0].set_title("Energy drift, every step, all %d trajectories" % NTRAJ)
axs[1].set_ylabel(r"$N(t)-N_0$")
axs[1].set_title("Norm drift, every step")
for ax in axs:
    ax.set_xlabel("time (fs)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("example02b_energy_conservation.png", dpi=130)
print("wrote example02b_energy_conservation.png")
