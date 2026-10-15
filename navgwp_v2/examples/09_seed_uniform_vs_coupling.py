"""Example 9 -- two ways to seed an empty electronic state.

The equations of motion divide by |c_j|^2, so a state that starts EMPTY must
be given a small amplitude before propagation (theta = 0.01).  Two choices,
both putting population sin^2(theta) on it:

    seed_mode='uniform'   psi_1 = cos(theta) g_1,  psi_2 = sin(theta) g_1
    seed_mode='coupling'  psi_1 = cos(theta) g_1,
                          psi_2 = sin(theta) V_c g_1 / ||V_c g_1||
                          -- the shape the coupling creates (first-order
                          short-time dynamics).

For each seed the exact TDSE starts from the SAME psi_1(x,0), psi_2(x,0) as
the VGWP run, so every comparison is like-for-like.  Holstein reference model
(2 states, 1 mode), frozen and thawed, for a coherent (width_scale=1) and a
squeezed (width_scale=2) initial packet.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import navgwp2 as ng
from navgwp2 import compute

K1 = K2 = 10.0; MASS = 1.0; X0 = 1.0; Y0 = 15.0
C1 = 1.0; C2 = 0.5 * np.sqrt(K1 * MASS); C3 = 2.0
Q_START = -2.0
DT, T_MAX = 0.001, 8.0

model = ng.holstein_ref(k1=K1, k2=K2, x0=X0, y0=Y0, c1=C1, mass=MASS)
zeta_coh = np.sqrt(MASS * K1)                 # coherent width sqrt(m)*omega

# exact reference on a grid
x, k, dx = compute.grid(-8.0, 8.0, 2048)
v1 = 0.5 * K1 * x ** 2
v2 = 0.5 * K2 * (x - X0) ** 2 + Y0
vc = C1 * np.exp(-C2 * (x - C3) ** 2)

def exact_from(psi1, psi2):
    return compute.propagate_exact(psi1, psi2, x, k, MASS, v1, v2, vc,
                                   0.0005, int(round(T_MAX / 0.0005)),
                                   record_every=100)


SEEDS = ("uniform", "coupling")
rows = []
for s in (1.0, 2.0):
    # unseeded exact (psi_2 = 0), for reference only
    ex0 = exact_from(compute.gaussian_on_grid(x, s * zeta_coh, Q_START, 0.0),
                     np.zeros(len(x), complex))
    runs = {(wm, sm): ng.run(model, [1, 0], q0=[[Q_START], [0.0]],
                             p0=[[0.0], [0.0]], Z0=zeta_coh + 0j,
                             width_scale=s, dt=DT, t_max=T_MAX,
                             record_every=20, width_mode=wm, seed_mode=sm)
            for wm in ("frozen", "thawed") for sm in SEEDS}
    # each seed's exact TDSE starts from the SAME psi_1(x,0), psi_2(x,0) as
    # its VGWP run (identical for frozen and thawed), so the comparison is
    # like-for-like
    exs = {}
    for sm in SEEDS:
        psi1, psi2 = compute.reconstruct(runs[("thawed", sm)], 0, x)
        exs[sm] = exact_from(psi1, psi2)
        print("s=%g %-8s seeded exact vs unseeded exact: max|dP| %.1e"
              % (s, sm, np.max(np.abs(exs[sm]["pops"] - ex0["pops"]))))
    for wm in ("frozen", "thawed"):
        res = {}
        for sm in SEEDS:
            r, ex = runs[(wm, sm)], exs[sm]
            P = np.interp(ex["t"], r["t"], r["pops"][:, 1])
            err = np.abs(P - ex["pops"][:, 1])
            res[sm] = (r, err)
            print("s=%g %-6s %-8s  final |c2|^2 %.4f (exact %.4f)  "
                  "max|dP| %.4f  mean|dP| %.4f  dE %.0e"
                  % (s, wm, sm, r["pops"][-1, 1], ex["pops"][-1, 1],
                     err.max(), err.mean(), r["dEmax"]))
        rows.append((s, wm, ex0, res))

# ------------------------------------------------ figure 1: populations
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
for ax, (s, wm, ex, res) in zip(axs.ravel(), rows):
    ax.plot(ex["t"], ex["pops"][:, 1], "k", lw=2.5, label="exact")
    for sm, st in (("uniform", "C0--"), ("coupling", "C3-")):
        r, err = res[sm]
        ax.plot(r["t"], r["pops"][:, 1], st, lw=1.5,
                label="%s seed (mean|dP| vs own exact %.3f)" % (sm, err.mean()))
    ax.set_title("width_scale=%g, %s" % (s, wm))
    ax.grid(alpha=.3); ax.legend(fontsize=8)
for ax in axs[1]:
    ax.set_xlabel("t (a.u.)")
for ax in axs[:, 0]:
    ax.set_ylabel(r"$|c_2|^2$")
fig.suptitle("Holstein: uniform vs coupling seed of the empty state")
fig.tight_layout(); fig.savefig("example09_seed_populations.png", dpi=130)

# ------------------------------------ figure 2: what each seed looks like
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, s in zip(axs, (1.0, 2.0)):
    Z0 = compute._initial_Z(model, zeta_coh + 0j) * s
    q0 = np.array([[Q_START], [0.0]]); p0 = np.zeros((2, 1))
    g1 = compute.wavefunction(Z0[0], q0[0], p0[0], x[:, None])
    target = np.abs(vc * g1) ** 2
    ax.plot(x, np.abs(g1) ** 2 / np.max(np.abs(g1) ** 2), "k", lw=2,
            label=r"$|\psi_1|^2$ (occupied)")
    ax.plot(x, vc / C1, "0.6", lw=1, label=r"$V_{12}(x)$ (scaled)")
    ax.plot(x, target / target.max(), "C2", lw=4, alpha=.4,
            label=r"$|V_{12}\psi_1|^2$ (exact shape)")
    for sm, st in (("uniform", "C0--"), ("coupling", "C3-")):
        if sm == "uniform":
            _, q, p, Z = compute.seed_uniform([1, 0], q0, p0, Z0)
        else:
            _, q, p, Z = compute.seed_coupling(model, [1, 0], q0, p0, Z0)
        g2 = np.abs(compute.wavefunction(Z[1], q[1], p[1], x[:, None])) ** 2
        ax.plot(x, g2 / g2.max(), st, lw=1.5, label="%s seed $|g_2|^2$" % sm)
    ax.set_xlim(-4, 4); ax.set_xlabel("x"); ax.set_ylabel("normalised")
    ax.set_title("seeded state-2 Gaussian, width_scale=%g" % s)
    ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("example09_seed_shapes.png", dpi=130)
print("wrote example09_seed_populations.png, example09_seed_shapes.png")
