"""Example 3 -- when does the Gaussian width matter?

A: two BOUND surfaces of equal curvature, started at the coherent width
   -> thawed == frozen exactly.  Use frozen: same answer, cheaper.

B: a bound bright state coupled to a DISSOCIATIVE surface
   -> the packet must spread; a frozen Gaussian cannot, and the error feeds
      back on the transferred population through the overlap in <g1|V12|g2>.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import navgwp2 as ng

M, W1 = 1.0, 0.010


def run_both(model, zeta0, q0, t_max, dt):
    kw = dict(zeta0=zeta0, q0=q0, p0=np.zeros((2, 1)), dt=dt, t_max=t_max,
              record_every=max(1, int(20 / dt)))
    return (ng.run(model, [1, 0], width_mode="frozen", **kw),
            ng.run(model, [1, 0], width_mode="thawed", **kw))


# ---- A: bound -> bound, equal curvature -------------------------------
mA = ng.lvc(E=[0.050, 0.0455], kappa=[[0.0], [-0.0006]], omega2=[W1 ** 2],
            J=[[0, 0.0008], [0.0008, 0]], mass=np.array([M]),
            name="A: bound -> bound")
zA = np.full((2, 1), np.sqrt(M) * W1, dtype=complex)
fA, tA = run_both(mA, zA, np.array([[-40.0], [-40.0]]), 3000.0, 1.0)

# ---- B: bound -> dissociative ------------------------------------------
mB = ng.lvc(E=[0.050, 0.050], kappa=[[0.0], [-0.0007]],
            omega2=[[W1 ** 2], [0.0]], J=[[0, 0.0008], [0.0008, 0]],
            mass=np.array([M]), name="B: bound -> dissociative")
fB, tB = run_both(mB, zA, np.zeros((2, 1)), 1200.0, 0.2)

print("A (bound->bound, coherent width)")
print("  max |pop_frozen - pop_thawed| = %.3e   <- identical"
      % np.max(np.abs(fA["pops"] - tA["pops"])))
print("B (bound->dissociative)")
print("  max |pop_frozen - pop_thawed| = %.3e   <- differs"
      % np.max(np.abs(fB["pops"] - tB["pops"])))
print("  final |c2|^2:  frozen %.4f   thawed %.4f"
      % (fB["pops"][-1, 1], tB["pops"][-1, 1]))
print("  sigma_2 grows x%.1f when thawed"
      % (tB["sigma"][-1, 1, 0] / tB["sigma"][0, 1, 0]))

fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
for row, (f, t, tag) in enumerate([(fA, tA, "A: bound -> bound"),
                                   (fB, tB, "B: bound -> dissociative")]):
    ax[row, 0].plot(f["t"], f["pops"][:, 1], "C3-", lw=3, alpha=.5, label="frozen")
    ax[row, 0].plot(t["t"], t["pops"][:, 1], "C0--", lw=1.8, label="thawed")
    ax[row, 0].set_title(tag + " : population on state 2")
    ax[row, 1].plot(f["t"], f["sigma"][:, 1, 0], "C3-", lw=3, alpha=.5, label="frozen")
    ax[row, 1].plot(t["t"], t["sigma"][:, 1, 0], "C0--", lw=1.8, label="thawed")
    ax[row, 1].set_title(tag + " : width on state 2")
    for a in ax[row]:
        a.set_xlabel("t"); a.grid(alpha=.3); a.legend(fontsize=9)
fig.tight_layout(); fig.savefig("example03_frozen_vs_thawed.png", dpi=130)
print("wrote example03_frozen_vs_thawed.png")
