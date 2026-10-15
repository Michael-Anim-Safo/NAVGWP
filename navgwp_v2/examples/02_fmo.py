"""Example 2 -- FMO exciton transfer with an explicit bath.

Demonstrates: many electronic states (3 sites), many nuclear modes
(3 x 20 = 60), thermal sampling and ensemble averaging.
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

model = ng.fmo(nsites=3, nmodes_per_site=20)
print("%s   (N=%d, D=%d)" % (model["name"], model["N"], model["D"]))

dt = ng.suggest_dt(model)
t_max = 1000.0 / ng.FS
c0 = [1, 0, 0]                  # photoexcite site 1

acc = None
t0 = time.time()
for it in range(NTRAJ):
    rng = np.random.default_rng(100 + it)
    q0, p0 = ng.boltzmann_sample(model, ng.KB * T_K, rng=rng)
    r = ng.run(model, c0, q0=q0, p0=p0, dt=dt, t_max=t_max, record_every=40)
    acc = r["pops"] if acc is None else acc + r["pops"]
pops = acc / NTRAJ
print("%d trajectories in %.1f s ; dE=%.1e dN=%.1e"
      % (NTRAJ, time.time() - t0, r["dEmax"], r["dNmax"]))
print("final site populations: %s" % np.round(pops[-1], 4))

fig, ax = plt.subplots(figsize=(8, 5))
for n in range(model["N"]):
    ax.plot(r["t"] * ng.FS, pops[:, n], lw=2, label="site %d" % (n + 1))
ax.set_xlabel("time (fs)"); ax.set_ylabel("site population")
ax.set_title("FMO 3-site + bath (D=%d), %d trajectories, %g K"
             % (model["D"], NTRAJ, T_K))
ax.grid(alpha=.3); ax.legend(); ax.set_ylim(-0.02, 1.02)
fig.tight_layout(); fig.savefig("example02_fmo.png", dpi=130)
print("wrote example02_fmo.png")
