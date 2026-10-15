"""Example 1 -- Tully model I: scattering through an avoided crossing.

Demonstrates: the separable-diagonal path (V_1 is exponential, not polynomial),
Gaussian inter-state coupling, and writing p/q/a/coeff/wfc output files.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import navgwp2 as ng

model = ng.tully1()
print(model["name"])

# incoming packet from the left with momentum k
k = 20.0
res = ng.run(model, c0=[1, 0],
             q0=[[-10.0], [-10.0]], p0=[[k], [k]],
             dt=0.2, t_max=3000.0, record_every=20)

print("final diabatic populations: %s" % np.round(res["pops"][-1], 4))
print("final centres q1=%.2f q2=%.2f" % (res["q"][-1, 0, 0], res["q"][-1, 1, 0]))
print("conservation: dE=%.1e  dN=%.1e" % (res["dEmax"], res["dNmax"]))

ng.write_columns(res, outdir="tully1_output", n_snapshots=25)
print("wrote tully1_output/{p,q,a,coeff,wfc}.txt")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
x = np.linspace(-10, 10, 400)
V = ng.potential_curves(model, x)
ax[0].plot(x, V[:, 0], label="diabatic 1"); ax[0].plot(x, V[:, 1], label="diabatic 2")
ax[0].set_title("Tully I potentials"); ax[0].set_xlabel("x"); ax[0].legend()
ax[1].plot(res["t"], res["pops"][:, 0], label=r"$|c_1|^2$")
ax[1].plot(res["t"], res["pops"][:, 1], label=r"$|c_2|^2$")
ax[1].set_title("populations"); ax[1].set_xlabel("t"); ax[1].legend()
ax[2].plot(res["t"], res["q"][:, 0, 0], label="$q_1$")
ax[2].plot(res["t"], res["q"][:, 1, 0], label="$q_2$")
ax[2].set_title("Gaussian centres"); ax[2].set_xlabel("t"); ax[2].legend()
for a in ax:
    a.grid(alpha=.3)
fig.tight_layout(); fig.savefig("example01_tully.png", dpi=130)
print("wrote example01_tully.png")
