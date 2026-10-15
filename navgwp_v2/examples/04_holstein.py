"""Example 4 -- Holstein model (reference parameters) vs EXACT grid propagation.

Model 3 of the reference Fortran source:
    am = 1, ak1 = ak2 = 10, aka = 1, qs1 = 0, qs2 = 1, qc = 2, v0 = 15,
    beta = sqrt(ak1*am)/2

    V_11(x) = 5 x^2
    V_22(x) = 5 (x-1)^2 + 15
    V_12(x) = 1.0 * exp[ -1.5811 (x - 2)^2 ]

The two parabolas cross at x = 2, exactly where the Gaussian coupling is
centred.  A coherent state released at rest from q = -2 has energy 20, which
is the crossing energy, so it arrives at the coupling region precisely at its
classical turning point.

Because this is 2 states x 1 mode we can solve the full two-component TDSE
exactly on a grid (split-operator FFT) and check the variational result.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import navgwp2 as ng

# ------------------------------------------------------------- parameters --
K1 = K2 = 10.0
X0, Y0 = 1.0, 15.0
C1, C3 = 1.0, 2.0
MASS = 1.0
C2 = 0.5 * np.sqrt(K1 * MASS)          # beta = sqrt(ak1*am)/2

Q0, P0 = -2.0, 0.0                     # coherent g1 released at rest
T_MAX = 8.0
DT_EX, DT_VG = 0.0005, 0.001

model = ng.holstein_ref(k1=K1, k2=K2, x0=X0, y0=Y0, c1=C1, c2=C2, c3=C3,
                        mass=MASS)
OMEGA = np.sqrt(K1 / MASS)
ZETA0 = np.sqrt(MASS) * OMEGA          # coherent (stationary) width

print(model["name"])
print("omega = %.4f   period = %.4f   coherent width zeta = %.4f (sigma=%.4f)"
      % (OMEGA, 2 * np.pi / OMEGA, ZETA0, 1 / np.sqrt(2 * ZETA0)))
print("V1(-2) = %.1f, crossing energy V1(2) = %.1f  -> released at the "
      "turning point\n" % (0.5 * K1 * Q0 ** 2, 0.5 * K1 * C3 ** 2))

# ------------------------------------------------------- exact reference --
x, k, dx = ng.grid(-8.0, 8.0, 2048)
v1 = 0.5 * K1 * x ** 2
v2 = 0.5 * K2 * (x - X0) ** 2 + Y0
vc = C1 * np.exp(-C2 * (x - C3) ** 2)

psi1 = ng.gaussian_on_grid(x, ZETA0, Q0, P0)
ex = ng.propagate_exact(psi1, np.zeros_like(psi1), x, k, MASS, v1, v2, vc,
                        DT_EX, int(round(T_MAX / DT_EX)), record_every=100)
print("exact (2048 grid pts, dt=%.4g): norm drift %.2e"
      % (DT_EX, np.max(np.abs(ex["norm"] - 1.0))))

# ---------------------------------------------------------- variational --
def vgwp(q2, label, width_mode="frozen"):
    r = ng.run(model, c0=[1, 0], q0=[[Q0], [q2]], p0=[[P0], [0.0]],
               zeta0=ZETA0 + 0j, dt=DT_VG, t_max=T_MAX, record_every=20,
               width_mode=width_mode, seed=1e-2)
    P = np.array([np.interp(ex["t"], r["t"], r["pops"][:, i])
                  for i in (0, 1)]).T
    print("  %-32s final |c2|^2 = %.4f   max|dP| = %.4f   dE=%.1e dN=%.1e"
          % (label, r["pops"][-1, 1], np.max(np.abs(P - ex["pops"])),
             r["dEmax"], r["dNmax"]))
    return r, P

print("variational (one Gaussian per state):")
rB, PB = vgwp(Q0, "B: seed at q2 = q1 = -2")
rC, PC = vgwp(0.0, "C: 'shifted' seed, q2 = 0")
rT, PT = vgwp(0.0, "C with thawed widths", width_mode="thawed")
print("\nexact final |c2|^2 = %.4f" % ex["pops"][-1, 1])

# ------------------------------------------------------------- figures ---
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

xx = np.linspace(-3.5, 4.0, 600)
ax[0].plot(xx, 0.5 * K1 * xx ** 2, "C0", lw=2, label=r"$V_{11}=5x^2$")
ax[0].plot(xx, 0.5 * K2 * (xx - X0) ** 2 + Y0, "C3", lw=2,
           label=r"$V_{22}=5(x-1)^2+15$")
ax[0].plot(xx, 20 * C1 * np.exp(-C2 * (xx - C3) ** 2), "C2--", lw=1.6,
           label=r"$20\times V_{12}$")
Vm = np.zeros((len(xx), 2, 2))
Vm[:, 0, 0] = 0.5 * K1 * xx ** 2
Vm[:, 1, 1] = 0.5 * K2 * (xx - X0) ** 2 + Y0
Vm[:, 0, 1] = Vm[:, 1, 0] = C1 * np.exp(-C2 * (xx - C3) ** 2)
Ead = np.linalg.eigvalsh(Vm)
ax[0].plot(xx, Ead[:, 0], "k:", lw=1.4, label="adiabatic")
ax[0].plot(xx, Ead[:, 1], "k:", lw=1.4)
ax[0].axvline(C3, color="0.6", ls="--", lw=1)
ax[0].axvline(Q0, color="0.8", ls=":", lw=1)
ax[0].set_xlabel("x"); ax[0].set_ylabel("V"); ax[0].set_ylim(-2, 50)
ax[0].set_title("crossing at x=2 = coupling centre")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

ax[1].plot(ex["t"], ex["pops"][:, 1], "k-", lw=3, alpha=.45, label="exact (grid)")
ax[1].plot(ex["t"], PB[:, 1], "C1--", lw=1.7, label="VGWP B")
ax[1].plot(ex["t"], PC[:, 1], "C0-.", lw=1.7, label="VGWP C")
ax[1].set_xlabel("t"); ax[1].set_ylabel(r"$|c_2|^2$")
ax[1].set_title("population transferred to state 2")
ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)

ax[2].plot(ex["t"], ex["xmean"][:, 0], "k-", lw=3, alpha=.45,
           label=r"exact $\langle x\rangle_1$")
ax[2].plot(rC["t"], rC["q"][:, 0, 0], "C0-.", lw=1.7, label="VGWP $q_1$")
ax[2].plot(rC["t"], rC["q"][:, 1, 0], "C3-.", lw=1.7, label="VGWP $q_2$")
ax[2].axhline(C3, color="0.6", ls="--", lw=1)
ax[2].set_xlabel("t"); ax[2].set_ylabel(r"$\langle x\rangle$")
ax[2].set_title("nuclear motion (dashed line = crossing)")
ax[2].legend(fontsize=9); ax[2].grid(alpha=.3)

fig.suptitle("Holstein model, reference parameters: variational Gaussians vs "
             "exact grid propagation", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("example04_holstein.png", dpi=130)

ng.write_columns(rC, outdir="holstein_output", n_snapshots=25)
print("\nwrote example04_holstein.png and holstein_output/{p,q,a,coeff,wfc}.txt")
