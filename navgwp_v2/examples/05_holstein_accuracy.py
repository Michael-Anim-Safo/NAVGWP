"""Example 5 -- accuracy of the ansatz for the reference Holstein model.

Scans the Gaussian coupling amplitude c1 (= aka) and the initial displacement
q0 about the reference point (c1 = 1, q0 = -2), measuring the error of the
variational result against numerically exact grid propagation.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import navgwp2 as ng

K1 = K2 = 10.0
X0, Y0, C3, MASS = 1.0, 15.0, 2.0, 1.0
C2 = 0.5 * np.sqrt(K1 * MASS)
OMEGA = np.sqrt(K1 / MASS)
ZETA0 = np.sqrt(MASS) * OMEGA
T_MAX, DT_EX, DT_VG = 8.0, 0.001, 0.002


def compare(c1, q0):
    x, k, _ = ng.grid(-8.0, 8.0, 2048)
    v1 = 0.5 * K1 * x ** 2
    v2 = 0.5 * K2 * (x - X0) ** 2 + Y0
    vc = c1 * np.exp(-C2 * (x - C3) ** 2)
    psi1 = ng.gaussian_on_grid(x, ZETA0, q0, 0.0)
    ex = ng.propagate_exact(psi1, np.zeros_like(psi1), x, k, MASS, v1, v2, vc,
                            DT_EX, int(round(T_MAX / DT_EX)), record_every=100)
    m = ng.holstein_ref(k1=K1, k2=K2, x0=X0, y0=Y0, c1=c1, c2=C2, c3=C3,
                        mass=MASS)
    r = ng.run(m, c0=[1, 0], q0=[[q0], [q0]], p0=[[0.0], [0.0]],
               zeta0=ZETA0 + 0j, dt=DT_VG, t_max=T_MAX, record_every=20)
    P = np.array([np.interp(ex["t"], r["t"], r["pops"][:, i])
                  for i in (0, 1)]).T
    return ex, P, np.max(np.abs(P - ex["pops"])), ex["pops"][-1, 1], P[-1, 1]


print("Holstein reference model: k1=k2=10, x0=1, y0=15, qc=2, beta=%.4f" % C2)
print("\nscan of the coupling amplitude c1 (aka), q0 = -2")
print(" %-6s %-12s %-12s %-12s" % ("c1", "max|dP|", "|c2|^2 exact", "VGWP"))
c1s = [0.25, 1.0, 2.0]
errs, fex, fvg, curves = [], [], [], {}
for c1 in c1s:
    ex, P, e, f_e, f_v = compare(c1, -2.0)
    errs.append(e); fex.append(f_e); fvg.append(f_v); curves[c1] = (ex, P)
    print(" %-6.2f %-12.4f %-12.4f %-12.4f" % (c1, e, f_e, f_v))

print("\nscan of the initial displacement q0 (c1 = 1)")
q0s = [-2.0, -2.5]
errs_q = []
for q0 in q0s:
    _, _, e, f_e, f_v = compare(1.0, q0)
    errs_q.append(e)
    print("  q0 = %5.2f   max|dP| = %.4f   |c2|^2 exact %.4f  VGWP %.4f"
          % (q0, e, f_e, f_v))

# ------------------------------------------------------------- figure ----
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

ax[0].plot(c1s, errs, "o-", color="C0")
ax[0].set_xlabel(r"coupling amplitude $c_1$ (aka)")
ax[0].set_ylabel(r"max $|\Delta P|$")
ax[0].set_title("error vs coupling strength"); ax[0].grid(alpha=.3)

ax[1].plot(c1s, fex, "ks-", lw=2, label="exact")
ax[1].plot(c1s, fvg, "C1o--", lw=2, label="VGWP")
ax[1].set_xlabel(r"$c_1$"); ax[1].set_ylabel(r"final $|c_2|^2$")
ax[1].set_title("transferred population"); ax[1].legend(fontsize=9)
ax[1].grid(alpha=.3)

for c1, col in [(0.25, "C0"), (1.0, "C2"), (2.0, "C3")]:
    ex, P = curves[c1]
    ax[2].plot(ex["t"], ex["pops"][:, 1], col + "-", lw=2.5, alpha=.45)
    ax[2].plot(ex["t"], P[:, 1], col + "--", lw=1.5, label="$c_1$=%.2f" % c1)
ax[2].set_xlabel("t"); ax[2].set_ylabel(r"$|c_2|^2$")
ax[2].set_title("solid = exact, dashed = VGWP")
ax[2].legend(fontsize=9); ax[2].grid(alpha=.3)

fig.suptitle("Accuracy of one Gaussian per state -- Holstein reference model",
             fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("example05_holstein_accuracy.png", dpi=130)
print("\nwrote example05_holstein_accuracy.png")
