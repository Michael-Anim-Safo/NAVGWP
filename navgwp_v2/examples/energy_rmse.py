"""RMS energy error of every run in a seed sweep.

Energy is a conserved quantity of the exact TDVP flow, so the error measure is
the deviation from its initial value over the M recorded steps,

    RMSE_E = sqrt( (1/M) sum_m [ E(t_m) - E(0) ]^2 )

reported both absolutely and relative to |E(0)|, beside max|E - E0| (the
dEmax the propagator already tracks) and the same RMS measure for the norm.

Reads <sweep>/<seed_mode>/eps_<eps>/<dynamics>/energy.txt, which carries
t, energy, energy-E0, norm, norm-norm0 at every checked step.

    python energy_rmse.py [sweep_dir ...]      # default: all three sweeps

Writes <sweep>/energy_rmse.txt per sweep and a combined sweep_energy_rmse.txt.
"""
import os
import sys

import numpy as np

def discover():
    """Every directory whose name contains 'sweep' and that holds energy.txt."""
    out = []
    for d in sorted(os.listdir(".")):
        if os.path.isdir(d) and "sweep" in d:
            for root, _, files in os.walk(d):
                if "energy.txt" in files:
                    out.append(d)
                    break
    return out
HEADER = ("# %-10s %-18s %-12s %-22s %8s %14s %14s %14s %14s %14s"
          % ("sweep", "seed_mode", "eps", "dynamics", "M", "E0",
             "RMSE_E", "RMSE_E/|E0|", "max|E-E0|", "RMSE_norm"))


def rms_rows(sweep):
    """One dict per run folder under `sweep`, sorted by mode/eps/dynamics."""
    rows = []
    for mode in sorted(os.listdir(sweep)):
        mdir = os.path.join(sweep, mode)
        if not os.path.isdir(mdir):
            continue
        for epsd in sorted(os.listdir(mdir)):
            edir = os.path.join(mdir, epsd)
            if not os.path.isdir(edir):
                continue
            for dyn in sorted(os.listdir(edir)):
                path = os.path.join(edir, dyn, "energy.txt")
                if not os.path.isfile(path):
                    continue
                d = np.loadtxt(path)
                if d.ndim == 1:
                    d = d[None, :]
                t, E, dE, nrm, dN = (d[:, i] for i in range(5))
                E0 = E[0]
                rows.append(dict(
                    sweep=os.path.basename(sweep.rstrip("/")), seed_mode=mode,
                    eps=epsd.replace("eps_", ""), dynamics=dyn, M=len(t), E0=E0,
                    rmse=float(np.sqrt(np.mean(dE ** 2))),
                    rel=float(np.sqrt(np.mean(dE ** 2)) / abs(E0)) if E0 else float("nan"),
                    dmax=float(np.max(np.abs(dE))),
                    rmse_n=float(np.sqrt(np.mean(dN ** 2)))))
    return rows


def fmt(r):
    return ("  %-10s %-18s %-12s %-22s %8d %14.8e %14.6e %14.6e %14.6e %14.6e"
            % (r["sweep"], r["seed_mode"], r["eps"], r["dynamics"], r["M"],
               r["E0"], r["rmse"], r["rel"], r["dmax"], r["rmse_n"]))


def main(argv):
    sweeps = argv[1:] or discover()
    allrows = []
    for sweep in sweeps:
        if not os.path.isdir(sweep):
            print("skipping %s (not found)" % sweep)
            continue
        rows = rms_rows(sweep)
        if not rows:
            print("skipping %s (no energy.txt found)" % sweep)
            continue
        out = os.path.join(sweep, "energy_rmse.txt")
        with open(out, "w") as fh:
            fh.write("# RMS energy error per run: "
                     "RMSE_E = sqrt(mean_m [E(t_m) - E(0)]^2), atomic units\n")
            fh.write(HEADER + "\n")
            fh.write("\n".join(fmt(r) for r in rows) + "\n")
        print("wrote %s  (%d runs)" % (out, len(rows)))
        allrows += rows
    if allrows:
        with open("sweep_energy_rmse.txt", "w") as fh:
            fh.write("# RMS energy error per run, all sweeps: "
                     "RMSE_E = sqrt(mean_m [E(t_m) - E(0)]^2), atomic units\n")
            fh.write(HEADER + "\n")
            fh.write("\n".join(fmt(r) for r in allrows) + "\n")
        print("wrote sweep_energy_rmse.txt  (%d runs total)" % len(allrows))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
