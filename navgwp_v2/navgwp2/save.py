"""
io.py -- writing results to disk.

write_columns() produces the plain-text files

    p.txt      t   p_1..p_N (per mode)
    q.txt      t   q_1..q_N (per mode)
    a.txt      t   Re zeta / Im zeta (the widths)
    coeff.txt  t   |c_i|^2 for each state
    wfc.txt    wavepacket snapshots: blocks of  x  Re psi_i  Im psi_i ... |psi|^2

plus pickle and CSV helpers.
"""

import csv
import os
import pickle

import numpy as np

from .compute import reconstruct

__all__ = ["write_columns", "save_pickle", "load_pickle", "save_csv"]


def _hdr(names):
    return "  ".join("%-14s" % n for n in names)


def write_columns(result, outdir="output", xgrid=None, n_snapshots=25,
                  margin=6.0, npts=601, mode=0):
    """Write p/q/a/coeff/wfc text files for one trajectory.

    mode : which nuclear mode to tabulate in p.txt/q.txt/a.txt (default 0);
           use mode=None to write every mode (columns grow as N*D).
    """
    os.makedirs(outdir, exist_ok=True)
    t = result["t"]
    N, D = result["N"], result["D"]
    ks = range(D) if mode is None else [mode]

    def path(n):
        return os.path.join(outdir, n)

    qcols = [result["q"][:, i, k] for i in range(N) for k in ks]
    pcols = [result["p"][:, i, k] for i in range(N) for k in ks]
    znames = ["Re_z%d_%d" % (i + 1, k) for i in range(N) for k in ks] + \
             ["Im_z%d_%d" % (i + 1, k) for i in range(N) for k in ks]
    zcols = [np.real(result["zeta"][:, i, k]) for i in range(N) for k in ks] + \
            [np.imag(result["zeta"][:, i, k]) for i in range(N) for k in ks]

    np.savetxt(path("q.txt"), np.column_stack([t] + qcols), fmt="%.8e",
               header=_hdr(["t"] + ["q%d_%d" % (i + 1, k)
                                    for i in range(N) for k in ks]))
    np.savetxt(path("p.txt"), np.column_stack([t] + pcols), fmt="%.8e",
               header=_hdr(["t"] + ["p%d_%d" % (i + 1, k)
                                    for i in range(N) for k in ks]))
    np.savetxt(path("a.txt"), np.column_stack([t] + zcols), fmt="%.8e",
               header=_hdr(["t"] + znames))
    np.savetxt(path("coeff.txt"),
               np.column_stack([t] + [result["pops"][:, i] for i in range(N)]),
               fmt="%.8e",
               header=_hdr(["t"] + ["|c%d|^2" % (i + 1) for i in range(N)]))

    # ---- wavepacket snapshots (only meaningful for D == 1) ----
    if D == 1:
        if xgrid is None:
            lo = result["q"][:, :, 0].min() - margin
            hi = result["q"][:, :, 0].max() + margin
            xgrid = np.linspace(lo, hi, npts)
        idx = np.unique(np.linspace(0, len(t) - 1, n_snapshots).astype(int))
        with open(path("wfc.txt"), "w") as f:
            f.write("# na-VGWP wavepacket snapshots\n")
            f.write("# columns: x " + " ".join(
                "Re(psi%d) Im(psi%d)" % (i + 1, i + 1) for i in range(N))
                + " |psi|^2\n# blocks separated by a blank line\n")
            for s in idx:
                psis = reconstruct(result, s, xgrid)
                dens = sum(np.abs(p) ** 2 for p in psis)
                f.write("# t = %.6e\n" % t[s])
                block = [xgrid]
                for p in psis:
                    block += [p.real, p.imag]
                block.append(dens)
                np.savetxt(f, np.column_stack(block), fmt="%.8e")
                f.write("\n")
    return outdir


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_csv(rows, header, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path
