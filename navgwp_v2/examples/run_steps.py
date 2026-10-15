"""RK4 step count and propagated time of every run in the seed sweeps.

Each run folder's summary.txt records dt and t_max, so

    RK4 steps   = round(t_max / dt)          (4 rhs evaluations each)
    rhs calls   = 4 * steps
    time        = t_max                      atomic units
                = t_max * FS                 femtoseconds (FS = 0.02418884254)

Wall-clock is not stored by the sweeps, so with --calibrate the script times a
short propagation of the matching notebook case for each distinct
(model, width_mode, drop_corrections) and multiplies by the step count.  That
column is an estimate, marked est_wall_s; everything else is read from file.

    python run_steps.py                 # steps and propagated time
    python run_steps.py --calibrate     # adds the wall-clock estimate

Writes <sweep>/run_steps.txt per sweep and a combined sweep_run_steps.txt.
"""
import os
import sys
import time

import numpy as np

FS = 0.02418884254                      # a.u. of time -> fs
CASE_OF = {"holstein": 3, "tully": 0, "coronado": 8}


def discover():
    out = []
    for d in sorted(os.listdir(".")):
        if os.path.isdir(d) and "sweep" in d:
            for _root, _dirs, files in os.walk(d):
                if "summary.txt" in files:
                    out.append(d)
                    break
    return out


def read_summary(path):
    """summary.txt -> dict of its 'key   value' lines (first token is the key)."""
    out = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        for key in ("dt", "t_max", "frames", "width_mode", "drop_corr",
                    "seed_mode", "states", "modes"):
            if line.startswith(key) or (key in ("states", "modes") and parts[0] == key):
                out[key] = parts[-1]
    return out


def rows_of(sweep):
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
                path = os.path.join(edir, dyn, "summary.txt")
                if not os.path.isfile(path):
                    continue
                s = read_summary(path)
                dt, tmax = float(s["dt"]), float(s["t_max"])
                rows.append(dict(sweep=sweep, seed_mode=mode,
                                 eps=epsd.replace("eps_", ""), dynamics=dyn,
                                 N=int(s["states"]), D=int(s["modes"]),
                                 width_mode=s["width_mode"],
                                 drop=int(s["drop_corr"]), dt=dt, t_max=tmax,
                                 steps=int(round(tmax / dt)),
                                 frames=int(s["frames"])))
    return rows


def calibrate(rows, nsteps=200):
    """Measure us/step for each distinct (sweep family, width_mode, drop)."""
    sys.path.insert(0, ".."), sys.path.insert(0, ".")
    import navgwp2 as ng
    import model as mdl
    cost = {}
    for r in rows:
        fam = next((k for k in CASE_OF if k in r["sweep"]), None)
        key = (fam, r["width_mode"], r["drop"])
        if fam is None or key in cost:
            continue
        case = CASE_OF[fam]
        mp = mdl.model_params(case)
        m = mdl.build_model(mp)
        dyn = dict(mdl.dyn_params(case, mp), width_mode=r["width_mode"],
                   drop_corrections=r["drop"], dt=r["dt"],
                   t_max=nsteps * r["dt"], dt_out=None, record_every=nsteps)
        t0 = time.time()
        ng.run(m, **dyn)
        cost[key] = (time.time() - t0) / nsteps
    for r in rows:
        fam = next((k for k in CASE_OF if k in r["sweep"]), None)
        r["est_wall_s"] = cost.get((fam, r["width_mode"], r["drop"]), float("nan")) * r["steps"]
    return rows


def write(path, rows, note, timed):
    cols = ["sweep", "seed_mode", "eps", "dynamics", "N", "D", "width_mode",
            "drop", "dt", "steps", "rhs_calls", "t_max_au", "t_max_fs", "frames"]
    if timed:
        cols.append("est_wall_s")
    W = {"sweep": 30, "seed_mode": 11, "eps": 8, "dynamics": 21, "N": 3, "D": 3,
         "width_mode": 9, "drop": 6}
    w = lambda c: W.get(c, 13)
    head = "# " + "".join("%-*s" % (w(c), c) for c in cols)
    def fmt(r):
        v = dict(r, rhs_calls=4 * r["steps"], t_max_au=r["t_max"],
                 t_max_fs=r["t_max"] * FS)
        out = []
        for c in cols:
            x = v[c]
            out.append(("%-*s" % (w(c), x)) if isinstance(x, str)
                       else ("%-*.6g" % (w(c), x)))
        return "  " + "".join(out)
    with open(path, "w") as fh:
        fh.write(note + "\n" + head + "\n" + "\n".join(fmt(r) for r in rows) + "\n")


def main(argv):
    timed = "--calibrate" in argv
    sweeps = [a for a in argv[1:] if not a.startswith("--")] or discover()
    note = ("# RK4 steps and propagated time per run; steps = round(t_max/dt), "
            "4 rhs calls each" + (", est_wall_s estimated from a timed "
                                  "200-step run" if timed else ""))
    allrows = []
    for sweep in sweeps:
        rows = rows_of(sweep)
        if not rows:
            continue
        if timed:
            calibrate(rows)
        write(os.path.join(sweep, "run_steps.txt"), rows, note, timed)
        print("wrote %s/run_steps.txt  (%d runs, %d RK4 steps total)"
              % (sweep, len(rows), sum(r["steps"] for r in rows)))
        allrows += rows
    if allrows:
        write("sweep_run_steps.txt", allrows, note, timed)
        print("wrote sweep_run_steps.txt  (%d runs, %.3g RK4 steps total%s)"
              % (len(allrows), sum(r["steps"] for r in allrows),
                 ", est %.0f s of propagation" % np.nansum([r["est_wall_s"] for r in allrows])
                 if timed else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
