"""plotting.py -- figures.  Each function returns the matplotlib Figure."""

import numpy as np
import matplotlib.pyplot as plt

from .compute import density, reconstruct, gauss_pair

__all__ = ["plot_potentials", "plot_populations", "plot_widths",
           "plot_conservation", "plot_density", "plot_density_grid",
           "plot_density_2d", "mode_density",
           "plot_summary", "plot_couplings", "plot_surface_2d",
           "plot_real_width", "plot_coupling_expectation",
           "plot_energy_momentum_norm", "plot_coupling_element"]


def _fin(fig, fname):
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=130, bbox_inches="tight")
    return fig


def _sweep_points(x, D, mode, other):
    """(M,D) grid of coordinates sweeping `mode` over `x`, other dims held
    fixed at `other` (default: all zero)."""
    base = np.zeros(D) if other is None else np.asarray(other, float)
    pts = np.broadcast_to(base, (len(x), D)).copy()
    pts[:, mode] = x
    return pts


def potential_curves(model, x, mode=0, other=None):
    """Diabatic diagonal potentials along one mode, other dims held fixed at
    `other` (default: all zero), shape (M,N)."""
    N, D = model["N"], model["D"]
    out = np.zeros((len(x), N))
    if model["diag_kind"] == "quadratic":
        pts = _sweep_points(x, D, mode, other)
        for i in range(N):
            out[:, i] = (model["E"][i] + pts @ model["kappa"][i]
                        + 0.5 * np.sum(model["omega2"][i] * pts ** 2, axis=1))
    else:
        for i in range(N):
            out[:, i] = model["vfun"][i](mode, x)
    return out


def plot_potentials(model, xlim=(-10, 10), npts=500, mode=0, other=None,
                    adiabatic=True, fname=None):
    """`other` fixes the non-swept dimensions (default: all zero) -- e.g. for
    a 2-D conical-intersection model, plot_potentials(model, mode=1,
    other=[X3, 0]) cuts along Y through the coupling centre X=X3 instead of
    through the (physically arbitrary) X=0."""
    x = np.linspace(*xlim, npts)
    D = model["D"]
    V = potential_curves(model, x, mode, other)
    N = model["N"]
    pts = _sweep_points(x, D, mode, other)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(N):
        ax.plot(x, V[:, i], "C%d" % (i % 10), lw=2, label="diabatic %d" % (i + 1))
    if adiabatic:
        Vm = np.zeros((len(x), N, N))
        for i in range(N):
            Vm[:, i, i] = V[:, i]
        g = model.get("gauss")
        gl = model.get("gauss_lin")
        qc = model.get("quad_coupling")
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                c = model["J"][i, j] + pts @ model["lam"][i, j]
                if g is not None:
                    beta_, xc_ = gauss_pair(g, i, j)
                    Aij = np.asarray(g["A"], float)[i, j]
                    c = c + Aij * np.exp(-np.sum(beta_ * (pts - xc_) ** 2, axis=1))
                if gl is not None:
                    beta_ = np.asarray(gl["beta"], float)
                    xc_ = np.asarray(gl["xc"], float)
                    L = gl["lin_dim"]
                    Aij = np.asarray(gl["A"], float)[i, j]
                    c = c + Aij * (pts[:, L] - xc_[L]) * np.exp(
                        -np.sum(beta_ * (pts - xc_) ** 2, axis=1))
                if qc is not None and {i, j} == {qc["i"], qc["j"]}:
                    c = c + qc["v"](pts[:, 0])
                Vm[:, i, j] = c
        E = np.linalg.eigvalsh(Vm)
        for k in range(N):
            ax.plot(x, E[:, k], "k:", lw=1.5,
                    label="adiabatic" if k == 0 else None)
    ax.set_xlabel("mode %d (a.u.)" % mode if D > 1 else "x (a.u.)")
    ax.set_ylabel("energy (a.u.)")
    ax.set_title(model.get("name", "")); ax.grid(alpha=.3); ax.legend(fontsize=9)
    return _fin(fig, fname)


def _diabatic_matrix_2d(model, pts):
    """(M,N,N) diabatic Hamiltonian matrix at each of M 2-D points."""
    N = model["N"]
    Vm = np.zeros((pts.shape[0], N, N))
    for i in range(N):
        Vm[:, i, i] = (model["E"][i] + pts @ model["kappa"][i]
                      + 0.5 * np.sum(model["omega2"][i] * pts ** 2, axis=1))
    g = model.get("gauss")
    gl = model.get("gauss_lin")
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            c = model["J"][i, j] + pts @ model["lam"][i, j]
            if g is not None:
                beta_, xc_ = gauss_pair(g, i, j)
                Aij = np.asarray(g["A"], float)[i, j]
                c = c + Aij * np.exp(-np.sum(beta_ * (pts - xc_) ** 2, axis=1))
            if gl is not None:
                beta_ = np.asarray(gl["beta"], float); xc_ = np.asarray(gl["xc"], float)
                L = gl["lin_dim"]; Aij = np.asarray(gl["A"], float)[i, j]
                c = c + Aij * (pts[:, L] - xc_[L]) * np.exp(
                    -np.sum(beta_ * (pts - xc_) ** 2, axis=1))
            Vm[:, i, j] = c
    return Vm


def plot_surface_2d(model, xlim, ylim, npts=150, which="adiabatic",
                    show_gap=True, title=None, fname=None):
    """Contour plots of a D=2 model's surfaces over its (mode-0, mode-1)
    plane -- the natural view for a conical intersection, e.g. the Ferretti
    model: `which="adiabatic"` shows the two adiabatic sheets, and (for
    N=2) an extra panel with the adiabatic gap E2-E1, which pinches to
    exactly zero AT the intersection point -- the diagnostic that
    distinguishes a true crossing from a merely-close avoided one."""
    if model["D"] != 2:
        raise ValueError("plot_surface_2d needs a 2-mode (D=2) model, got D=%d"
                         % model["D"])
    N = model["N"]
    xs = np.linspace(*xlim, npts)
    ys = np.linspace(*ylim, npts)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)

    Vm = _diabatic_matrix_2d(model, pts)
    surf = Vm[:, np.arange(N), np.arange(N)] if which == "diabatic" \
        else np.linalg.eigvalsh(Vm)

    show_gap = show_gap and which == "adiabatic" and N == 2
    ncols = N + (1 if show_gap else 0)
    fig, ax = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.5), squeeze=False)
    ax = ax[0]
    for k in range(N):
        Z = surf[:, k].reshape(npts, npts)
        cf = ax[k].contourf(X, Y, Z, levels=30, cmap="viridis")
        ax[k].contour(X, Y, Z, levels=12, colors="k", linewidths=.4, alpha=.5)
        fig.colorbar(cf, ax=ax[k], shrink=.85, label="energy (a.u.)")
        ax[k].set_xlabel("mode 0"); ax[k].set_ylabel("mode 1")
        ax[k].set_title("%s %d" % (which, k + 1))

    if show_gap:
        gap = (surf[:, 1] - surf[:, 0]).reshape(npts, npts)
        cf = ax[N].contourf(X, Y, gap, levels=30, cmap="magma")
        fig.colorbar(cf, ax=ax[N], shrink=.85, label=r"$E_2-E_1$ (a.u.)")
        ax[N].set_xlabel("mode 0"); ax[N].set_ylabel("mode 1")
        ax[N].set_title(r"adiabatic gap $E_2-E_1$")

    if title:
        fig.suptitle(title)
    return _fin(fig, fname)


def plot_populations(result, exact=None, t_scale=1.0, t_label="time (a.u.)",
                     names=None, title=None, fname=None):
    t = result["t"] * t_scale
    P = result["pops"]
    N = P.shape[1]
    names = names or ["state %d" % (i + 1) for i in range(N)]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for i in range(N):
        c = "C%d" % (i % 10)
        if exact is not None:
            ax.plot(t, exact[:, i], "-", color=c, lw=2, label=names[i] + " (exact)")
            ax.plot(t, P[:, i], "o", color=c, ms=3,
                    markevery=max(1, len(t) // 40), label=names[i] + " (na-VGWP)")
        else:
            ax.plot(t, P[:, i], "-", color=c, lw=2, label=names[i])
    ax.set_xlabel(t_label); ax.set_ylabel("population"); ax.set_ylim(-.02, 1.02)
    ax.grid(alpha=.3); ax.legend(fontsize=9, ncol=min(N, 3))
    if title:
        ax.set_title(title)
    return _fin(fig, fname)


def plot_widths(results, labels=None, state=0, mode=0, exact=None,
                t_label="time (a.u.)", title=None, fname=None):
    results = results if isinstance(results, (list, tuple)) else [results]
    labels = labels or [r["width_mode"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    if exact is not None:
        ax.plot(results[0]["t"], exact, "k-", lw=3, alpha=.35, label="exact")
    for r, lab in zip(results, labels):
        ax.plot(r["t"], r["sigma"][:, state, mode], lw=2, label=lab)
    ax.set_xlabel(t_label); ax.set_ylabel(r"width $\sigma$")
    ax.grid(alpha=.3); ax.legend(fontsize=9)
    if title:
        ax.set_title(title)
    return _fin(fig, fname)


def plot_conservation(result, fname=None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(result["t"], result["energy"] - result["E0"], "C2")
    ax[0].set_ylabel(r"$E(t)-E_0$"); ax[0].set_title(
        "energy drift (max %.1e)" % result["dEmax"])
    ax[1].plot(result["t"], result["norm"] - result["norm"][0], "C4")
    ax[1].set_ylabel(r"$N(t)-N_0$"); ax[1].set_title(
        "norm drift (max %.1e)" % result["dNmax"])
    for a in ax:
        a.set_xlabel("time (a.u.)"); a.grid(alpha=.3)
        a.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    return _fin(fig, fname)


def plot_real_width(result, mode=0, names=None, ax=None, fname=None):
    """Real width A = Re Z_kk (k = `mode`) of every state vs time.
    Larger A = narrower packet in x (sigma_x = 1/sqrt(2A)); constant for
    frozen widths."""
    t = result["t"]
    N = result["Z"].shape[1]
    names = names or ["state %d" % (i + 1) for i in range(N)]
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 4))
    for i in range(N):
        ax.plot(t, np.real(result["Z"][:, i, mode, mode]), lw=2,
                label=r"$A_%d$ (%s)" % (i + 1, names[i]))
    ax.set_xlabel("time (a.u.)")
    ax.set_ylabel(r"$A = \mathrm{Re}\,Z$ (mode %d)" % mode)
    ax.set_title("real width (%s)" % result.get("width_mode", ""))
    ax.grid(alpha=.3); ax.legend(fontsize=9)
    return _fin(ax.figure, fname) if own else ax


def plot_coupling_element(result, i=0, j=1, ax=None, fname=None):
    """Real part of the coupling matrix element between the Gaussians,
    <V_c>_r(t) = Re <g_i|V_ij|g_j> (result["W"], no c's)."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(result["t"], result["W"][:, i, j].real, "C0", lw=2)
    ax.set_xlabel("time (a.u.)")
    ax.set_ylabel(r"$\langle V_c \rangle_r$ (a.u.)")
    ax.grid(alpha=.3)
    return _fin(ax.figure, fname) if own else ax


def plot_coupling_expectation(result, ax=None, fname=None):
    """Coupling expectation value <V_c>(t) = sum_{i != j} c_i^* c_j W_ij,
    as recorded by compute.run (result["coupling"]), and -- when the run
    recorded it -- Re<g_1|V_c|g_2> (the same element without the c's)."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(result["t"], result["coupling"], "C3", lw=2,
            label=r"$\langle V_c \rangle = \sum_{i\neq j} c_i^* c_j \langle g_i|V_c|g_j\rangle$")
    if "W" in result and result["W"].shape[1] > 1:
        ax.plot(result["t"], result["W"][:, 0, 1].real, "C0", lw=1.5, ls="--",
                label=r"Re$\langle g_1|V_c|g_2\rangle$")
    ax.set_xlabel("time (a.u.)")
    ax.set_ylabel("coupling (a.u.)")
    ax.set_title("coupling")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    return _fin(ax.figure, fname) if own else ax


def plot_summary(result, mode=0, names=None, title=None, fname=None,
                 exact=None):
    """Four-panel diagnostic: centre positions, populations (+ total norm),
    real width A = Re Z of each state, and <V_c>_r = Re <g_1|V_c|g_2>.
    The energy and norm drift are given in the figure title.
    exact : optional exact (SOFT) reference, a dict with "t" and "pops"
            (T,N) as returned by compute.propagate_exact(_grid); drawn as
            black dashed lines in the population panel."""
    t = result["t"]
    N = result["q"].shape[1]
    names = names or ["state %d" % (i + 1) for i in range(N)]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    for i in range(N):
        ax[0, 0].plot(t, result["q"][:, i, mode], lw=2,
                      label=r"$q_%d$ (%s)" % (i + 1, names[i]))
    ax[0, 0].set_xlabel("time (a.u.)"); ax[0, 0].set_ylabel("centre position (a.u.)")
    ax[0, 0].grid(alpha=.3); ax[0, 0].legend(fontsize=9)

    for i in range(N):
        ax[0, 1].plot(t, result["pops"][:, i], lw=2,
                      label=r"$|c_%d|^2$ (%s)" % (i + 1, names[i]))
    if exact is not None:
        for i in range(N):
            ax[0, 1].plot(exact["t"], exact["pops"][:, i], "--", color="C%d" % i,
                          lw=1.4, alpha=0.9, label=r"$P_%d$ exact (SOFT)" % (i + 1))
    ax[0, 1].plot(t, result["pops"].sum(axis=1), "k:", lw=1.2, label="total norm")
    ax[0, 1].set_xlabel("time (a.u.)"); ax[0, 1].set_ylabel("population")
    ax[0, 1].grid(alpha=.3); ax[0, 1].legend(fontsize=9)

    plot_real_width(result, mode=mode, names=names, ax=ax[1, 0])
    plot_coupling_element(result, ax=ax[1, 1])

    drift = "energy drift %.1e, norm drift %.1e" % (result["dEmax"], result["dNmax"])
    fig.suptitle((title + "\n" if title else "") + drift)
    return _fin(fig, fname)


def plot_energy_momentum_norm(result, mode=0, names=None, title=None,
                              fname=None):
    """Three panels: energy drift E(t) - E0, momentum p_i(t) of every state
    (mode `mode`), and norm drift N(t) - N(0)."""
    t = result["t"]
    N = result["p"].shape[1]
    names = names or ["state %d" % (i + 1) for i in range(N)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    ax[0].plot(t, result["energy"] - result["E0"], "C2", lw=1.8)
    ax[0].set_ylabel(r"$E(t)-E_0$ (a.u.)")
    ax[0].set_title("energy drift (max %.1e)" % result["dEmax"])
    ax[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    for i in range(N):
        ax[1].plot(t, result["p"][:, i, mode], lw=2,
                   label=r"$p_%d$ (%s)" % (i + 1, names[i]))
    ax[1].set_ylabel("momentum p (a.u.)")
    ax[1].set_title("momentum (mode %d)" % mode)
    ax[1].legend(fontsize=9)

    ax[2].plot(t, result["norm"] - result["norm"][0], "C4", lw=1.8)
    ax[2].set_ylabel(r"$N(t)-N(0)$")
    ax[2].set_title("norm drift (max %.1e)" % result["dNmax"])
    ax[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    for a in ax:
        a.set_xlabel("time (a.u.)"); a.grid(alpha=.3)
    if title:
        fig.suptitle(title)
    return _fin(fig, fname)


def plot_couplings(model, names=None, title=None, fname=None):
    """Site energies (diagonal) and the constant electronic coupling matrix
    J (off-diagonal) -- the natural view for a multi-bath / multi-site model
    (e.g. FMO) where a 1D nuclear-coordinate cut of the PES isn't
    informative, since each site's diagonal potential only depends on its
    own, separate bath modes."""
    N = model["N"]
    names = names or ["state %d" % (i + 1) for i in range(N)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

    E = model.get("E")
    if E is not None:
        ax[0].bar(range(N), np.asarray(E, float), color="C0")
        ax[0].set_xticks(range(N)); ax[0].set_xticklabels(names, rotation=30, ha="right")
        ax[0].set_ylabel("site energy (a.u.)")
        ax[0].grid(alpha=.3, axis="y")
    else:
        ax[0].axis("off")
        ax[0].text(0.5, 0.5, "no fixed diagonal energies\n(separable diag_kind)",
                  ha="center", va="center", transform=ax[0].transAxes)
    ax[0].set_title("site energies")

    J = np.asarray(model["J"], float)
    scale = np.max(np.abs(J)) if np.any(J) else 1.0
    im = ax[1].imshow(J, cmap="coolwarm", vmin=-scale, vmax=scale)
    ax[1].set_xticks(range(N)); ax[1].set_xticklabels(names, rotation=30, ha="right")
    ax[1].set_yticks(range(N)); ax[1].set_yticklabels(names)
    for i in range(N):
        for j in range(N):
            ax[1].text(j, i, "%.2g" % J[i, j], ha="center", va="center", fontsize=8,
                      color="white" if abs(J[i, j]) > 0.5 * scale else "black")
    fig.colorbar(im, ax=ax[1], shrink=0.85, label="coupling J (a.u.)")
    ax[1].set_title("electronic coupling matrix")

    if title:
        fig.suptitle(title)
    return _fin(fig, fname)


def mode_density(result, step, state, mode, x):
    """Reduced density of one state along one mode, all other modes
    integrated out:  |c_i|^2 sqrt(a/pi) exp[-a (x - q)^2],  a = Re zeta.

    Exact for the product ansatz (each other mode's |g|^2 integrates to 1),
    and for D=1 identical to |psi_i(x)|^2 from `reconstruct`."""
    a = np.real(result["zeta"][step, state, mode])
    q = result["q"][step, state, mode]
    return (result["pops"][step, state] * np.sqrt(a / np.pi)
            * np.exp(-a * (np.asarray(x) - q) ** 2))


def _auto_range(result, snaps, states, mode, nsig=4.0):
    """x-range covering q +- nsig*sigma of every plotted state and snapshot."""
    q = result["q"][np.ix_(snaps, states)][..., mode]
    sig = result["sigma"][np.ix_(snaps, states)][..., mode]
    return float(np.min(q - nsig * sig)), float(np.max(q + nsig * sig))


def plot_density_grid(result, params):
    """Grid of snapshots, one panel per time step in params["snaps"], each
    showing the PER-STATE density (not summed) along one nuclear mode.

    For D=1 this is |psi_i(x)|^2.  For D>1 it is the reduced density along
    `mode` with every other mode integrated out (see `mode_density`), which
    is exact for the product-Gaussian ansatz.

    params keys:
        prefix        figure title
        states        list of (0-indexed) state indices to draw
        1Dcolors      list of colors, one per entry in `states`
        size          [width, height] in inches
        legend_loc    matplotlib legend `loc`
        legend_size   legend fontsize
        mode          nuclear mode to plot along (default 0)
        xmin, xmax    grid range (default: automatic, q +- 4 sigma)
        npoints       number of grid points
        snaps         list of time-step indices, one subplot each
        t_scale       multiply times by this for the panel titles (default 1),
                      e.g. initialize.FS for femtoseconds
        t_label       time unit for the panel titles (default "a.u.")
        ncols         panels per row (default 2)
        fname         optional path to save the figure
    """
    states = list(params.get("states", range(result["N"])))
    colors = params.get("1Dcolors", ["k", "b"] + ["C%d" % i for i in range(2, 10)])
    snaps = list(params["snaps"])
    mode = params.get("mode", 0)
    lo, hi = _auto_range(result, snaps, states, mode)
    x = np.linspace(params.get("xmin", lo), params.get("xmax", hi),
                    params.get("npoints", 400))
    t_scale, t_label = params.get("t_scale", 1.0), params.get("t_label", "a.u.")
    ncols = params.get("ncols", 2)
    nrows = int(np.ceil(len(snaps) / ncols))
    size = params.get("size", [5.5 * ncols, 4 * nrows])
    fig, ax = plt.subplots(nrows, ncols, figsize=size, squeeze=False)

    xlabel = "x (a.u.)" if result["D"] == 1 else "mode %d (a.u.)" % mode
    ylabel = (r"$|\psi|^2$" if result["D"] == 1
              else r"$\int |\psi_i|^2\,d(\mathrm{other\ modes})$")
    for idx, s in enumerate(snaps):
        a = ax[idx // ncols][idx % ncols]
        for j, i in enumerate(states):
            a.plot(x, mode_density(result, s, i, mode, x),
                   color=colors[j % len(colors)], lw=2, label="state %d" % i)
        a.set_xlabel(xlabel); a.set_ylabel(ylabel)
        a.grid(alpha=.3)
        a.legend(loc=params.get("legend_loc", "best"), fontsize=params.get("legend_size", 9))
        a.set_title("t = %.2f %s" % (result["t"][s] * t_scale, t_label), fontsize=10)

    for idx in range(len(snaps), nrows * ncols):
        ax[idx // ncols][idx % ncols].axis("off")

    if params.get("prefix"):
        fig.suptitle(params["prefix"])
    return _fin(fig, params.get("fname"))


def plot_density_2d(result, params):
    """Contour snapshots of each state's density |c_i g_i(x0, x1)|^2 over
    the plane of two modes (rows = states, columns = snapshots).  For D>2
    the remaining modes are integrated out.

    params keys:
        prefix        figure title
        states        state indices (rows), default all
        snaps         time-step indices (columns)
        modes         the two modes spanning the plane (default [0, 1])
        xlim, ylim    plot ranges (default: automatic, q +- 4 sigma)
        npoints       grid points per axis (default 150)
        panel_size    inches per panel (default 3.2)
        fname         optional path to save the figure
    """
    states = list(params.get("states", range(result["N"])))
    snaps = list(params["snaps"])
    m0, m1 = params.get("modes", [0, 1])
    n = params.get("npoints", 150)
    xs = np.linspace(*params.get("xlim", _auto_range(result, snaps, states, m0)), n)
    ys = np.linspace(*params.get("ylim", _auto_range(result, snaps, states, m1)), n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    ps = params.get("panel_size", 3.2)
    fig, ax = plt.subplots(len(states), len(snaps), squeeze=False,
                           figsize=(ps * len(snaps) + 1.0, ps * len(states)),
                           constrained_layout=True)
    # shared colour scale per state, so population transfer is visible
    for r, i in enumerate(states):
        rhos = []
        for s in snaps:
            rx = mode_density(result, s, i, m0, xs)
            ry = mode_density(result, s, i, m1, ys) / max(result["pops"][s, i], 1e-300)
            rhos.append(rx[:, None] * ry[None, :])
        vmax = max(float(r_.max()) for r_ in rhos) or 1.0
        for c, s in enumerate(snaps):
            a = ax[r][c]
            cf = a.contourf(X, Y, rhos[c], levels=np.linspace(0, vmax, 16),
                            cmap="viridis")
            a.set_title("state %d   t = %.1f" % (i, result["t"][s]), fontsize=9)
            a.set_xlabel("mode %d" % m0, fontsize=8)
            a.set_ylabel("mode %d" % m1, fontsize=8)
            a.tick_params(labelsize=7)
        fig.colorbar(cf, ax=ax[r].tolist(), shrink=.85, label=r"$|\psi_%d|^2$" % i)
    if params.get("prefix"):
        fig.suptitle(params["prefix"])
    if params.get("fname"):
        fig.savefig(params["fname"], dpi=130, bbox_inches="tight")
    return fig


def plot_density(result, steps, x, title=None, fname=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in steps:
        ax.plot(x, density(result, s, x), lw=2, label="t = %.0f" % result["t"][s])
    ax.set_xlabel("x (a.u.)"); ax.set_ylabel(r"$|\psi|^2$")
    ax.grid(alpha=.3); ax.legend(fontsize=9)
    if title:
        ax.set_title(title)
    return _fin(fig, fname)
