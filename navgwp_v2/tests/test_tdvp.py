"""The equations of motion ARE the Dirac-Frenkel (TDVP) equations.

Independent check.  Each component Psi_j = c_j g_j lives on a complex-
analytic manifold (A exp[-x^T Z x/2 + eta.x]), so the exact TDVP derivative
of Psi_j is the orthogonal projection of -i (H Psi)_j onto

    span{ 1, d_k, d_k d_l } g_j        d = x - q_j
    (thawed, full Z;  only d_k^2 for diagonal Z;  only {1, d_k} for frozen).

Here (H Psi)_j is built on a grid (FFT kinetic energy, potential matrix at
the grid points), the projection is done by direct summation, and the
parameter rates are read off:

    Zdot = -4 A ( E[f d d^T] - E[f] cov ) A          cov = A^-1 / 2
    Z qdot + i pdot = 2 A E[f d]
    cdot / c = E[f] + (i/4) tr(A^-1 Bdot) + i p.qdot

with E[f h] = <Psi_j| h (-i H Psi)_j> / |c_j|^2.  Nothing here shares code
with the closed-form matrix elements being tested.  Phases of c are compared
up to the global gauge (a common real shift of Im(cdot_j/c_j)).
"""
import numpy as np

import navgwp2 as ng
from navgwp2 import compute as cm      # the one engine; kept as `cm`
cm.wavefunction_full = cm.wavefunction  # renamed when the engines merged


# --------------------------------------------------------------------------
def _grid_potential(model, X):
    """(..., N, N) potential matrix on grid points X (..., D), any model."""
    N, D = model["N"], model["D"]
    V = np.zeros(X.shape[:-1] + (N, N))
    kind = model["diag_kind"]
    for i in range(N):
        if kind == "separable":
            V[..., i, i] = sum(model["vfun"][i](k, X[..., k]) for k in range(D))
        elif kind == "quadratic":
            V[..., i, i] = (model["E"][i] + X @ model["kappa"][i]
                            + 0.5 * np.sum(model["omega2"][i] * X ** 2, axis=-1))
        else:
            V[..., i, i] = cm.potential_matrix(model, X)[..., i, i]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            v = model["J"][i, j] + X @ model["lam"][i, j]
            g = model.get("gauss")
            if g is not None:
                v = v + np.asarray(g["A"])[i, j] * np.exp(
                    -np.sum(g["beta"] * (X - g["xc"]) ** 2, axis=-1))
            gl = model.get("gauss_lin")
            if gl is not None:
                L = gl["lin_dim"]
                v = v + np.asarray(gl["A"])[i, j] * (X[..., L] - gl["xc"][L]) * np.exp(
                    -np.sum(gl["beta"] * (X - gl["xc"]) ** 2, axis=-1))
            qc = model.get("quad_coupling")
            if qc is not None and {i, j} == {qc["i"], qc["j"]}:
                v = v + qc["v"](X[..., 0])
            V[..., i, j] = v
    return V


def _tdvp_rates(model, q, p, Z, c, axes, width_mode, width_shape):
    """Exact TDVP rates (qdot, pdot, Zdot, cdot) by grid projection."""
    N, D, mass = model["N"], model["D"], model["mass"]
    X = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    dV = np.prod([a[1] - a[0] for a in axes])
    ks = [2 * np.pi * np.fft.fftfreq(len(a), a[1] - a[0]) for a in axes]
    K2m = sum(np.meshgrid(*[k ** 2 / (2 * m) for k, m in zip(ks, mass)],
                          indexing="ij"))
    psi = np.array([c[j] * cm.wavefunction_full(Z[j], q[j], p[j], X)
                    for j in range(N)])
    Hpsi = (np.einsum("...ij,j...->i...", _grid_potential(model, X), psi)
            + np.array([np.fft.ifftn(K2m * np.fft.fftn(ps)) for ps in psi]))
    qd = np.zeros((N, D)); pd = np.zeros((N, D))
    Zd = np.zeros((N, D, D), complex); cd = np.zeros(N, complex)
    for j in range(N):
        n = abs(c[j]) ** 2
        w = np.conj(psi[j]) * (-1j) * Hpsi[j] * dV / n     # E[f h] = sum(w h)
        d = X - q[j]
        Ef = np.sum(w)
        Efd = np.array([np.sum(w * d[..., k]) for k in range(D)])
        Efdd = np.array([[np.sum(w * d[..., k] * d[..., l]) for l in range(D)]
                         for k in range(D)])
        A, B = np.real(Z[j]), np.imag(Z[j])
        if width_mode == "thawed":
            Zd[j] = -4.0 * A @ (Efdd - Ef * 0.5 * np.linalg.inv(A)) @ A
            if width_shape == "diagonal":
                Zd[j] = np.diag(np.diag(Zd[j]))
        alpha1 = 2.0 * A @ Efd
        qd[j] = np.linalg.solve(A, alpha1.real)
        pd[j] = alpha1.imag - B @ qd[j]
        cd[j] = c[j] * (Ef + 0.25j * np.trace(np.linalg.solve(A, Zd[j].imag))
                        + 1j * p[j] @ qd[j])
    return qd, pd, Zd, cd


def _compare(ref, got, c):
    qd, pd, Zd, cd = ref
    gq, gp, gZ, gc = got
    r_ref, r_got = cd / c, gc / c
    ph_ref = r_ref.imag - r_ref.imag[0]
    ph_got = r_got.imag - r_got.imag[0]
    scale = lambda a: max(1.0, float(np.max(np.abs(a))))
    return max(np.max(np.abs(gq - qd)) / scale(qd),
               np.max(np.abs(gp - pd)) / scale(pd),
               np.max(np.abs(gZ - Zd)) / scale(Zd),
               np.max(np.abs(r_got.real - r_ref.real)) / scale(r_ref.real),
               np.max(np.abs(ph_got - ph_ref)) / scale(ph_ref))


def _check_compute(model, q, p, zeta, c, axes, tol):
    """The one engine restricted to diagonal widths, against the grid TDVP."""
    N, D = model["N"], model["D"]
    Z = np.array([np.diag(z) for z in zeta])
    _check_matrix(model, q, p, Z, c, axes, "diagonal", tol)


def _check_matrix(model, q, p, Z, c, axes, shape, tol):
    N, D = model["N"], model["D"]
    for mode in ("frozen", "thawed"):
        ref = _tdvp_rates(model, q, p, Z, c, axes, mode, shape)
        y = cm.rhs(cm.pack(q, p, Z, c), model, eps=0.0, width_mode=mode,
                   width_shape=shape)
        err = _compare(ref, cm.unpack(y, N, D), c)
        assert err < tol, (mode, shape, err)


# --------------------------------------------------------------------------
_x1 = [np.linspace(-14, 14, 1024, endpoint=False)]
_x2 = [np.linspace(-9, 9, 160, endpoint=False)] * 2


def _lvc_1d_three_state():
    lam = np.zeros((3, 3, 1)); lam[0, 1] = lam[1, 0] = 0.02; lam[1, 2] = lam[2, 1] = -0.03
    return ng.lvc([0.0, 0.01, -0.02], [[0.0], [0.03], [-0.02]],
                  [[0.05], [0.08], [0.03]],
                  [[0, 0.01, 0.004], [0.01, 0, 0.006], [0.004, 0.006, 0]],
                  lam=lam, mass=np.array([1.3]),
                  gauss=dict(A=0.02 * (1 - np.eye(3)), beta=np.array([0.4]),
                             xc=np.array([0.3])))


_st1 = dict(q=np.array([[0.5], [-1.0], [0.2]]), p=np.array([[0.3], [-0.2], [0.1]]),
            zeta=np.array([[0.9 + 0.4j], [0.6 - 0.3j], [1.2 + 0.1j]]),
            c=np.array([0.7, 0.5j, 0.3 - 0.4j]))


def test_tdvp_1d_three_states_linear_and_gaussian_coupling():
    m = _lvc_1d_three_state()
    _check_compute(m, _st1["q"], _st1["p"], _st1["zeta"], _st1["c"], _x1, 1e-9)
    Z = np.array([np.diag(z) for z in _st1["zeta"]])
    _check_matrix(m, _st1["q"], _st1["p"], Z, _st1["c"], _x1, "full", 1e-9)


def test_tdvp_2d_diagonal_widths_all_analytic_couplings():
    """D=2, gauss + gauss_lin + linear couplings, diagonal widths."""
    lam = np.zeros((2, 2, 2)); lam[0, 1] = lam[1, 0] = [0.01, -0.02]
    m = ng.lvc([0.0, 0.02], [[0.0, 0.0], [0.03, -0.02]],
               [[0.3, 0.2], [0.25, 0.4]], [[0, 0.01], [0.01, 0]], lam=lam,
               mass=np.array([1.0, 1.5]),
               gauss=dict(A=np.array([[0, 0.03], [0.03, 0]]),
                          beta=np.array([0.3, 0.5]), xc=np.array([0.2, -0.1])),
               gauss_lin=dict(A=np.array([[0, 0.05], [0.05, 0]]),
                              beta=np.array([0.4, 0.3]), xc=np.array([0.1, 0.0]),
                              lin_dim=1))
    q = np.array([[0.3, -0.2], [-0.5, 0.4]]); p = np.array([[0.2, 0.1], [-0.1, 0.3]])
    zeta = np.array([[1.0 + 0.3j, 0.8 - 0.2j], [0.7 + 0.1j, 1.2 + 0.4j]])
    c = np.array([0.8, 0.6j])
    _check_compute(m, q, p, zeta, c, _x2, 1e-8)


def test_tdvp_2d_full_matrix_width_rotated_surfaces():
    """D=2, non-diagonal Hessians, non-diagonal complex Z, every analytic
    coupling type -- the full-matrix equations."""
    th = 0.6
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    lam = np.zeros((2, 2, 2)); lam[0, 1] = lam[1, 0] = [0.01, -0.02]
    m = cm.lvc_full([0.0, 0.02], [[0.0, 0.0], [0.03, -0.02]],
                    [np.diag([0.3, 0.2]), R @ np.diag([0.5, 0.15]) @ R.T],
                    [[0, 0.01], [0.01, 0]], lam=lam, mass=np.array([1.0, 1.5]),
                    gauss=dict(A=np.array([[0, 0.03], [0.03, 0]]),
                               beta=np.array([0.3, 0.5]), xc=np.array([0.2, -0.1])),
                    gauss_lin=dict(A=np.array([[0, 0.05], [0.05, 0]]),
                                   beta=np.array([0.4, 0.3]), xc=np.array([0.1, 0.0]),
                                   lin_dim=1))
    q = np.array([[0.3, -0.2], [-0.5, 0.4]]); p = np.array([[0.2, 0.1], [-0.1, 0.3]])
    Z = np.array([[[1.0 + 0.3j, 0.2 - 0.1j], [0.2 - 0.1j, 0.8 - 0.2j]],
                  [[0.7 + 0.1j, -0.15 + 0.2j], [-0.15 + 0.2j, 1.2 + 0.4j]]])
    c = np.array([0.8, 0.6j])
    _check_matrix(m, q, p, Z, c, _x2, "full", 1e-8)
    Zd = np.array([np.diag(np.diag(z)) for z in Z])
    _check_matrix(m, q, p, Zd, c, _x2, "diagonal", 1e-8)


def test_tdvp_separable_diagonal_2d():
    """Separable non-polynomial diagonal in D=2 (Gauss-Hermite energies and
    forces, finite-difference curvature), linear + Gaussian couplings.

    A smooth potential is used on purpose: Tully I's V' has a kink at x=0,
    where 48-point Gauss-Hermite converges only slowly (~1e-4 relative) --
    a quadrature limit, not an equation-of-motion one."""
    a = np.array([[0.02, 0.01], [-0.015, 0.03]]); b = np.array([0.8, 1.2])
    w2 = np.array([[0.2, 0.3], [0.25, 0.15]])
    vf = [lambda k, x, i=i: a[i, k] * np.tanh(b[k] * x) + 0.5 * w2[i, k] * x ** 2
          for i in range(2)]
    dvf = [lambda k, x, i=i: a[i, k] * b[k] / np.cosh(b[k] * x) ** 2 + w2[i, k] * x
           for i in range(2)]
    lam = np.zeros((2, 2, 2)); lam[0, 1] = lam[1, 0] = [0.01, 0.02]
    m = ng.separable(vf, dvf, N=2, D=2, J=[[0, 0.01], [0.01, 0]], lam=lam,
                     mass=np.array([1.0, 1.4]),
                     gauss=dict(A=np.array([[0, 0.03], [0.03, 0]]),
                                beta=np.array([0.3, 0.5]), xc=np.array([0.2, -0.1])))
    q = np.array([[0.3, -0.2], [-0.5, 0.4]]); p = np.array([[0.2, 0.1], [-0.1, 0.3]])
    zeta = np.array([[1.0 + 0.3j, 0.8 - 0.2j], [0.7 + 0.1j, 1.2 + 0.4j]])
    _check_compute(m, q, p, zeta, np.array([0.8, 0.6j]), _x2, 1e-6)


def test_tdvp_quadrature_coupling_tully3():
    m = ng.tully3(npts=4000)
    q = np.array([[0.4], [-0.3]]); p = np.array([[10.0], [12.0]])
    zeta = np.array([[1.0 + 0.3j], [0.8 - 0.2j]]); c = np.array([0.8, 0.6j])
    _check_compute(m, q, p, zeta, c, [np.linspace(-12, 12, 4096, endpoint=False)], 1e-5)


def test_thawed_energy_and_norm_conserved():
    """Exact TDVP on a complex-analytic manifold conserves <H> and the norm;
    the residual drift is RK4 truncation, so it must shrink ~dt^4."""
    m = _lvc_1d_three_state()
    kw = dict(q0=_st1["q"], p0=_st1["p"], zeta0=_st1["zeta"], t_max=40.0,
              record_every=50, width_mode="thawed", seed=None, eps=0.0)
    r1 = ng.run(m, _st1["c"], dt=0.02, **kw)
    r2 = ng.run(m, _st1["c"], dt=0.01, **kw)
    assert r2["dEmax"] < 1e-9 and r2["dNmax"] < 1e-9
    assert r1["dEmax"] / max(r2["dEmax"], 1e-300) > 8.0      # ~16 for RK4
