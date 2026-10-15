"""
compute.py -- the physics engine: Gaussian integrals, equations of motion,
propagation, observables, and the numerically-exact references.

ONE Gaussian per electronic state, carrying a FULL D x D complex symmetric
width matrix:

    g(x) = (det Re Z / pi^D)^{1/4}
           exp[ -1/2 (x-q)^T Z (x-q) + i p^T (x-q) ]

Z = A + iB with A positive definite:
    A = Re Z   sets the spatial shape.  The density |g|^2 is a Gaussian with
               covariance Sigma = (2A)^-1, so an off-diagonal A_kl TILTS the
               packet -- modes k and l become correlated.
    B = Im Z   is the chirp (position-dependent local momentum).

There is a single code path.  A frozen Gaussian is the special case where Z
is real and its time derivative is switched off; a packet with one
independent width per mode is the special case where Z stays diagonal
(`width_shape='diagonal'`, which keeps only the diagonal of Zdot).

Equations of motion (hbar = 1), state i, with C_i = sum_{j!=i} c_i^* c_j M_ij:

  qdot_i = M^-1 p_i + (2/|c_i|^2) Im C_i

  pdot_i = -<grad V_ii> - (2/|c_i|^2) ( A_i Re C_i + B_i Im C_i )

  Zdot_i = -i ( Z_i M^-1 Z_i - <Hess V_ii> )
           + (4i/|c_i|^2) A_i [ sum_{j!=i} c_i^* c_j (R_ij - W_ij A_i^-1/2) ] A_i

  i cdot_i = ( <T+V_ii> - p_i.qdot_i - 1/4 tr(A_i^-1 Bdot_i) ) c_i
             + sum_{j!=i} W_ij c_j

with  W_ij = <g_i|V_ij|g_j>
      M_ij = <g_i|(x-q_i) V_ij|g_j>                     (D,)
      R_ij = <g_i|(x-q_i)(x-q_i)^T V_ij|g_j>            (D,D)

These are the Dirac-Frenkel (TDVP) equations for the ansatz: the time
derivative of each component c_i g_i is the orthogonal projection of
-i (H Psi)_i onto span{1, (x-q_i)_k, (x-q_i)_k (x-q_i)_l} g_i.  The first
line of Zdot is Heller's thawed-Gaussian equation on state i's own surface;
the second is the inter-state coupling.  For a diagonal Z in D=1 the second
line reduces to -(2i a/|c|^2) sum c^* c [W - 2a R].

`legacy=True` drops the coupling term of Zdot, the B_i Im C_i term of pdot
and the 1/4 tr(A^-1 Bdot) phase, reproducing an earlier, incomplete version
of the equations.  All three vanish for frozen (real, constant) widths.  It
exists only so the corrections can be demonstrated; never use it otherwise.

Supported potentials
--------------------
Diagonal V_ii, any of:
    diag_kind='quadratic'       E_i + kappa_i.x + 1/2 sum_k omega2_ik x_k^2
    diag_kind='quadratic_full'  E_i + kappa_i.x + 1/2 x^T hess_i x
    diag_kind='separable'       sum_k v_i(k, x_k), any shape, by 1-D
                                Gauss-Hermite on each mode's MARGINAL width
Off-diagonal V_ij, additively:
    J (constant), lam (linear), gauss, gauss_lin  -- all closed form
    quad_coupling -- numerical quadrature, D=1 only

Cost is O(N^2 D^3) per step: one D x D solve per state pair, because the
integrals no longer factorise over modes.  This is the price of letting the
packets tilt; see the note in run().

State vector (real, length 2ND + 2ND^2 + 2N):
      y = [ q(N,D) | p(N,D) | Re Z(N,D,D) | Im Z(N,D,D) | Re c(N) | Im c(N) ]

Plain functions, no classes.
"""

import numpy as np

__all__ = [
    # gaussians
    "pair_moments", "kinetic_expectation", "wavefunction",
    "coherent_Z", "coherent_zeta", "marginal_widths", "state_hessian",
    # models
    "lvc_full", "default_Z", "suggest_dt_full",
    # dynamics
    "pack", "unpack", "diagonal_terms", "coupling_terms", "rhs", "rk4_step",
    "run", "seed_uniform", "seed_coupling", "GH_ORDER",
    # observables
    "total_norm", "total_energy", "populations", "reconstruct", "density",
    "exact_level_populations", "free_width_exact", "breathing_width_exact",
    # exact references
    "grid", "gaussian_on_grid", "propagate_exact",
    "potential_matrix", "propagate_exact_grid", "gauss_pair",
]


# ==========================================================================
#  Gaussian toolbox
# ==========================================================================
# conj(g1) g2 has exponent -x^T Zp x + Yp^T x + Kp with
#
#     Zp = (conj(Z1) + Z2)/2
#     Yp = conj(Z1) q1 + Z2 q2 + i (p2 - p1)
#     Kp = -q1^T conj(Z1) q1/2 - q2^T Z2 q2/2 + i p1.q1 - i p2.q2
#
#     int exp(-x^T Zp x + Yp^T x) dx = sqrt(pi^D / det Zp) exp(Yp^T Zp^-1 Yp/4)
#
# zbar = Zp^-1 Yp / 2,   cov = Zp^-1 / 2   (<x x^T> = zbar zbar^T + cov).
# Every eigenvalue of Zp has positive real part (Re Zp is positive definite),
# so det^{-1/2} = prod lambda^{-1/2} on the principal branch is continuous.

def _norm(Z):
    """Normalisation (det A / pi^D)^{1/4}, A = Re Z, as prod_k (lambda_k/pi)^{1/4}
    over the eigenvalues of A.  Built from eigenvalues rather than det(A) so
    that A losing positive-definiteness gives nan: a determinant alone can
    stay positive with two negative eigenvalues."""
    return np.prod((np.linalg.eigvalsh(np.real(Z)) / np.pi) ** 0.25)


def pair_moments(Z1, q1, p1, Z2, q2, p2, beta=None, xc=None):
    """Return (S, zbar (D,), cov (D,D)) for <g1| . |g2>, optionally with an
    extra factor exp[-sum_k beta_k (x_k - xc_k)^2] in the integrand
    (completing the square only shifts Zp, Yp and Kp)."""
    Z1c = np.conj(Z1)
    Zp = 0.5 * (Z1c + Z2)
    Yp = Z1c @ q1 + Z2 @ q2 + 1j * (p2 - p1)
    Kp = (-0.5 * q1 @ Z1c @ q1 - 0.5 * q2 @ Z2 @ q2
          + 1j * (p1 @ q1) - 1j * (p2 @ q2))
    if beta is not None:
        Zp = Zp + np.diag(beta)
        Yp = Yp + 2.0 * beta * xc
        Kp = Kp - np.sum(beta * xc ** 2)
    Zinv = np.linalg.inv(Zp)
    sol = Zinv @ Yp
    # sqrt(pi^D / det Zp) = prod_k sqrt(pi / lambda_k), principal branch
    S = (_norm(Z1) * _norm(Z2) * np.prod(np.sqrt(np.pi / np.linalg.eigvals(Zp)))
         * np.exp(0.25 * (Yp @ sol) + Kp))
    return S, 0.5 * sol, 0.5 * Zinv


def kinetic_expectation(Z, p, mass):
    """<g|T|g> = sum_k [ p_k^2 + 1/2 (Z A^-1 Z^*)_kk ] / (2 m_k), a float.
    For a real diagonal Z this is the familiar sum_k p_k^2/2m_k + a_k/4m_k."""
    A = np.real(Z)
    G = Z @ np.linalg.solve(A, np.conj(Z))
    return float(np.sum((p ** 2 + 0.5 * np.real(np.diag(G))) / (2.0 * mass)))


def wavefunction(Z, q, p, x):
    """Evaluate the normalised Gaussian on grid points.
    Z : (D,D) complex,  q, p : (D,),  x : (..., D)
    returns a complex array of shape x.shape[:-1]
    """
    d = x - q
    quad = np.einsum("...k,kl,...l->...", d, Z, d)
    return _norm(Z) * np.exp(-0.5 * quad + 1j * (d @ p))


def coherent_Z(mass, hess):
    """Stationary width matrix: the solution of Z M^-1 Z = hess, i.e.Yolo!1097
    Z = M^1/2 sqrt(M^-1/2 hess M^-1/2) M^1/2.  Falls back to unit eigenvalue
    along directions with non-positive curvature (so a free mode gets
    Z_kk = m_k there, the mass-scaled analogue of zeta = 1)."""
    s = np.sqrt(np.asarray(mass, float))
    Hm = np.asarray(hess, float) / np.outer(s, s)
    w, U = np.linalg.eigh(Hm)
    w = np.where(w > 1e-12, np.sqrt(np.maximum(w, 0.0)), 1.0)
    return (U * w) @ U.T * np.outer(s, s)


def coherent_zeta(mass, omega):
    """Stationary width sqrt(m)*omega of a single mode -- the scalar form of
    coherent_Z on a harmonic surface.  Kept because initialize.model_widths
    uses it to pick default per-mode widths."""
    return np.sqrt(np.asarray(mass, float)) * np.asarray(omega, float)


def marginal_widths(Z):
    """Effective per-mode width a_k of one state's width matrix Z (D,D).

    |g|^2 is a Gaussian with covariance Sigma = (2 Re Z)^-1.  Integrating out
    every other mode leaves mode k with variance Sigma_kk, i.e. the shape of
    a 1-D Gaussian exp[-a_k (x_k - q_k)^2] with

        a_k = 1 / (2 Sigma_kk) = 1 / [(Re Z)^-1]_kk .

    For a diagonal Z this is exactly a_k = Re Z_kk.  Otherwise a_k is SMALLER:
    a tilted packet is wider in each individual mode than its own diagonal
    element suggests.  Used for the separable-diagonal quadrature (each term
    v_i(k, x_k) sees only mode k's marginal) and for the per-mode widths
    reported by run().

    Returns a real (D,) array.
    """
    Ainv = np.linalg.inv(np.real(Z))
    D = len(Ainv)
    a = np.zeros(D)
    for k in range(D):
        a[k] = 1.0 / Ainv[k, k]
    return a


# ==========================================================================
#  Model helpers
# ==========================================================================
def lvc_full(E, kappa, hess, J, lam=None, mass=None, gauss=None,
             gauss_lin=None, name="LVC with full Hessian"):
    """Quadratic-diagonal model with a general (non-diagonal) Hessian:

        V_ii(x) = E_i + kappa_i . x + 1/2 x^T hess_i x

    hess : (N,D,D) symmetric, or (D,D) shared by all states.  Couplings as in
    initialize.lvc.  A non-diagonal Hessian is one of the two things that
    make a packet tilt (the other is the inter-state coupling bracket).
    """
    E = np.atleast_1d(np.asarray(E, float))
    N = len(E)
    kappa = np.atleast_2d(np.asarray(kappa, float))
    D = kappa.shape[1]
    hess = np.asarray(hess, float)
    if hess.ndim == 2:
        hess = np.tile(hess, (N, 1, 1))
    if not np.allclose(hess, np.swapaxes(hess, 1, 2)):
        raise ValueError("hess must be symmetric")
    J = np.asarray(J, float).copy()
    np.fill_diagonal(J, 0.0)
    lam = np.zeros((N, N, D)) if lam is None else np.asarray(lam, float)
    mass = np.ones(D) if mass is None else np.asarray(mass, float)
    return dict(name=name, N=N, D=D, mass=mass, diag_kind="quadratic_full",
                E=E, kappa=kappa, hess=hess,
                omega2=np.array([np.diag(h) for h in hess]),
                J=J, lam=lam, gauss=gauss, gauss_lin=gauss_lin,
                quad_coupling=None)


def state_hessian(model, i):
    """Constant (D,D) Hessian of state i's diagonal potential.

    Defined for the two quadratic kinds only.  A separable diagonal has no
    constant Hessian -- its curvature is an expectation that changes with the
    packet, so diagonal_terms() computes it per call.
    """
    kind = model["diag_kind"]
    if kind == "quadratic_full":
        return model["hess"][i]
    if kind == "quadratic":
        return np.diag(np.asarray(model["omega2"][i], float))
    raise NotImplementedError("no constant Hessian for diag_kind=%r" % kind)


def default_Z(model):
    """(N,D,D) default initial widths: each state's coherent width matrix.

    Uses the state's own Hessian for the quadratic kinds and diag(omega2_ref)
    for a separable one, which is what that field is there for.

    For a DIAGONAL Hessian (every model in the package except lvc_full) this
    is built per mode as Z_kk = sqrt(m_k K_k), falling back to 1.0 on a flat
    mode -- identical to initialize.model_widths, so a free or flat direction
    gets width 1 and not the mass.  coherent_Z's own fallback is a unit
    eigenvalue of the MASS-WEIGHTED Hessian, which would give Z_kk = m_k
    there instead; that difference is a different physical initial condition,
    not a rounding detail, so the diagonal case is handled explicitly.
    """
    N, D = model["N"], model["D"]
    mass = np.asarray(model["mass"], float)
    out = np.zeros((N, D, D), dtype=complex)
    for i in range(N):
        if model["diag_kind"] == "separable":
            H = np.diag(np.asarray(model["omega2"][i], float))
        else:
            H = state_hessian(model, i)
        Hd = np.diag(H)
        if np.allclose(H, np.diag(Hd)):
            z = np.sqrt(np.maximum(Hd, 0.0) * mass)
            z = np.where(z > 1e-12, z, 1.0)
            out[i] = np.diag(z).astype(complex)
        else:
            out[i] = coherent_Z(mass, H)
    return out


def suggest_dt_full(model, safety=0.02):
    """Time step resolving the fastest frequency: mass-weighted Hessian
    eigenvalues, couplings and the diabatic energy span.

    initialize.suggest_dt is the equivalent for a model whose curvatures are
    per-mode; the two agree whenever the Hessian is diagonal.
    """
    s = np.sqrt(model["mass"])
    w2 = []
    for i in range(model["N"]):
        if model["diag_kind"] == "separable":
            H = np.diag(np.asarray(model["omega2"][i], float))
        else:
            H = state_hessian(model, i)
        w2.append(np.max(np.linalg.eigvalsh(H / np.outer(s, s))))
    rates = [float(np.sqrt(max(max(w2), 0.0))),
             float(np.max(np.abs(model["J"]))) if model["J"].size else 0.0,
             float(np.ptp(model["E"])) if model.get("E") is not None else 0.0,
             1e-12]
    for key in ("gauss", "gauss_lin"):
        if model.get(key):
            rates.append(float(np.max(np.abs(model[key]["A"]))))
    qc = model.get("quad_coupling")
    if qc is not None:
        m = qc.get("margin", 15.0)
        rates.append(float(np.max(np.abs(qc["v"](np.linspace(-m, m, 400))))))
    dt = safety * 2.0 * np.pi / max(rates)
    if qc is not None:
        dt /= 20.0          # see initialize.suggest_dt for why
    return dt


# ==========================================================================
#  Dynamics
# ==========================================================================
GH_ORDER = 48                       # nodes for separable-diagonal quadrature
_GH_CACHE = {}


def _gh(n=GH_ORDER):
    if n not in _GH_CACHE:
        _GH_CACHE[n] = np.polynomial.hermite.hermgauss(n)
    return _GH_CACHE[n]


def pack(q, p, Z, c):
    return np.concatenate([np.asarray(q, float).ravel(),
                           np.asarray(p, float).ravel(),
                           np.real(Z).ravel(), np.imag(Z).ravel(),
                           np.real(c), np.imag(c)])


def unpack(y, N, D):
    n, m = N * D, N * D * D
    q = y[0:n].reshape(N, D)
    p = y[n:2 * n].reshape(N, D)
    Z = (y[2 * n:2 * n + m] + 1j * y[2 * n + m:2 * n + 2 * m]).reshape(N, D, D)
    o = 2 * n + 2 * m
    c = y[o:o + N] + 1j * y[o + N:o + 2 * N]
    return q, p, Z, c


def _coupled(model, i, j):
    """True if states i and j have any direct coupling term."""
    if model["J"][i, j] != 0.0 or np.any(np.asarray(model["lam"][i, j]) != 0.0):
        return True
    for key in ("gauss", "gauss_lin"):
        g = model.get(key)
        if g is not None and np.asarray(g["A"], float)[i, j] != 0.0:
            return True
    qc = model.get("quad_coupling")
    return qc is not None and {i, j} == {qc["i"], qc["j"]}


def seed_uniform(c0, q0, p0, Z0, theta=1e-2):
    """Uniform seed:  psi_j = sin(theta) g_j  for every exactly-empty state j.

    The equations of motion divide by |c_j|^2, so an exactly-empty state
    must be given a small amplitude before propagation.

    Source i is the most-populated state.  Each empty state KEEPS the
    Gaussian given for it in q0/p0/Z0; only its amplitude is set.
    Amplitudes are a rotation by theta:  c_i -> c_i cos(theta),
    c_j = c_i sin(theta), so for c0 = [1, 0]
        psi_1 = cos(theta) g_1,   psi_2 = sin(theta) g_2.
    Give state j the same q0/p0/Z0 as state i to seed a copy of g_i.

    Returns (c, q, p, Z) with c renormalised (a no-op for one empty state
    and a normalised c0).
    """
    c = np.array(c0, dtype=complex)
    q = np.array(q0, dtype=float)
    p = np.array(p0, dtype=float)
    Z = np.array(Z0, dtype=complex)
    empty = np.flatnonzero(np.abs(c) == 0.0)
    if len(empty) == 0:
        return c, q, p, Z
    i = int(np.argmax(np.abs(c)))
    for j in empty:
        c[j] = np.sin(theta) * c[i]
    c[i] = np.cos(theta) * c[i]
    return c / np.linalg.norm(c), q, p, Z


def gauss_pair(g, i, j):
    """(beta, xc), each (D,), of the Gaussian coupling between states i, j.

    g["beta"] and g["xc"] are either (D,) -- one width and centre shared by
    every pair -- or (N,N,D) -- a separate one per pair, e.g. couplings
    centred at different crossing points."""
    beta = np.asarray(g["beta"], float)
    xc = np.asarray(g["xc"], float)
    if beta.ndim == 3:
        beta = beta[i, j]
    if xc.ndim == 3:
        xc = xc[i, j]
    return beta, xc


def seed_coupling(model, c0, q0, p0, Z0, theta=1e-2, width_shape="full",
                  width="coupling", Zcoh=None):
    """Coupling-shaped seed:  psi_j = eps V_ji psi_i  for every exactly-empty j.

    seed_uniform() gives an empty state a copy of the occupied Gaussian.
    Here the empty state's Gaussian is instead the one the coupling itself
    would create: to first order in a
    short time tau, exact dynamics give psi_j(tau) = -i tau V_ji psi_i, so
    V_ji psi_i is the SHAPE population arrives in.

    Source i is the most-populated state.  The Gaussian g_j is fitted to
    f = V_ji g_i:
        |f|^2 = V_ji^2 |g_i|^2  ->  q_j = mean,  A_j = (2 cov)^-1
        V_ji is real, so f keeps g_i's chirp and momentum: B_j = B_i and
        p_j = p_i.
    This is EXACT for a constant coupling (g_j = g_i) and a Gaussian one
    (a product of Gaussians is a Gaussian, handled in closed form).  For any
    other coupling (linear, gauss_lin, quad_coupling, or a mix) f is not a
    Gaussian and the moments come from Gauss-Hermite quadrature over g_i
    (D <= 3 only); g_j is then the best-matching single Gaussian.
    For an odd coupling (e.g. gauss_lin at its node) f has two lobes, and one
    Gaussian can only cover both with a broad, centred packet.

    Amplitudes are a rotation by theta, with the -i of the short-time
    result psi_j(tau) = -i tau V_ji psi_i:
        c_i -> c_i cos(theta),   c_j = -i |c_i| sin(theta),
    so for c0 = [1, 0]
        psi_1 = cos(theta) g_1,   psi_2 = -i sin(theta) V g_1 / ||V g_1||
    (up to a constant phase; exactly a Gaussian when V g_1 is one).  V g_1 is NORMALISED: taken
    literally, sin(theta) V g_1 has population sin^2 ||V g_1||^2, which is
    ~1e-16 when the packet starts far from the coupling and makes the
    1/|c_j|^2 in the equations of motion blow up.

    An empty state with NO direct coupling to the source (e.g. a third state
    coupled only to state 2) is seeded as in seed_uniform: it keeps the
    q0, p0, Z0 given for it and gets c_j = |c_i| sin(theta).

    width : the width given to g_j.
        'coupling' (default) -- the fitted width of V_ji g_i, e.g.
                   A_j = A_i + 2 beta for a Gaussian coupling (twice the
                   coherent width when A_i = 2 beta, as for Holstein).
        'coherent' -- the width Zcoh[j] (run() passes the Z0 given for
                   state j; default default_Z(model)[j], i.e. sqrt(m k_j) on
                   a harmonic surface), keeping the fitted centre q_j,
                   p_j = p_i and c_j = -i sin(theta).  A coherent packet does
                   not breathe on its own surface, so frozen widths stay
                   physical.

    Returns (c, q, p, Z) with c renormalised.
    """
    if width not in ("coupling", "coherent"):
        raise ValueError("width must be 'coupling' or 'coherent' (got %r)"
                         % (width,))
    c = np.array(c0, dtype=complex)
    q = np.array(q0, dtype=float)
    p = np.array(p0, dtype=float)
    Z = np.array(Z0, dtype=complex)
    empty = np.flatnonzero(np.abs(c) == 0.0)
    if len(empty) == 0:
        return c, q, p, Z
    D = model["D"]
    i = int(np.argmax(np.abs(c)))
    Ai, Bi = np.real(Z[i]), np.imag(Z[i])
    g = model.get("gauss")
    gl = model.get("gauss_lin")
    qc = model.get("quad_coupling")

    for j in empty:
        has_J = model["J"][j, i] != 0.0
        has_lam = np.any(np.asarray(model["lam"][j, i]) != 0.0)
        has_g = g is not None and np.asarray(g["A"], float)[j, i] != 0.0
        has_gl = gl is not None and np.asarray(gl["A"], float)[j, i] != 0.0
        has_qc = qc is not None and {i, j} == {qc["i"], qc["j"]}
        if not (has_J or has_lam or has_g or has_gl or has_qc):
            # no direct coupling to the source (e.g. a third state coupled
            # only to state 2): seed it as seed_uniform does -- keep the
            # Gaussian given for it, c_j = |c_i| sin(theta)
            c[j] = abs(c[i]) * np.sin(theta)
            continue

        if not (has_lam or has_g or has_gl or has_qc):
            # constant coupling: V_ji psi_i = J psi_i, same Gaussian
            Aj, qj = Ai.copy(), q[i].copy()
        elif has_g and not (has_J or has_lam or has_gl or has_qc):
            # A exp[-sum beta (x-xc)^2] g_i is again a Gaussian
            beta, xc = gauss_pair(g, j, i)
            Aj = Ai + 2.0 * np.diag(beta)
            qj = np.linalg.solve(Aj, Ai @ q[i] + 2.0 * beta * xc)
        else:
            if D > 3:
                raise NotImplementedError(
                    "seed_mode='coupling' with a non-constant, non-Gaussian "
                    "coupling needs quadrature: D <= 3 only (got D=%d)" % D)
            n = {1: 80, 2: 40, 3: 20}[D]
            t, w = np.polynomial.hermite.hermgauss(n)
            T = np.stack(np.meshgrid(*([t] * D), indexing="ij"), -1).reshape(-1, D)
            Wt = np.prod(np.stack(np.meshgrid(*([w] * D), indexing="ij"), -1)
                         .reshape(-1, D), axis=1)
            L = np.linalg.cholesky(np.linalg.inv(2.0 * Ai))
            X = q[i] + np.sqrt(2.0) * T @ L.T
            V = potential_matrix(model, X)[:, j, i]
            wt = Wt * V ** 2
            s = np.sum(wt)
            if s <= 1e-300:
                Aj, qj = Ai.copy(), q[i].copy()
            else:
                qj = wt @ X / s
                d = X - qj
                cov = (wt[:, None] * d).T @ d / s
                Aj = 0.5 * np.linalg.inv(cov)

        if width_shape == "diagonal":
            # keep each mode's marginal width, drop the correlation
            Aj = np.diag(1.0 / (2.0 * np.diag(np.linalg.inv(2.0 * Aj))))
        Zj = Aj + 1j * Bi
        if width == "coherent":
            Zj = (default_Z(model) if Zcoh is None else Zcoh)[j].copy()
        pj = p[i].copy()
        q[j], p[j], Z[j] = qj, pj, Zj

        c[j] = -1j * abs(c[i]) * np.sin(theta)

    c[i] = np.cos(theta) * c[i]
    return c / np.linalg.norm(c), q, p, Z


def diagonal_terms(model, i, q, Z):
    """Return (<V_ii>, <grad V_ii> (D,), <Hess V_ii> (D,D)) for state i.

    'quadratic' and 'quadratic_full' are closed form, with the Hessian
    constant.  'separable' uses one 48-node Gauss-Hermite rule per mode on
    that mode's MARGINAL width (marginal_widths), which is exact for
    V_ii = sum_k v_i(k, x_k) because each term sees only its own mode's
    marginal; its Hessian is diagonal (a separable potential has no cross
    derivatives) and is obtained by central-differencing the force
    expectation, so it carries both quadrature and finite-difference error.
    A separable model that carries model["morse"] (N, D, 4) = (De, b, Re, c)
    per state and mode instead uses the EXACT closed forms for Morse modes.
    """
    if model["diag_kind"] in ("quadratic", "quadratic_full"):
        K = state_hessian(model, i)
        V = float(model["E"][i] + model["kappa"][i] @ q + 0.5 * q @ K @ q
                  + 0.25 * np.trace(np.linalg.solve(np.real(Z), K)))
        return V, model["kappa"][i] + K @ q, K

    a = marginal_widths(Z)
    D = model["D"]
    morse = model.get("morse")
    # Morse modes v = De (1 - e^{-b y})^2 + c, y = x - Re: exact, since
    # <e^{-n b y}> = exp[-n b (q - Re) + n^2 b^2 / (4 a)] for a Gaussian with
    # marginal width a (variance 1/(2a)).  A row of NaNs marks a mode that is
    # NOT Morse (e.g. a harmonic state in a mixed model); those fall through to
    # the Gauss-Hermite branch below, mode by mode.
    is_morse = np.zeros(D, dtype=bool)
    if morse is not None:
        V = 0.0
        F = np.zeros(D)
        d2 = np.zeros(D)
        for k in range(D):
            if not np.all(np.isfinite(morse[i][k])):
                continue
            is_morse[k] = True
            De, b, Re, c = morse[i][k]
            E1 = np.exp(-b * (q[k] - Re) + b * b / (4.0 * a[k]))
            E2 = np.exp(-2.0 * b * (q[k] - Re) + b * b / a[k])
            V += De * (1.0 - 2.0 * E1 + E2) + c
            F[k] = 2.0 * De * b * (E1 - E2)
            d2[k] = 2.0 * De * b * b * (2.0 * E2 - E1)
        if is_morse.all():
            return float(V), F, np.diag(d2)
    else:
        V = 0.0
        F = np.zeros(D)
        d2 = np.zeros(D)

    t, w = _gh()
    vf, df = model["vfun"][i], model["dvfun"][i]
    for k in range(D):
        if is_morse[k]:
            continue
        xk = q[k] + t / np.sqrt(a[k])
        V += float(np.sum(w * vf(k, xk)) / np.sqrt(np.pi))
        F[k] = float(np.sum(w * df(k, xk)) / np.sqrt(np.pi))
        h = 1e-4 / np.sqrt(a[k])
        xp = q[k] + h + t / np.sqrt(a[k])
        xm = q[k] - h + t / np.sqrt(a[k])
        d2[k] = float((np.sum(w * df(k, xp)) - np.sum(w * df(k, xm)))
                      / (2 * h * np.sqrt(np.pi)))
    return V, F, np.diag(d2)


def coupling_terms(model, i, j, q1, p1, Z1, q2, p2, Z2, second=False):
    """Return (W_ij, M_ij (D,)), or (W_ij, M_ij, R_ij (D,D)) if `second`.

        W_ij = <g_i|V_ij|g_j>
        M_ij = <g_i|(x - q_i) V_ij|g_j>
        R_ij = <g_i|(x - q_i)(x - q_i)^T V_ij|g_j>

    All four coupling channels add.  With u = x - zbar under the pair measure
    (covariance cov) and d = zbar - q1:  <(x-q1)(x-q1)^T> = d d^T + cov, and a
    linear factor (x_l - a) contributes the extra d cov[:,l]^T + cov[:,l] d^T.
    R is needed only by the thawed width equation, so it is computed only when
    asked for.
    """
    D = len(q1)
    W = 0.0 + 0.0j
    M = np.zeros(D, dtype=complex)
    R = np.zeros((D, D), dtype=complex)

    Jij = model["J"][i, j]
    lij = model["lam"][i, j]
    if Jij != 0.0 or np.any(lij != 0.0):
        S, zb, cov = pair_moments(Z1, q1, p1, Z2, q2, p2)
        V_at = Jij + lij @ zb
        W += S * V_at
        M += S * ((zb - q1) * V_at + cov @ lij)
        d, cl = zb - q1, cov @ lij
        R += S * ((np.outer(d, d) + cov) * V_at + np.outer(d, cl) + np.outer(cl, d))

    g = model.get("gauss")
    if g is not None:
        Aij = np.asarray(g["A"], float)[i, j]
        if Aij != 0.0:
            S, zb, cov = pair_moments(Z1, q1, p1, Z2, q2, p2,
                                      *gauss_pair(g, i, j))
            W += Aij * S
            M += Aij * S * (zb - q1)
            R += Aij * S * (np.outer(zb - q1, zb - q1) + cov)

    gl = model.get("gauss_lin")
    if gl is not None:
        Aij = np.asarray(gl["A"], float)[i, j]
        if Aij != 0.0:
            L = gl["lin_dim"]
            xc = np.asarray(gl["xc"], float)
            S, zb, cov = pair_moments(Z1, q1, p1, Z2, q2, p2,
                                      np.asarray(gl["beta"], float), xc)
            lin = zb[L] - xc[L]
            W += Aij * S * lin
            M += Aij * S * ((zb - q1) * lin + cov[:, L])
            d = zb - q1
            R += Aij * S * ((np.outer(d, d) + cov) * lin
                            + np.outer(d, cov[:, L]) + np.outer(cov[:, L], d))

    qc = model.get("quad_coupling")
    if qc is not None and {i, j} == {qc["i"], qc["j"]}:
        # A 1-D coupling V_ij(x) with no closed form (e.g. Tully III's
        # asymmetric ECR coupling, which does not vanish as x -> +inf).
        # Evaluate the matrix elements by direct quadrature, on a grid
        # REBUILT each call around wherever the packets currently are (plus
        # x=0, where the coupling itself lives) -- not a fixed window, which a
        # travelling packet would eventually leave.
        if D != 1:
            raise NotImplementedError("quad_coupling is D=1 only (got D=%d)" % D)
        npts = qc.get("npts", 800)
        margin = qc.get("margin", 15.0)
        lo = min(0.0, q1[0], q2[0]) - margin
        hi = max(0.0, q1[0], q2[0]) + margin
        xg = np.linspace(lo, hi, npts)
        dx = xg[1] - xg[0]
        v12 = qc["v"](xg)
        g1 = wavefunction(Z1, q1, p1, xg[:, None])
        g2 = wavefunction(Z2, q2, p2, xg[:, None])
        integrand = np.conj(g1) * v12 * g2
        W += np.sum(integrand) * dx
        M[0] += np.sum(integrand * (xg - q1[0])) * dx
        R[0, 0] += np.sum(integrand * (xg - q1[0]) ** 2) * dx

    return (W, M, R) if second else (W, M)


def rhs(y, model, eps=1e-10, width_mode="frozen", width_shape="full",
        legacy=False, drop_corrections=0):
    """The TDVP equations of motion -- the only function that knows them.

    drop_corrections : 0 -> full equations of motion.
                       1 -> drop every coupling correction, i.e. all the
                            1/|c_i|^2 terms: (2/|c|^2) Im C in qdot,
                            (2/|c|^2)(A Re C + B Im C) in pdot and the
                            (4i/|c|^2) A[...]A bracket in Zdot.  Each Gaussian
                            then moves on its own surface only (qdot = p/m,
                            pdot = -<grad V_ii>, Heller's Zdot); the states
                            still exchange population through W in cdot.
                            Not the variational solution, so the energy is
                            no longer exactly conserved.
    """
    N, D, mass = model["N"], model["D"], model["mass"]
    q, p, Z, c = unpack(y, N, D)
    Minv = 1.0 / mass

    Hd = np.empty(N)
    F = np.empty((N, D))
    hess = np.empty((N, D, D))
    for i in range(N):
        V, F[i], hess[i] = diagonal_terms(model, i, q[i], Z[i])
        Hd[i] = kinetic_expectation(Z[i], p[i], mass) + V

    width_coupling = width_mode == "thawed" and not legacy \
        and not drop_corrections
    W = np.zeros((N, N), dtype=complex)
    Mc = np.zeros((N, N, D), dtype=complex)
    Rc = np.zeros((N, N, D, D), dtype=complex)
    for i in range(N):
        for j in range(N):
            if i != j:
                out = coupling_terms(model, i, j, q[i], p[i], Z[i],
                                     q[j], p[j], Z[j], second=width_coupling)
                W[i, j], Mc[i, j] = out[0], out[1]
                if width_coupling:
                    Rc[i, j] = out[2]

    qd = np.empty((N, D))
    pd = np.empty((N, D))
    Zd = np.zeros((N, D, D), dtype=complex)
    phase = np.zeros(N)
    for i in range(N):
        d = abs(c[i]) ** 2 + eps
        C = np.zeros(D, dtype=complex)
        for j in range(N):
            if j != i:
                C += np.conj(c[i]) * c[j] * Mc[i, j]
        A, B = np.real(Z[i]), np.imag(Z[i])
        qd[i] = Minv * p[i]
        pd[i] = -F[i]
        if not drop_corrections:
            qd[i] += (2.0 / d) * C.imag
            pd[i] -= (2.0 / d) * (A @ C.real)
            if not legacy:
                pd[i] -= (2.0 / d) * (B @ C.imag)
        if width_mode == "thawed":
            Zd[i] = -1j * ((Z[i] * Minv) @ Z[i] - hess[i])
            if width_coupling:
                half_Ainv = 0.5 * np.linalg.inv(A)
                X = np.zeros((D, D), dtype=complex)
                for j in range(N):
                    if j != i:
                        X += np.conj(c[i]) * c[j] * (Rc[i, j] - W[i, j] * half_Ainv)
                Zd[i] += (4j / d) * (A @ X @ A)
            if width_shape == "diagonal":
                Zd[i] = np.diag(np.diag(Zd[i]))
            if not legacy:
                phase[i] = 0.25 * np.trace(np.linalg.solve(A, np.imag(Zd[i])))

    # Global-phase gauge: subtracting the mean of the electronic diagonal is
    # the SAME for every state, hence a pure global phase.  Populations,
    # coherences c_i^* c_j, the nuclear EOMs and the energy are all exactly
    # unchanged, but the (O(D)) common kinetic phase is removed, which is
    # essential for stability at large D.
    Aph = np.array([Hd[i] - p[i] @ qd[i] - phase[i] for i in range(N)])
    Aph = Aph - Aph.mean()
    cd = -1j * (Aph * c + W @ c)

    return np.concatenate([qd.ravel(), pd.ravel(),
                           Zd.real.ravel(), Zd.imag.ravel(),
                           cd.real, cd.imag])


def rk4_step(y, dt, model, **kw):
    k1 = rhs(y, model, **kw)
    k2 = rhs(y + 0.5 * dt * k1, model, **kw)
    k3 = rhs(y + 0.5 * dt * k2, model, **kw)
    k4 = rhs(y + dt * k3, model, **kw)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _initial_Z(model, Z0):
    """(N,D,D) complex initial widths from None / scalar / (D,) / (D,D) / (N,D,D)."""
    N, D = model["N"], model["D"]
    if Z0 is None:
        return default_Z(model)
    Z0 = np.asarray(Z0, complex)
    if Z0.ndim == 0:
        Z0 = Z0 * np.eye(D)
    if Z0.ndim == 1:
        Z0 = np.diag(Z0)
    if Z0.ndim == 2 and Z0.shape == (N, D) and D != N:
        # a per-state list of per-mode widths
        Z0 = np.array([np.diag(row) for row in Z0])
    if Z0.ndim == 2:
        Z0 = np.tile(Z0, (N, 1, 1))
    return Z0.copy()


def run(model, c0, q0=None, p0=None, Z0=None, dt=None, t_max=100.0,
        width_mode="frozen", width_shape="full", legacy=False,
        record_every=1, eps=1e-10, seed=1e-2, stop=None, zeta0=None,
        energy_every=None, width_scale=1.0, seed_mode="uniform",
        drop_corrections=0, dt_out=None, seed_track=False,
        track_factor=3.0):
    """Propagate one trajectory.

    dt          : the RK4 integration step.
    dt_out      : the OUTPUT step, in time units: frames (t, q, p, Z, c, pops,
                  energy) are stored every dt_out.  Must be a whole multiple
                  of dt.  Overrides record_every; None -> use record_every.
    record_every: store a frame every this many RK4 steps (dt_out = this * dt).

    width_mode  : 'frozen' (Z held at Z0) or 'thawed' (Z evolves)
    width_shape : 'full' (general Z) or 'diagonal' (only diag(Zdot) is kept,
                  so Z stays diagonal -- one independent width per mode).
                  Requires a diagonal Z0.
    Z0          : None (each state's coherent width matrix), a scalar, a (D,)
                  diagonal, an (N,D) list of per-mode widths, a (D,D), or an
                  (N,D,D).
    zeta0       : accepted as an alias for Z0, for code written against the
                  older per-mode interface.  Passing both raises.
    width_scale : s, multiplies the initial width: Z(0) = s * Z0.  With
                  Z0=None this picks coherent or non-coherent widths:
                    s = 1  coherent (the ground-state width of each state's
                           own well, stationary there)
                    s > 1  non-coherent, narrower in x (squeezed)
                    s < 1  non-coherent, wider in x
                  A non-coherent packet breathes on its own surface when
                  width_mode='thawed'; 'frozen' keeps it at s * coherent.
                  With an explicit Z0 it scales that Z0 instead.
    seed        : theta, the amplitude put on each exactly-empty state (the
                  1/|c|^2 in the equations of motion is singular at c=0);
                  seeded population ~ theta^2.  None or 0 turns seeding off.
    seed_track  : True -> "track then release": every initially EMPTY state
                  keeps its Gaussian (q, p, Z) ON the Gaussian of the
                  most-populated state it is directly coupled to, until its
                  own population exceeds the release threshold; from then on it moves
                  freely, i.e. once track_factor x its seeded population
                  sin^2(seed) (population has started to arrive).  For
                  localised couplings on steep surfaces (e.g. the
                  Coronado Morse models), where a Gaussian seeded at t=0 is
                  thrown away before the population arrives.  While a state is
                  tracked its Gaussian is reset every step, so energy and norm
                  are conserved only to O(population of the tracked state).
    track_factor : a tracked state is released when its population exceeds
                  track_factor * sin^2(seed) (default 3: tested on the
                  Coronado models against the exact result).
    seed_mode   : how an empty state is seeded:
                  'uniform'  -- psi_i = cos(theta) g_i,
                                psi_j = sin(theta) g_j  (seed_uniform):
                                the empty state KEEPS the q0, p0, Z0 given
                                for it (same as state i -> a copy of g_i).
                  'coupling' -- psi_j = -i sin(theta) V_ji g_i / ||V_ji g_i||
                                from the most-populated state i: the empty
                                state's Gaussian is fitted to the coupling
                                acting on g_i, p_j = p_i (seed_coupling);
                                replaces the q0, p0, Z0 given for it.
                  'shift'    -- as 'coupling' (shifted centre, p_j = p_i,
                                c_j = -i sin(theta)) but state j KEEPS the
                                width given for it in Z0 (times width_scale)
                                instead of the fitted one.  With Z0=None that
                                is its coherent width, so the seeded packet
                                does not breathe and frozen widths stay
                                physical; for a surface with no well (Tully)
                                give Z0 explicitly.
    stop      : optional callable stop(q, p, c) -> bool, checked each step;
                  when it first returns True the loop ends, so the returned
                  arrays are SHORTER than t_max implies -- read result["t"][-1].
    legacy      : reproduce the pre-correction equations (see module docstring).
    drop_corrections : 0 (default) propagates the full equations of motion;
                  1 drops all coupling corrections (the 1/|c|^2 terms in
                  qdot, pdot and Zdot), so each Gaussian moves on its own
                  surface and the states couple only through W in cdot.
                  See rhs().
    energy_every: record the energy and norm every this many steps,
                  independently of record_every (None -> record_every).
                  energy_every=1 checks conservation at EVERY time step
                  without storing the full state that often; costs one
                  total_energy() per recorded step (about a quarter of an
                  RK4 step).

    Returns a dict of arrays:
        t (T,), q p (T,N,D), Z (T,N,D,D) complex, c (T,N) complex,
        zeta (T,N,D) complex and sigma (T,N,D): the per-mode MARGINAL widths
            (marginal_widths, plus each mode's own chirp Im Z_kk) and
            sigma = 1/sqrt(2 Re zeta).  Exact for a diagonal Z; for a tilted
            one the real part is still the exact marginal but the chirp is
            not a marginal quantity.  Read Z for the full width.
        pops (T,N), energy (T,), norm (T,), coupling (T,) = <V_c>, the
            coupling part of the energy (coupling_expectation), and
            W (T,N,N) complex = <g_i|V_ij|g_j> between the Gaussians
            themselves, without the c's (coupling_matrix)  -- at the
            record_every frames,
        t_E (TE,), energy_steps (TE,), norm_steps (TE,)  -- at the
            energy_every steps; dEmax and dNmax are taken over these,
        offdiag (T,) = max |Z_kl| over states, k != l -- zero exactly when the
            packets never tilted, i.e. when 'diagonal' would have done,
        E0, dEmax, dNmax, N, D, dt, width_mode, width_shape, width_scale,
        seed_mode, drop_corrections, legacy, model_name.

    Cost is O(N^2 D^3) per step.  Measured against the older per-mode engine:
    about 3x at D=10, 24x at D=60 and 200x at D=200, since the integrals no
    longer factorise over modes.  width_shape='diagonal' restricts the ANSATZ
    but not the cost -- the pair integrals are still D x D.
    """
    if width_mode not in ("frozen", "thawed"):
        raise ValueError("width_mode must be 'frozen' or 'thawed' (got %r)"
                         % (width_mode,))
    if width_shape not in ("full", "diagonal"):
        raise ValueError("width_shape must be 'full' or 'diagonal' (got %r)"
                         % (width_shape,))
    if zeta0 is not None:
        if Z0 is not None:
            raise ValueError("pass Z0 or zeta0, not both")
        Z0 = zeta0
    N, D = model["N"], model["D"]
    if dt is None:
        dt = suggest_dt_full(model)
    c0 = np.asarray(c0, complex)
    c0_input = c0.copy()
    q0 = np.zeros((N, D)) if q0 is None else np.broadcast_to(
        np.asarray(q0, float), (N, D)).copy()
    p0 = np.zeros((N, D)) if p0 is None else np.broadcast_to(
        np.asarray(p0, float), (N, D)).copy()
    Z0 = width_scale * _initial_Z(model, Z0)
    if seed:
        if seed_mode == "uniform":
            c0, q0, p0, Z0 = seed_uniform(c0, q0, p0, Z0, seed)
        elif seed_mode in ("coupling", "shift"):
            c0, q0, p0, Z0 = seed_coupling(
                model, c0, q0, p0, Z0, seed, width_shape,
                width="coherent" if seed_mode == "shift" else "coupling",
                Zcoh=Z0)
        else:
            raise ValueError("seed_mode must be 'uniform', 'coupling' or "
                             "'shift'")
    if width_shape == "diagonal":
        off = Z0 - np.array([np.diag(np.diag(z)) for z in Z0])
        if np.any(off != 0):
            raise ValueError("width_shape='diagonal' needs a diagonal Z0")
    if drop_corrections not in (0, 1):
        raise ValueError("drop_corrections must be 0 or 1")
    kw = dict(eps=eps, width_mode=width_mode, width_shape=width_shape,
              legacy=legacy, drop_corrections=drop_corrections)

    y = pack(q0, p0, Z0, c0)
    E0 = total_energy(y, model)
    N0 = float(np.sum(np.abs(c0) ** 2))

    if dt_out is not None:
        record_every = int(round(dt_out / dt))
        if record_every < 1 or abs(record_every * dt - dt_out) > 1e-9 * dt_out:
            raise ValueError("dt_out=%g must be a whole multiple of dt=%g"
                             % (dt_out, dt))
    if energy_every is None:
        energy_every = record_every
    ts, qs, ps, Zs, cs, Es, Vcs, Ws = [], [], [], [], [], [], [], []
    tEs, Efine, Nfine = [], [], []
    nsteps = int(round(t_max / dt))
    t = 0.0
    tracked = []
    if seed_track:
        tracked = [j for j in range(N) if abs(c0_input[j]) == 0.0]
        feeders = {j: [i for i in range(N) if i != j and _coupled(model, i, j)]
                   or [i for i in range(N) if i != j] for j in tracked}
    release_t = {}
    for step in range(nsteps + 1):
        q, p, Z, c = unpack(y, N, D)
        if tracked:
            pop = np.abs(c) ** 2 / np.sum(np.abs(c) ** 2)
            for j in list(tracked):
                if pop[j] > track_factor * np.sin(seed or 0.0) ** 2:
                    tracked.remove(j)
                    release_t[j] = t
                    continue
                src = max(feeders[j], key=lambda i: pop[i])
                q[j], p[j], Z[j] = q[src], p[src], Z[src]
            y = pack(q, p, Z, c)
        if step % record_every == 0:
            ts.append(t); qs.append(q.copy()); ps.append(p.copy())
            Zs.append(Z.copy()); cs.append(c.copy())
            Es.append(total_energy(y, model))
            Ws.append(coupling_matrix(y, model))
            Vcs.append(float(np.real(np.conj(c) @ Ws[-1] @ c)
                             / np.sum(np.abs(c) ** 2)))
        done = step == nsteps or (stop is not None and stop(q, p, c))
        if step % energy_every == 0 or done:
            tEs.append(t)
            Efine.append(Es[-1] if step % record_every == 0
                         else total_energy(y, model))
            Nfine.append(float(np.sum(np.abs(c) ** 2)))
        if done:
            break
        y = rk4_step(y, dt, model, **kw)
        t += dt

    Zs, cs, Es = np.array(Zs), np.array(cs), np.array(Es)
    Efine, Nfine = np.array(Efine), np.array(Nfine)
    nrm = np.sum(np.abs(cs) ** 2, axis=1)

    nt = len(ts)
    zeta = np.zeros((nt, N, D), dtype=complex)
    sigma = np.zeros((nt, N, D))
    offdiag = np.zeros(nt)
    for it in range(nt):
        for i in range(N):
            a = marginal_widths(Zs[it, i])
            for k in range(D):
                zeta[it, i, k] = a[k] + 1j * np.imag(Zs[it, i, k, k])
                sigma[it, i, k] = 1.0 / np.sqrt(2.0 * a[k])
                for l in range(D):
                    if l != k and abs(Zs[it, i, k, l]) > offdiag[it]:
                        offdiag[it] = abs(Zs[it, i, k, l])

    return dict(t=np.array(ts), q=np.array(qs), p=np.array(ps), Z=Zs, c=cs,
                zeta=zeta, sigma=sigma,
                pops=np.abs(cs) ** 2 / nrm[:, None], energy=Es, norm=nrm,
                coupling=np.array(Vcs), W=np.array(Ws),
                t_E=np.array(tEs), energy_steps=Efine, norm_steps=Nfine,
                offdiag=offdiag, E0=E0,
                dEmax=float(np.max(np.abs(Efine - E0))),
                dNmax=float(np.max(np.abs(Nfine - N0))), N=N, D=D, dt=dt, dt_out=record_every * dt,
                width_mode=width_mode, width_shape=width_shape,
                width_scale=width_scale, seed_mode=seed_mode, seed_track=seed_track,
                release_t={int(k): float(v) for k, v in release_t.items()},
                drop_corrections=drop_corrections, legacy=legacy, model_name=model.get("name", ""))


# ==========================================================================
#  Observables
# ==========================================================================
def total_norm(y, N, D):
    _, _, _, c = unpack(y, N, D)
    return float(np.sum(np.abs(c) ** 2))


def total_energy(y, model):
    """Variational energy <Psi|H|Psi>/<Psi|Psi>, including every off-diagonal
    W_ij term.  This is what run() records and what conservation checks use."""
    N, D = model["N"], model["D"]
    q, p, Z, c = unpack(y, N, D)
    num = 0.0 + 0.0j
    for i in range(N):
        V, _, _ = diagonal_terms(model, i, q[i], Z[i])
        num += abs(c[i]) ** 2 * (kinetic_expectation(Z[i], p[i], model["mass"]) + V)
        for j in range(N):
            if j != i:
                W, _ = coupling_terms(model, i, j, q[i], p[i], Z[i],
                                      q[j], p[j], Z[j])
                num += np.conj(c[i]) * c[j] * W
    return float((num / np.sum(np.abs(c) ** 2)).real)


def coupling_matrix(y, model):
    """(N,N) complex W_ij = <g_i|V_ij|g_j> between the (normalised) Gaussians
    of states i != j; zero on the diagonal.  Hermitian: W_ji = W_ij^*."""
    N, D = model["N"], model["D"]
    q, p, Z, c = unpack(y, N, D)
    W = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            if j != i:
                W[i, j] = coupling_terms(model, i, j, q[i], p[i], Z[i],
                                         q[j], p[j], Z[j])[0]
    return W


def coupling_expectation(y, model):
    """<V_c> = sum_{i != j} c_i^* c_j W_ij / <Psi|Psi>, with W_ij =
    <g_i|V_ij|g_j>: the off-diagonal (coupling) part of total_energy.
    Real, because W_ji = W_ij^*."""
    N, D = model["N"], model["D"]
    q, p, Z, c = unpack(y, N, D)
    num = 0.0 + 0.0j
    for i in range(N):
        for j in range(N):
            if j != i:
                W, _ = coupling_terms(model, i, j, q[i], p[i], Z[i],
                                      q[j], p[j], Z[j])
                num += np.conj(c[i]) * c[j] * W
    return float((num / np.sum(np.abs(c) ** 2)).real)


def populations(result):
    """(T,N) diabatic populations (already normalised in run)."""
    return result["pops"]


def reconstruct(result, step, x, state=None, c=None):
    """Evaluate psi_i(x) = c_i g_i(x) at a recorded time index.

    x     : (M,D) grid points (or (M,) for D=1)
    state : which electronic state, or None for all -> list of arrays
    c     : optional amplitudes; defaults to the recorded ones, so unlike the
            older per-mode engine the phases are right without passing them.
    """
    N, D = result["N"], result["D"]
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    amps = result["c"][step] if c is None else np.asarray(c, complex)
    out = []
    idx = range(N) if state is None else [state]
    for i in idx:
        g = wavefunction(result["Z"][step, i], result["q"][step, i],
                         result["p"][step, i], x)
        out.append(amps[i] * g)
    return out if state is None else out[0]


def density(result, step, x, c=None):
    """Total nuclear density sum_i |psi_i(x)|^2 at a recorded time index."""
    psis = reconstruct(result, step, x, state=None, c=c)
    return sum(np.abs(p) ** 2 for p in psis)


def exact_level_populations(H, c0, times):
    """|c(t)|^2 for i cdot = H c with constant H (matrix exponential)."""
    from scipy.linalg import expm
    H = np.asarray(H, complex); c0 = np.asarray(c0, complex)
    return np.array([np.abs(expm(-1j * H * t) @ c0) ** 2 for t in times])


def free_width_exact(zeta0, mass, t):
    """zeta(t) = zeta0 / (1 + i zeta0 t/m)  (omega = 0)."""
    return zeta0 / (1.0 + 1j * zeta0 * np.asarray(t) / mass)


def breathing_width_exact(zeta0, mass, omega, t):
    """Closed-form solution of zetadot = -i(zeta^2/m - omega^2):

        zeta(t) = sqrt(m) w (1 + K e^{-2i Omega t})/(1 - K e^{-2i Omega t})
        K = (zeta0 - sqrt(m) w)/(zeta0 + sqrt(m) w),  Omega = w/sqrt(m)
    """
    zc = np.sqrt(mass) * omega
    K = (zeta0 - zc) / (zeta0 + zc)
    Om = omega / np.sqrt(mass)
    e = K * np.exp(-2j * Om * np.asarray(t))
    return zc * (1.0 + e) / (1.0 - e)


# ==========================================================================
#  Exact reference -- 2 states, 1 mode
# ==========================================================================
# Split-operator FFT propagation of the full two-component TDSE
#
#     i d/dt [psi_1, psi_2] = ( T + V(x) ) [psi_1, psi_2]
#
# with the 2x2 potential matrix exponentiated ANALYTICALLY at each grid point.
# For a Hermitian 2x2 matrix  V = abar*I + [[d, vc], [vc, -d]]  with
# d = (v1-v2)/2, abar = (v1+v2)/2 and R = sqrt(d^2 + vc^2),
#
#     exp(-i V tau) = e^{-i abar tau} [ cos(R tau) I
#                                       - i sin(R tau)/R * [[d, vc],[vc, -d]] ]
#
# An INDEPENDENT check on the variational method: a different algorithm
# (grid + FFT) solving the same physics with no Gaussian ansatz.

def grid(xmin, xmax, n):
    """Uniform grid and matching FFT momenta."""
    x = np.linspace(xmin, xmax, n, endpoint=False)
    dx = x[1] - x[0]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    return x, k, dx


def gaussian_on_grid(x, zeta, q, p):
    """Normalised complex-width Gaussian on a 1-D grid."""
    return (np.real(zeta) / np.pi) ** 0.25 * np.exp(
        -0.5 * zeta * (x - q) ** 2 + 1j * p * (x - q))


def _expV(v1, v2, vc, tau):
    """Analytic exp(-i V tau) for the 2x2 potential; returns U11, U12, U22."""
    abar = 0.5 * (v1 + v2)
    d = 0.5 * (v1 - v2)
    R = np.sqrt(d ** 2 + vc ** 2)
    Rs = np.where(R > 1e-300, R, 1.0)
    ph = np.exp(-1j * abar * tau)
    co, si = np.cos(R * tau), np.sin(R * tau) / Rs
    U11 = ph * (co - 1j * si * d)
    U12 = ph * (-1j * si * vc)
    U22 = ph * (co + 1j * si * d)
    return U11, U12, U22


def propagate_exact(psi1, psi2, x, k, mass, v1, v2, vc, dt, nsteps,
                    record_every=1):
    """Split-operator propagation.  Returns dict with t, pops (T,2), xmean
    (T,2 = <x> on each diabatic state), norm."""
    U11h, U12h, U22h = _expV(v1, v2, vc, dt / 2.0)
    Tk = np.exp(-1j * (k ** 2) / (2.0 * mass) * dt)
    dx = x[1] - x[0]

    ts, pops, xm, nrm = [], [], [], []
    t = 0.0
    for step in range(nsteps + 1):
        if step % record_every == 0:
            n1 = np.sum(np.abs(psi1) ** 2) * dx
            n2 = np.sum(np.abs(psi2) ** 2) * dx
            tot = n1 + n2
            ts.append(t); pops.append([n1 / tot, n2 / tot]); nrm.append(tot)
            xm.append([np.sum(np.abs(psi1) ** 2 * x) * dx / max(n1, 1e-300),
                       np.sum(np.abs(psi2) ** 2 * x) * dx / max(n2, 1e-300)])
        if step == nsteps:
            break
        a1 = U11h * psi1 + U12h * psi2            # half potential
        a2 = U12h * psi1 + U22h * psi2
        a1 = np.fft.ifft(Tk * np.fft.fft(a1))     # full kinetic
        a2 = np.fft.ifft(Tk * np.fft.fft(a2))
        psi1 = U11h * a1 + U12h * a2              # half potential
        psi2 = U12h * a1 + U22h * a2
        t += dt

    return dict(t=np.array(ts), pops=np.array(pops), xmean=np.array(xm),
                norm=np.array(nrm), psi1=psi1, psi2=psi2)


# ==========================================================================
#  Exact reference -- N states, D modes (small D only: cost n^D)
# ==========================================================================
def potential_matrix(model, x):
    """(..., N, N) diabatic potential matrix at grid points x (..., D)."""
    N = model["N"]
    V = np.zeros(x.shape[:-1] + (N, N))
    for i in range(N):
        if model["diag_kind"] == "separable":
            for k in range(model["D"]):
                V[..., i, i] += model["vfun"][i](k, x[..., k])
        else:
            H = state_hessian(model, i)
            V[..., i, i] = (model["E"][i] + x @ model["kappa"][i]
                            + 0.5 * np.einsum("...k,kl,...l->...", x, H, x))
        for j in range(N):
            if j == i:
                continue
            V[..., i, j] += model["J"][i, j] + x @ model["lam"][i, j]
            g = model.get("gauss")
            if g is not None:
                beta, xc = gauss_pair(g, i, j)
                V[..., i, j] += np.asarray(g["A"])[i, j] * np.exp(
                    -np.sum(beta * (x - xc) ** 2, axis=-1))
            gl = model.get("gauss_lin")
            if gl is not None:
                L = gl["lin_dim"]
                V[..., i, j] += (np.asarray(gl["A"])[i, j]
                                 * (x[..., L] - gl["xc"][L]) * np.exp(
                                     -np.sum(gl["beta"] * (x - gl["xc"]) ** 2,
                                             axis=-1)))
            qc = model.get("quad_coupling")
            if qc is not None and {i, j} == {qc["i"], qc["j"]}:
                V[..., i, j] += qc["v"](x[..., 0])
    return V


def propagate_exact_grid(model, psi0, axes, dt, nsteps, record_every=1,
                         snapshot_steps=()):
    """Split-operator propagation of the N-state TDSE on a tensor grid.

    psi0  : (N, n_1, ..., n_D) initial components
    axes  : list of D uniform 1-D grids
    Returns dict: t, pops (T,N), xmean (T,N,D), psi (final), snaps
    {step: psi}, plus X (grid points, (n_1..n_D, D)) and dV.
    """
    N, D = model["N"], model["D"]
    X = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    dV = np.prod([a[1] - a[0] for a in axes])
    ks = [2.0 * np.pi * np.fft.fftfreq(len(a), d=a[1] - a[0]) for a in axes]
    Kg = np.stack(np.meshgrid(*ks, indexing="ij"), axis=-1)
    Tk = np.exp(-1j * dt * np.sum(Kg ** 2 / (2.0 * model["mass"]), axis=-1))

    lam, U = np.linalg.eigh(potential_matrix(model, X))
    Uh = np.einsum("...ik,...k,...jk->...ij", U,
                   np.exp(-0.5j * dt * lam), U)            # exp(-i V dt/2)

    psi = np.moveaxis(np.asarray(psi0, complex), 0, -1)   # (..., N)
    fft_axes = tuple(range(D))
    ts, pops, xm, snaps = [], [], [], {}
    for step in range(nsteps + 1):
        if step % record_every == 0:
            rho = np.abs(psi) ** 2                          # (..., N)
            n = np.sum(rho, axis=fft_axes) * dV
            ts.append(step * dt); pops.append(n / n.sum())
            xm.append(rho.reshape(-1, N).T @ X.reshape(-1, D) * dV
                      / np.maximum(n, 1e-300)[:, None])
        if step in snapshot_steps:
            snaps[step] = np.moveaxis(psi, -1, 0).copy()
        if step == nsteps:
            break
        psi = np.einsum("...ij,...j->...i", Uh, psi)
        psi = np.fft.ifftn(Tk[..., None] * np.fft.fftn(psi, axes=fft_axes),
                           axes=fft_axes)
        psi = np.einsum("...ij,...j->...i", Uh, psi)
    return dict(t=np.array(ts), pops=np.array(pops), xmean=np.array(xm),
                psi=np.moveaxis(psi, -1, 0), snaps=snaps, X=X, dV=dV)
