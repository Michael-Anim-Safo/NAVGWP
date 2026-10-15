"""
initialize.py -- model definitions and initial conditions.

A MODEL is a plain dict describing an N-state, D-dimensional problem:

    N, D          number of electronic states / nuclear modes
    mass   (D,)   masses (1 for mass-weighted coordinates)

  DIAGONAL potential, either
    diag_kind = 'quadratic'        (exact, closed form)
        E      (N,)     state energies
        kappa  (N,D)    linear coupling of mode k to state i
        omega2 (N,D)    squared harmonic frequencies
    or
    diag_kind = 'separable'        (exact, 1-D Gauss-Hermite per mode)
        vfun   list of N callables v_i(k, x) -> V contribution of mode k
        dvfun  list of N callables dv_i(k, x)

  OFF-DIAGONAL coupling (all may be present, they add)
        J    (N,N)      constant
        lam  (N,N,D)    linear:  V_ij = J_ij + lam_ij . x
        gauss = dict(A=(N,N), beta=(D,), xc=(D,)) for
                V_ij += A_ij exp[-sum_k beta_k (x_k-xc_k)^2]
        gauss_lin = dict(A=(N,N), beta=(D,), xc=(D,), lin_dim=int) for
                V_ij += A_ij (x_{lin_dim}-xc_{lin_dim}) exp[-sum_k beta_k (x_k-xc_k)^2]
        quad_coupling = dict(i=int, j=int, v=callable, npts=int, margin=float)
                for a single pair with no closed form -- V_ij given as a
                callable, integrated numerically on a grid rebuilt each call
                around the current wavepacket positions. D=1 only.

Builders provided: lvc, spin_boson, fmo, tully1, tully2, tully3, holstein,
holstein_ref, ferretti.
Plain functions, no classes.
"""

import numpy as np

from .compute import coherent_zeta

__all__ = ["lvc", "separable", "spin_boson", "fmo", "tully1", "tully2", "tully3",
           "coronado", "coronado_x0", "coronado_harmonic", "villaseco", "villaseco_init",
           "holstein", "holstein_ref", "ferretti",
           "debye_bath", "suggest_dt", "boltzmann_sample", "model_widths",
           "CM1", "FS", "KB", "FMO7_CM1"]

CM1 = 1.0 / 219474.6313702        # cm^-1 -> hartree
FS = 0.02418884254                # a.u. of time -> fs
KB = 3.166811563e-6               # hartree / K


# --------------------------------------------------------------------------
#  Builders
# --------------------------------------------------------------------------
def lvc(E, kappa, omega2, J, lam=None, mass=None, gauss=None, gauss_lin=None,
        quad_coupling=None, name="LVC / system-bath model"):
    """Linear-vibronic-coupling / system-bath model (quadratic diagonal).

    gauss_lin : optional dict(A=(N,N), beta=(D,), xc=(D,), lin_dim=int) for
                V_ij += A_ij (x_{lin_dim} - xc_{lin_dim}) exp[-sum_k beta_k (x_k-xc_k)^2]
                -- a Gaussian-modulated coupling that is odd (vanishes) along
                the `lin_dim` axis at xc, e.g. the Ferretti conical-intersection
                model's H12 = gamma*Y*exp(-alpha(X-X3)^2)*exp(-beta*Y^2).
    quad_coupling : optional dict(i, j, v, npts, margin) for a single
                off-diagonal pair whose V_ij(x) has no closed form -- v is a
                callable, and the matrix elements are computed by numerical
                quadrature (on a grid rebuilt each call around the current
                wavepacket positions) instead of an analytic formula. D=1
                only. E.g. Tully III's asymmetric ECR coupling, which (unlike
                gauss/gauss_lin) doesn't vanish as x -> +-inf.
    """
    E = np.atleast_1d(np.asarray(E, float))
    N = len(E)
    kappa = np.atleast_2d(np.asarray(kappa, float))
    D = kappa.shape[1]
    omega2 = np.asarray(omega2, float)
    if omega2.ndim == 1:
        omega2 = np.tile(omega2, (N, 1))
    J = np.asarray(J, float).copy()
    np.fill_diagonal(J, 0.0)
    lam = np.zeros((N, N, D)) if lam is None else np.asarray(lam, float)
    mass = np.ones(D) if mass is None else np.asarray(mass, float)
    return dict(name=name, N=N, D=D, mass=mass, diag_kind="quadratic",
                E=E, kappa=kappa, omega2=omega2, J=J, lam=lam, gauss=gauss,
                gauss_lin=gauss_lin, quad_coupling=quad_coupling)


def separable(vfun, dvfun, N, D, J, lam=None, mass=None, gauss=None,
              omega2_ref=None, name="separable-diagonal model"):
    """Model whose diagonal potentials are separable but not polynomial,
    V_ii(x) = sum_k v_i(k, x_k).  Diagonal elements use 1-D Gauss-Hermite
    (exact); couplings stay analytic.  omega2_ref (N,D) is used only to pick
    default Gaussian widths."""
    mass = np.ones(D) if mass is None else np.asarray(mass, float)
    J = np.asarray(J, float).copy()
    np.fill_diagonal(J, 0.0)
    lam = np.zeros((N, N, D)) if lam is None else np.asarray(lam, float)
    omega2_ref = np.ones((N, D)) if omega2_ref is None else np.asarray(omega2_ref, float)
    return dict(name=name, N=N, D=D, mass=mass, diag_kind="separable",
                vfun=vfun, dvfun=dvfun, J=J, lam=lam, gauss=gauss,
                omega2=omega2_ref)


# --------------------------------------------------------------------------
def debye_bath(nmodes, lam, omega_c, omega_max_factor=20.0):
    """Discretise J(w) = 2 lam w wc/(w^2+wc^2) into `nmodes` modes.

    w_j = wc tan[(j-1/2) theta_max / n],  kappa_j = w_j sqrt(2 lam/n)

    The tangent mapping is truncated at omega_max_factor*omega_c: the
    discarded tail carries negligible weight but the admissible time step
    improves by orders of magnitude.
    """
    j = np.arange(1, nmodes + 1)
    theta_max = np.arctan(omega_max_factor)
    omega = omega_c * np.tan((j - 0.5) * theta_max / nmodes)
    kappa = omega * np.sqrt(2.0 * lam / nmodes)
    return omega, kappa


def spin_boson(eps, Delta, lam, omega_c, nmodes):
    """V_11/22 = +/-eps +/- kappa.x + w^2x^2/2 ;  V_12 = Delta."""
    w, k = debye_bath(nmodes, lam, omega_c)
    return lvc(E=[eps, -eps], kappa=np.vstack([k, -k]), omega2=w ** 2,
               J=[[0.0, Delta], [Delta, 0.0]],
               name="spin-boson (%d modes)" % nmodes)


FMO7_CM1 = np.array([
    [200, -87.7,   5.5,  -5.9,   6.7, -13.7,  -9.9],
    [-87.7, 320,  30.8,   8.2,   0.7,  11.8,   4.3],
    [5.5,  30.8,    0,  -53.5,  -2.2,  -9.6,   6.0],
    [-5.9,  8.2, -53.5,  110,  -70.7, -17.0, -63.3],
    [6.7,   0.7,  -2.2, -70.7,  270,   81.1,  -1.3],
    [-13.7, 11.8, -9.6, -17.0,  81.1,  420,   39.7],
    [-9.9,  4.3,   6.0, -63.3,  -1.3,  39.7,  230]], float)


def fmo(nsites=3, nmodes_per_site=20, lam_cm=35.0, tau_c_fs=50.0):
    """Frenkel-exciton FMO model; each site carries its OWN bath.
    D = nsites * nmodes_per_site.  Energies in hartree."""
    H = FMO7_CM1[:nsites, :nsites] * CM1
    E = np.diag(H).copy()
    J = H.copy(); np.fill_diagonal(J, 0.0)
    w, k = debye_bath(nmodes_per_site, lam_cm * CM1, 1.0 / (tau_c_fs / FS))
    D = nsites * nmodes_per_site
    kappa = np.zeros((nsites, D))
    omega2 = np.zeros(D)
    for n in range(nsites):
        sl = slice(n * nmodes_per_site, (n + 1) * nmodes_per_site)
        kappa[n, sl] = k
        omega2[sl] = w ** 2
    return lvc(E, kappa, omega2, J,
               name="FMO %d-site + bath (D=%d)" % (nsites, D))


def holstein(omega=1.0, kappa=1.0, V=0.5, eps=0.0, mass=1.0):
    """Two-site Holstein (molecular dimer) model, 1 vibrational mode.

        V_11 = +eps/2 + kappa q + omega^2 q^2 / 2
        V_22 = -eps/2 - kappa q + omega^2 q^2 / 2
        V_12 = V                                    (transfer integral)

    Each site is linearly coupled to the same local mode with opposite sign,
    so the two diabatic minima sit at q = -/+ kappa/omega^2.  Being 2 states x
    1 mode, this model can be compared with numerically exact grid
    propagation -- see examples/04_holstein.py.
    """
    return lvc(E=[eps / 2, -eps / 2],
               kappa=[[kappa], [-kappa]],
               omega2=[omega ** 2],
               J=[[0.0, V], [V, 0.0]],
               mass=np.array([mass]),
               name="Holstein dimer (w=%g, kappa=%g, V=%g)" % (omega, kappa, V))


def holstein_ref(k1=10.0, k2=10.0, x0=1.0, y0=15.0,
                 c1=1.0, c2=None, c3=2.0, mass=1.0):
    """Holstein model in the parametrisation of the reference Fortran source
    (model 3), with  am=1, ak1=ak2=10, aka=1, qs1=0, qs2=1, qc=2, v0=15,
    beta = sqrt(ak1*am)/2:

        V_11(x) = (k1/2) x^2                          (minimum at qs1 = 0)
        V_22(x) = (k2/2) (x - x0)^2 + y0              (minimum at qs2 = x0)
        V_12(x) = c1 exp[ -c2 (x - c3)^2 ]            (Gaussian, centred at qc)

    with c2 = beta = sqrt(k1*mass)/2 by default.

    Note the internal consistency of the reference parameters: the two diabatic
    parabolas cross where (k1/2)x^2 = (k2/2)(x-x0)^2 + y0, i.e. at x = 2 for
    k1 = k2 = 10, x0 = 1, y0 = 15 -- exactly the centre c3 = qc = 2 of the
    Gaussian coupling.  A coherent state released at rest from x = -2 therefore
    reaches the crossing precisely at its classical turning point.

    Mapping onto the internal LVC form V_ii = E_i + kappa_i x + omega2_i x^2/2:
        state 1: E=0,                    kappa=0,       omega2=k1
        state 2: E=k2 x0^2/2 + y0,       kappa=-k2 x0,  omega2=k2
    """
    if c2 is None:
        c2 = 0.5 * np.sqrt(k1 * mass)
    E = [0.0, 0.5 * k2 * x0 ** 2 + y0]
    kappa = [[0.0], [-k2 * x0]]
    omega2 = [[k1], [k2]]
    gauss = dict(A=np.array([[0.0, c1], [c1, 0.0]]),
                 beta=np.array([c2]), xc=np.array([c3]))
    return lvc(E, kappa, omega2, J=np.zeros((2, 2)), mass=np.array([mass]),
               gauss=gauss,
               name=("Holstein (ref params: k1=%g k2=%g x0=%g y0=%g "
                     "c1=%g c2=%.4f c3=%g)" % (k1, k2, x0, y0, c1, c2, c3)))


def ferretti(Kx=0.02, Ky=0.10, Delta=0.01, X1=4.0, X2=3.0, X3=3.0,
            gamma=0.01, alpha=3.0, beta=1.5, Mx=20000.0, My=6667.0):
    """Ferretti conical-intersection model (Ferretti, Granucci, Lami, Persico
    & Villani, J. Chem. Phys. 104, 5517 (1996)): two 2-D coupled harmonic
    diabatic surfaces in symmetric-stretch (X) / antisymmetric-stretch (Y)
    coordinates,

        H11(X,Y) = (Kx/2)(X-X1)^2 + (Ky/2) Y^2
        H22(X,Y) = (Kx/2)(X-X2)^2 + (Ky/2) Y^2 + Delta
        H12(X,Y) = gamma * Y * exp[-alpha (X-X3)^2] * exp[-beta Y^2]

    The two diabatic surfaces are identical paraboloids except for the
    X-displacement and the Delta offset; H12 is ODD in Y and vanishes
    identically along Y=0, which together with the intersection sitting at
    (X3, 0) makes this a true (not just avoided) crossing -- the coupling
    can't be turned into an avoided crossing by any 1-D cut through Y=0.

    X1, X2 : diabatic-well centres along X (state 1, state 2)
    X3     : centre of the Gaussian coupling envelope along X
    Mx, My : reduced masses for the X, Y coordinates

    Mapped onto lvc()'s V_ii = E_i + kappa_i.q + 0.5 sum omega2_i q^2 form:
        state 1: E = Kx X1^2/2,       kappa = (-Kx X1, 0),  omega2 = (Kx, Ky)
        state 2: E = Kx X2^2/2+Delta, kappa = (-Kx X2, 0),  omega2 = (Kx, Ky)
    and H12 via `gauss_lin` (lin_dim=1 selects Y as the odd coordinate).
    """
    E = [0.5 * Kx * X1 ** 2, 0.5 * Kx * X2 ** 2 + Delta]
    kappa = [[-Kx * X1, 0.0], [-Kx * X2, 0.0]]
    omega2 = [[Kx, Ky], [Kx, Ky]]
    gauss_lin = dict(A=np.array([[0.0, gamma], [gamma, 0.0]]),
                     beta=np.array([alpha, beta]), xc=np.array([X3, 0.0]),
                     lin_dim=1)
    return lvc(E, kappa, omega2, J=np.zeros((2, 2)), mass=np.array([Mx, My]),
               gauss_lin=gauss_lin,
               name=("Ferretti conical intersection (Kx=%g Ky=%g Delta=%g "
                     "X1=%g X2=%g X3=%g gamma=%g)"
                     % (Kx, Ky, Delta, X1, X2, X3, gamma)))


def _tully(A, B, C, Dd, dual, form="tanh"):
    """Shared builder for the Tully models (1 mode, 2 states)."""
    if dual:                                  # Tully II
        E0 = 0.05
        def v0(k, x): return np.zeros_like(np.asarray(x, float))
        def v1(k, x): return -A * np.exp(-B * np.asarray(x, float) ** 2) + E0
        def d0(k, x): return np.zeros_like(np.asarray(x, float))
        def d1(k, x):
            x = np.asarray(x, float)
            return 2.0 * A * B * x * np.exp(-B * x ** 2)
        nm = "Tully II (dual avoided crossing)"
    elif form == "tanh":                      # Tully I, smooth tanh form
        def v0(k, x): return A * (1.0 + np.tanh(B * np.asarray(x, float)))
        def v1(k, x): return A * (1.0 - np.tanh(B * np.asarray(x, float)))
        def d0(k, x):
            return A * B / np.cosh(B * np.asarray(x, float)) ** 2
        def d1(k, x): return -d0(k, x)
        nm = "Tully I (single avoided crossing, tanh)"
    elif form == "piecewise":                 # Tully I, original 1990 form
        def v0(k, x):
            x = np.asarray(x, float)
            return np.where(x >= 0, A * (1 - np.exp(-B * x)),
                            -A * (1 - np.exp(B * x)))
        def v1(k, x): return -v0(k, x)
        def d0(k, x):
            return A * B * np.exp(-B * np.abs(np.asarray(x, float)))
        def d1(k, x): return -d0(k, x)
        nm = "Tully I (single avoided crossing, piecewise)"
    else:
        raise ValueError("form must be 'tanh' or 'piecewise'")
    gauss = dict(A=np.array([[0.0, C], [C, 0.0]]),
                 beta=np.array([Dd]), xc=np.array([0.0]))
    return separable([v0, v1], [d0, d1], N=2, D=1,
                     J=np.zeros((2, 2)), mass=np.array([2000.0]),
                     gauss=gauss, omega2_ref=np.ones((2, 1)), name=nm)


def tully1(A=0.01, B=1.147, C=0.005, D=1.0, form="tanh"):
    """Tully I -- single avoided crossing (SAC).

    form='tanh' (default, smooth derivatives everywhere):
        V11 = A (1 + tanh(B x)),  V22 = A (1 - tanh(B x)),
        V12 = V21 = C exp(-D x^2),  A=0.01 B=1.147 C=0.005 D=1.0
    form='piecewise' (Tully, J. Chem. Phys. 93, 1061 (1990); use B=1.6):
        V11 = sign(x) A (1 - exp(-B |x|)),  V22 = -V11
    """
    return _tully(A, B, C, D, dual=False, form=form)


def tully2(A=0.10, B=0.28, C=0.015, D=0.06):
    return _tully(A, B, C, D, dual=True)


_CORONADO = {
    # Coronado, Xing & Miller, Chem. Phys. Lett. 349, 521 (2001), Tables 1-3.
    # diabats V_ii = De (1 - exp[-b (x - Re)])^2 + c   : (De, b, Re, c)
    # couplings V_ij = A exp[-a (x - R)^2]             : {(i, j): (A, R, a)}
    # x0 = ground-state Re, where the packet starts on state 1
    1: dict(diab=[(0.003, 0.65, 5.0, 0.00), (0.004, 0.60, 4.0, 0.01),
                  (0.003, 0.65, 6.0, 0.006)],
            cpl={(0, 1): (0.002, 3.40, 16.0), (1, 2): (0.002, 4.80, 16.0)},
            x0=2.9),
    2: dict(diab=[(0.02, 0.65, 4.5, 0.00), (0.01, 0.40, 4.0, 0.01),
                  (0.003, 0.65, 4.4, 0.02)],
            cpl={(0, 1): (0.005, 3.66, 32.0), (0, 2): (0.005, 3.34, 32.0)},
            x0=3.3),
    3: dict(diab=[(0.02, 0.40, 4.0, 0.02), (0.02, 0.65, 4.5, 0.00),
                  (0.003, 0.65, 6.0, 0.02)],
            cpl={(0, 1): (0.005, 3.40, 32.0), (0, 2): (0.005, 4.97, 32.0)},
            x0=2.1),
}


def coronado(variant=1, mass=20000.0, omega_ground=5e-3):
    """Three-state photodissociation models I, II, III of Coronado, Xing &
    Miller, Chem. Phys. Lett. 349, 521 (2001) (1 mode, 3 states).

        V_ii(x) = De_i (1 - exp[-b_i (x - Re_i)])^2 + c_i        (Morse)
        V_ij(x) = A_ij exp[-a_ij (x - R_ij)^2]                    (Gaussian)

    The couplings sit at DIFFERENT crossing points, so gauss["beta"] and
    gauss["xc"] are given per pair, shape (N, N, D).  mass = 20000 and the
    ground-state frequency omega = 0.005 set the default (coherent) width:
    omega2_ref = m omega^2, so default_Z = m omega = 100, the initial
    ground-state packet of the paper.  The packet starts on state 1 at
    x0 = coronado_x0(variant) (2.9, 3.3, 2.1 for I, II, III).
    """
    if variant not in _CORONADO:
        raise ValueError("variant must be 1, 2 or 3 (models I, II, III)")
    par = _CORONADO[variant]
    N = 3

    def make(De, b, Re, c):
        def v(k, x):
            e = np.exp(-b * (np.asarray(x, float) - Re))
            return De * (1.0 - e) ** 2 + c
        def dv(k, x):
            e = np.exp(-b * (np.asarray(x, float) - Re))
            return 2.0 * De * b * e * (1.0 - e)
        return v, dv
    funcs = [make(*d) for d in par["diab"]]

    A = np.zeros((N, N))
    beta = np.ones((N, N, 1))
    xc = np.zeros((N, N, 1))
    for (i, j), (Aij, R, a) in par["cpl"].items():
        for (k, l) in ((i, j), (j, i)):
            A[k, l], beta[k, l, 0], xc[k, l, 0] = Aij, a, R
    gauss = dict(A=A, beta=beta, xc=xc)
    model = separable([f[0] for f in funcs], [f[1] for f in funcs], N=N, D=1,
                      J=np.zeros((N, N)), mass=np.array([mass]), gauss=gauss,
                      omega2_ref=np.full((N, 1), mass * omega_ground ** 2),
                      name="Coronado model %s (3 states, Morse)"
                           % {1: "I", 2: "II", 3: "III"}[variant])
    # exact Gaussian averages of the Morse diabats (compute.diagonal_terms)
    model["morse"] = np.array([[d] for d in par["diab"]], float)   # (N, 1, 4)
    return model


_PIERONI = {
    # Pieroni & Agostini, J. Chem. Theory Comput. 17, 5969 (2021), Table 1:
    # (b, R3, Delta) with k=0.02, a=3.0, R1=6.0, R2=2.0, M=20000 fixed.
    # R0 and which state the packet starts in are from their Sec. 4.
    # E_r = k (R1-R2)^2 / 2 = 0.16 is the reorganisation energy; Delta < E_r
    # is Marcus' direct regime (I-III), Delta > E_r the inverted one (IV, V).
    1: dict(b=0.010, R3=3.875, Delta=0.01, R0=2.0,  start="excited"),
    2: dict(b=0.005, R3=3.875, Delta=0.01, R0=2.0,  start="excited"),
    3: dict(b=0.010, R3=3.875, Delta=0.00, R0=-2.5, start="ground"),
    4: dict(b=0.010, R3=1.5,   Delta=0.20, R0=6.0,  start="excited"),
    5: dict(b=0.005, R3=1.5,   Delta=0.20, R0=6.0,  start="excited"),
}


def villaseco(variant=1, k=0.02, a=3.0, R1=6.0, R2=2.0, mass=20000.0,
              b=None, R3=None, Delta=None):
    """Two shifted harmonic diabats with a Gaussian coupling (1 mode, 2 states):

        V_11(R) = k (R - R1)^2 / 2
        V_22(R) = k (R - R2)^2 / 2 + Delta
        V_12(R) = b exp[-a (R - R3)^2]

    Models I-V of Pieroni & Agostini, J. Chem. Theory Comput. 17, 5969 (2021),
    Table 1, selected by `variant`; variant 1 (Model I) is also the model of
    Villaseco Arribas, Maitra & Agostini, J. Chem. Phys. 160, 054102 (2024).
    b, R3 and Delta override the table if given.

        variant  b      R3     Delta   regime (E_r = 0.16)
        1        0.010  3.875  0.01    direct, weakly nonadiabatic
        2        0.005  3.875  0.01    direct, strongly nonadiabatic
        3        0.010  3.875  0.00    direct, degenerate minima
        4        0.010  1.5    0.20    inverted, weak coupling
        5        0.005  1.5    0.20    inverted, strong coupling

    Both diabats have curvature k, so both states share the coherent width
    sqrt(m k) = 20 -- the papers' initial packet, exp[-(R-R0)^2/(2 sigma^2)]
    with sigma = 0.223 (zeta = 1/sigma^2 = 20), which is what Z0=None gives.
    Use villaseco_init() for each variant's starting point and state.
    """
    if variant not in _PIERONI:
        raise ValueError("variant must be 1..5 (Models I-V)")
    par = _PIERONI[variant]
    b = par["b"] if b is None else b
    R3 = par["R3"] if R3 is None else R3
    Delta = par["Delta"] if Delta is None else Delta
    E = [0.5 * k * R1 ** 2, 0.5 * k * R2 ** 2 + Delta]
    kappa = [[-k * R1], [-k * R2]]
    omega2 = [[k], [k]]
    gauss = dict(A=np.array([[0.0, b], [b, 0.0]]),
                 beta=np.array([a]), xc=np.array([R3]))
    return lvc(E, kappa, omega2, J=np.zeros((2, 2)), mass=np.array([mass]),
               gauss=gauss,
               name="Pieroni-Agostini model %s (b=%g R3=%g Delta=%g)"
                    % ({1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}[variant],
                       b, R3, Delta))


def villaseco_init(model, variant=1):
    """(R0, c0) for one variant: where the packet starts and which DIABAT
    carries it.

    The papers start the packet in a single ADIABATIC state at R0 -- the
    excited one for Models I, II, IV and V (photoexcitation), the ground one
    for Model III.  At every one of those R0 the Gaussian coupling is
    negligible (the crossing is far away), so that adiabatic state is one
    diabat to better than 1e-5, and the diabatic c0 is a unit vector.  The
    assignment is computed here by diagonalising V(R0) rather than assumed.
    """
    par = _PIERONI[variant]
    R0 = par["R0"]
    x = np.array([float(R0)])
    V = np.zeros((2, 2))
    for i in range(2):
        V[i, i] = (model["E"][i] + model["kappa"][i] @ x
                   + 0.5 * float(model["omega2"][i] @ (x ** 2)))
    g = model["gauss"]
    V[0, 1] = V[1, 0] = float(np.asarray(g["A"])[0, 1] * np.exp(
        -np.sum(np.asarray(g["beta"], float) * (x - np.asarray(g["xc"], float)) ** 2)))
    w, U = np.linalg.eigh(V)
    col = 1 if par["start"] == "excited" else 0          # upper / lower adiabat
    i_diab = int(np.argmax(np.abs(U[:, col])))           # the diabat it is made of
    c0 = np.zeros(2)
    c0[i_diab] = 1.0
    return float(R0), c0


def coronado_harmonic(variant=1, b3_scale=0.5, fit="taylor", window=(2.5, 7.0),
                      mass=20000.0, omega_ground=5e-3, recentre_couplings=True):
    """A simplified stand-in for coronado(): states 1 and 2 become HARMONIC
    wells, state 3 stays a Morse made less steep (1 mode, 3 states).

        V_11, V_22 = harmonic fit of the Morse diabat (see `fit`)
        V_33       = De_3 (1 - exp[-b_3' (x - Re_3)])^2 + c_3,  b_3' = b3_scale b_3
        V_ij       = A_ij exp[-a_ij (x - xc_ij)^2]      as in coronado()

    `fit` chooses how each Morse well is replaced by a parabola:

      'minimum'  curvature at the Morse minimum, k = 2 De b^2: keeps that
                 state's frequency, zero-point energy and coherent width
                 exactly, but is far too low on the repulsive wall.  For
                 model I this DESTROYS the 1-2 crossing (3.36 -> no crossing
                 at positive x), because that crossing sits high on state 1's
                 wall where a parabola cannot reach.
      'lsq'      least-squares fit over `window`, the range the packet
                 actually visits.  Keeps both crossings to ~0.25 bohr at the
                 cost of the frequency at the bottom.
      'taylor'   (default) second-order Taylor expansion about the published
                 coupling centre of the first pair that state belongs to:
                 reproduces that pair's crossing exactly (model I: 1-2 stays
                 at 3.36).  With b3_scale=1 this is the closest stand-in for
                 coronado() measured here -- its exact populations differ
                 from the real Morse model's by 0.036.

    b3_scale < 1 makes state 3's walls less steep and its well wider
    (b3_scale = 1 leaves it as in coronado()).  With `recentre_couplings` each
    Gaussian coupling is moved to the crossing of the APPROXIMATED diabats,
    since the fit shifts where the curves meet; set False to keep the
    published centres.  A_ij and a_ij are unchanged.

    Initial width as in coronado(): omega2_ref = m omega_ground^2, so
    default_Z = m omega_ground = 100.
    """
    if variant not in _CORONADO:
        raise ValueError("variant must be 1, 2 or 3 (models I, II, III)")
    if fit not in ("minimum", "lsq", "taylor"):
        raise ValueError("fit must be 'minimum', 'lsq' or 'taylor'")
    par = _CORONADO[variant]
    N = 3
    diab = [tuple(d) for d in par["diab"]]
    De3, b3, Re3, c3 = diab[2]
    b3 = b3_scale * b3

    def V_morse(i, x):
        De, b, Re, c = diab[i]
        return De * (1.0 - np.exp(-b * (np.asarray(x, float) - Re))) ** 2 + c

    def first_centre(i):
        for (a_, b_), (_, R, _a) in par["cpl"].items():
            if i in (a_, b_):
                return R
        return diab[i][2]

    def quad_coeffs(i):
        """(k, Re_eff, c_eff) of the parabola c_eff + k (x - Re_eff)^2 / 2."""
        De, b, Re, c = diab[i]
        if fit == "minimum":
            return 2.0 * De * b ** 2, Re, c
        if fit == "lsq":
            xw = np.linspace(window[0], window[1], 2000)
            A = np.vstack([np.ones_like(xw), xw, xw ** 2]).T
            c0, c1, c2 = np.linalg.lstsq(A, V_morse(i, xw), rcond=None)[0]
            return 2.0 * c2, -c1 / (2.0 * c2), c0 - c1 ** 2 / (4.0 * c2)
        x0 = first_centre(i)                       # 'taylor'
        e = np.exp(-b * (x0 - Re))
        V0 = De * (1.0 - e) ** 2 + c
        V1 = 2.0 * De * b * e * (1.0 - e)
        V2 = 2.0 * De * b * b * (2.0 * e * e - e)
        return V2, x0 - V1 / V2, V0 - V1 ** 2 / (2.0 * V2)

    def harmonic(k, Re, c):
        def v(_, x): return c + 0.5 * k * (np.asarray(x, float) - Re) ** 2
        def dv(_, x): return k * (np.asarray(x, float) - Re)
        return v, dv

    def morse(De, b, Re, c):
        def v(_, x):
            e = np.exp(-b * (np.asarray(x, float) - Re))
            return De * (1.0 - e) ** 2 + c
        def dv(_, x):
            e = np.exp(-b * (np.asarray(x, float) - Re))
            return 2.0 * De * b * e * (1.0 - e)
        return v, dv

    pars = [quad_coeffs(0), quad_coeffs(1)]
    funcs = [harmonic(*pars[0]), harmonic(*pars[1]), morse(De3, b3, Re3, c3)]

    A = np.zeros((N, N))
    beta = np.ones((N, N, 1))
    xc = np.zeros((N, N, 1))
    xs = np.linspace(-5.0, 40.0, 450001)
    for (i, j), (Aij, R, a) in par["cpl"].items():
        centre = R
        if recentre_couplings:
            dif = funcs[i][0](0, xs) - funcs[j][0](0, xs)
            roots = xs[:-1][np.sign(dif[:-1]) != np.sign(dif[1:])]
            if len(roots):                 # the crossing nearest the published one
                centre = float(roots[np.argmin(np.abs(roots - R))])
        for (k_, l_) in ((i, j), (j, i)):
            A[k_, l_], beta[k_, l_, 0], xc[k_, l_, 0] = Aij, a, centre
    gauss = dict(A=A, beta=beta, xc=xc)

    model = separable([f[0] for f in funcs], [f[1] for f in funcs], N=N, D=1,
                      J=np.zeros((N, N)), mass=np.array([mass]), gauss=gauss,
                      omega2_ref=np.full((N, 1), mass * omega_ground ** 2),
                      name="Coronado model %s approximated (2 harmonic [%s] "
                           "+ 1 Morse, b3 x %g)"
                           % ({1: "I", 2: "II", 3: "III"}[variant], fit, b3_scale))
    # exact closed forms for the Morse state; NaN marks the harmonic ones,
    # which fall back to Gauss-Hermite in compute.diagonal_terms
    m = np.full((N, 1, 4), np.nan)
    m[2, 0] = (De3, b3, Re3, c3)
    model["morse"] = m
    model["harmonic"] = np.array(pars)          # (k, Re_eff, c_eff) per state
    return model


def coronado_x0(variant=1):
    """Starting position (ground-state Re) for Coronado model I, II, III."""
    return _CORONADO[variant]["x0"]


def tully3(A=0.0006, B=0.10, C=0.90, npts=150, margin=15.0):
    """Tully model C -- extended coupling with reflection (ECR).
    Tully, J. Chem. Phys. 93, 1061 (1990):

        V11 = A
        V22 = -A
        V12 = B exp(C x)          x < 0
        V12 = B (2 - exp(-C x))   x >= 0

    Unlike SAC (tully1) and DAC (tully2), V12 does NOT vanish as x -> +inf
    (it saturates to 2B) -- the "extended coupling region" this model is
    named for. That shape isn't a Gaussian or Gaussian*linear, so it can't
    use the closed-form `gauss`/`gauss_lin` coupling; it's handled via
    `quad_coupling` (numerical quadrature, on a grid rebuilt each call
    around wherever the wavepacket currently is -- see compute.coupling_terms)
    instead. npts=150 was confirmed (by direct comparison against npts=4000)
    to already be far finer than needed -- accuracy here is limited by dt,
    not grid resolution; see suggest_dt's extra safety margin for quad_coupling
    models.
    """
    E = [A, -A]
    kappa = [[0.0], [0.0]]
    omega2 = [[0.0], [0.0]]
    v12 = lambda x: np.where(x < 0, B * np.exp(C * x), B * (2.0 - np.exp(-C * x)))
    quad_coupling = dict(i=0, j=1, v=v12, npts=npts, margin=margin)
    return lvc(E, kappa, omega2, J=np.zeros((2, 2)), mass=np.array([2000.0]),
               quad_coupling=quad_coupling,
               name="Tully III (extended coupling region, A=%g B=%g C=%g)"
                    % (A, B, C))


# --------------------------------------------------------------------------
#  Helpers
# --------------------------------------------------------------------------
def model_widths(model, width=None):
    """Default initial widths, shape (N,D) complex.
    Uses the coherent width sqrt(m)*omega where a frequency is available,
    else 1.0 (or the user-supplied scalar/array `width`)."""
    N, D = model["N"], model["D"]
    if width is not None:
        return np.broadcast_to(np.asarray(width, complex), (N, D)).copy()
    w = np.sqrt(np.maximum(np.asarray(model["omega2"], float), 0.0))
    z = coherent_zeta(model["mass"], w)
    z = np.where(z > 1e-12, z, 1.0)
    return np.broadcast_to(z, (N, D)).astype(complex).copy()


def suggest_dt(model, safety=0.02):
    """Time step resolving the fastest oscillation present."""
    w_max = float(np.sqrt(np.max(model["omega2"])))
    j_max = float(np.max(np.abs(model["J"]))) if model["J"].size else 0.0
    g = model.get("gauss")
    a_max = float(np.max(np.abs(g["A"]))) if g else 0.0
    gl = model.get("gauss_lin")
    al_max = float(np.max(np.abs(gl["A"]))) if gl else 0.0
    qc = model.get("quad_coupling")
    qc_max = float(np.max(np.abs(qc["v"](np.linspace(-qc.get("margin", 15.0),
                                                      qc.get("margin", 15.0), 400))))) if qc else 0.0
    E = model.get("E")
    e_span = float(np.max(E) - np.min(E)) if E is not None else 0.0
    rate = max(w_max, j_max, a_max, al_max, qc_max, e_span, 1e-12)
    dt = safety * 2.0 * np.pi / rate
    if qc is not None:
        # empirically, a quad_coupling region with a broad/strong, non-smooth
        # coupling (e.g. Tully III's ECR) needs a MUCH finer dt during the
        # actual nonadiabatic crossing than this generic rate-based estimate
        # suggests -- coarser dt there doesn't diverge outright but silently
        # gives wrong (non-converged) transfer probabilities, and can still
        # blow up for strong/near-complete transfer. Found by direct dt
        # convergence testing (population converged to <0.1% only once dt was
        # ~15-20x finer than this formula's baseline estimate); 20x is a
        # conservative margin over that.
        dt /= 20.0
    return dt


def boltzmann_sample(model, kT, rng=None):
    """Classical Boltzmann sample of the uncoupled harmonic bath.
    Returns (q, p) each shape (D,).  Modes with omega = 0 are left at 0."""
    rng = np.random.default_rng() if rng is None else rng
    w2 = np.asarray(model["omega2"], float).mean(axis=0)
    m = model["mass"]
    sq = np.where(w2 > 1e-30, np.sqrt(kT / (m * np.where(w2 > 1e-30, w2, 1))), 0.0)
    sp = np.sqrt(kT * m)
    return rng.normal(0.0, sq), rng.normal(0.0, sp)
