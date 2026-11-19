#!/usr/bin/env python3
"""
nonscattering_branch.py
=======================

Numerical computation of the branch of non-circular non-scattering domains
bifurcating from the resonant disk, as constructed in

    F. Cakoni and M. Levitin, "A family of homogeneous convex non-scattering
    domains with analytic non-circular boundaries".

The branch is that of Theorem 1.2; the script also checks the expansions
(1.15), (1.16) of Proposition 1.4.  The data it writes are those used for
Figure 1 (the figure itself is produced separately).

Theorem and equation numbers refer to the version of the paper dated
19 November 2026.

WHAT IS COMPUTED
----------------
For each eps we look for a domain

    Omega_eps = { r (cos t, sin t) : 0 <= r < rho(t) },
    rho(t)    = 1 + eps cos 2t + sum_{j=2}^{N} a_j cos 2jt ,            (*)

and wave numbers k (exterior) and kappa (interior), coefficient
sigma = kappa^2 / k^2, such that Omega_eps with constant coefficient sigma
does not scatter the radial Herglotz wave (1.10),

    v(x) = J_0(k |x|).

The normalisation in (*) is (1.14): mean polar radius equal to 1, cos 2t
coefficient equal to eps.  Only cos 2jt modes appear because the domain is
symmetric in both coordinate axes.

THE EQUATIONS
-------------
Omega is non-scattering for v iff there is w satisfying (1.11),

    (Delta + kappa^2) w = (k^2 - kappa^2) v  in Omega,   w = d_nu w = 0 on dOmega.

By Lemma 3.2 this is equivalent to the integral identities (3.3), which for
a domain symmetric in both axes take the form (3.5),

    M_m = < cos(mt) F_m(rho(t); k, kappa) > = 0,     m = 0, 2, 4, ...

We use them in an equivalent boundary form.  Pairing with any entire solution
phi of (Delta + kappa^2) phi = 0 and using Green's formula gives

    Q[phi] := int_{dOmega} ( phi d_nu v - v d_nu phi ) ds
            = (kappa^2 - k^2) int_Omega v phi dx = 0,

and for the test functions

    phi_m(r, t) = J_m(kappa r) cos(m t),     m = 0, 2, 4, ...

one has  Q_m := Q[phi_m] = 2 pi (kappa^2 - k^2) M_m.

On the polar curve r = rho(t) one has  nu ds = (rho e_r - rho' e_t) dt,  so

    d_nu f ds = ( rho f_r - (rho'/rho) f_t ) dt ,

and therefore

    Q_m = int_0^{2 pi} {  cos(mt) * rho * [ v_r J_m(kappa rho) - v kappa J_m'(kappa rho) ]
                        - m sin(mt) * (rho'/rho) * v J_m(kappa rho) } dt ,

with v = J_0(k rho), v_r = -k J_1(k rho).

DISCRETISATION
--------------
Unknowns  : a_2, ..., a_N, k, kappa        (N + 1 numbers)
Equations : Q_m = 0 for m = 0, 2, ..., 2N  (N + 1 equations)

The t-integral is done by the trapezoidal rule with M equispaced nodes, which
is spectrally accurate for periodic analytic integrands.  The system is solved
by Newton's method with the exact (analytically differentiated) Jacobian of
the discretised equations; the branch is followed from the disk by stepping in eps
with a secant predictor.

THE SEED (eps = 0)
------------------
The unit disk is non-scattering for J_0(k r) iff (1.6) holds,
d_0(k) = d_0(kappa), where d_0(t) = t J_0'(t) / J_0(t).  The branch bifurcates
where, in addition, the angular order 2 is Dirichlet-resonant; these are the
parameters (1.12), well defined by Lemma 2.1, with numerical values (1.13):

    kappa_* = j_{2,1}  (first positive zero of J_2),
    k_*     in (0, j_{0,1})  with  d_0(k_*) = d_0(kappa_*).

CAVEATS
-------
* This is floating-point continuation, NOT a certified computation.  Small
  residuals of a truncated system do not prove non-scattering; existence comes
  from the analytical construction (Theorem 1.2).
* The smooth convex branch ends near eps_0 ~ 0.0505 (cf. Remark 1.3), where
  the tip of the domain reaches the first nodal circle of J_0(k|x|).  Fourier
  coefficients behave like a_j ~ (7 eps)^j, so for eps close to eps_0 increase
  N and M and distrust the output.  eps = 0.04 (Figure 1) is safely inside.

USAGE
-----
    python3 nonscattering_branch.py                  # eps = 0.04, N = 24, M = 256
    python3 nonscattering_branch.py --eps 0.03 --N 32 --M 320 --outdir out
    python3 nonscattering_branch.py --check          # also run a truncation check

OUTPUT (in --outdir, default: current directory)
------------------------------------------------
    nonscattering_branch.csv     one row per continuation step:
                                 eps, k, kappa, sigma, a_2, ..., a_N

Each row determines the corresponding boundary exactly through (*); the last
row is the domain at the target eps.  No separate boundary file is written.

Requires numpy and scipy.
"""

import argparse
import os

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, jn_zeros, jv


# ----------------------------------------------------------------------------
# 1. The resonant disk from which the branch bifurcates
# ----------------------------------------------------------------------------

def d0(t):
    """d_0(t) = t J_0'(t) / J_0(t) = -t J_1(t) / J_0(t), see (1.6)."""
    return -t * jv(1, t) / jv(0, t)


def resonant_seed():
    """Return (k_*, kappa_*) for the resonant unit disk, see (1.12), (1.13).

    kappa_* = j_{2,1};  k_* is the root of d_0(k) = d_0(kappa_*) in (0, j_{0,1}).
    On (0, j_{0,1}) the function d_0 decreases monotonically from 0 to -infinity,
    so the root is unique (Lemma 2.1).
    """
    kappa_star = jn_zeros(2, 1)[0]
    j01 = jn_zeros(0, 1)[0]
    k_star = brentq(lambda k: d0(k) - d0(kappa_star), 1e-6, j01 - 1e-9, xtol=1e-15)
    return k_star, kappa_star


# ----------------------------------------------------------------------------
# 2. The discretised moment conditions Q_m = 0
# ----------------------------------------------------------------------------

class MomentSystem:
    """The nonlinear system F(eps, x) = 0, x = (a_2, ..., a_N, k, kappa).

    N : highest retained mode cos(2 N t) in the polar radius
    M : number of trapezoidal nodes on [0, 2 pi)
    """

    def __init__(self, N, M):
        self.N, self.M = N, M
        self.k_star, self.kappa_star = resonant_seed()

        t = 2 * np.pi * np.arange(M) / M                 # quadrature nodes
        self.t = t
        self.j = np.arange(2, N + 1)                     # indices of unknown a_j
        self.cos2j = np.cos(2 * np.outer(self.j, t))     # cos(2 j t), shape (N-1, M)
        self.sin2j = np.sin(2 * np.outer(self.j, t))
        self.cos2, self.sin2 = np.cos(2 * t), np.sin(2 * t)

        # test orders m = 0, 2, ..., 2N, as a column for broadcasting
        self.m = (2 * np.arange(N + 1))[:, None]
        self.cosm = np.cos(self.m * t)
        self.sinm = np.sin(self.m * t)

        # Row scaling.  J_m(kappa rho) ~ (kappa/2)^m / m!, so the raw Q_m span
        # many orders of magnitude.  Multiplying Q_m by m! / (kappa_*/2)^m
        # makes all equations O(1); it does not change the solution set.
        mm = self.m[:, 0]
        self.scale = np.exp(gammaln(mm + 1) - mm * np.log(self.kappa_star / 2))

    # -- geometry -------------------------------------------------------------
    def radius(self, eps, a):
        """rho(t) and rho'(t) at the quadrature nodes."""
        rho = 1 + eps * self.cos2 + a @ self.cos2j
        drho = -2 * eps * self.sin2 - (2 * self.j * a) @ self.sin2j
        return rho, drho

    # -- residual and Jacobian ------------------------------------------------
    def F_and_jacobian(self, eps, x):
        """Scaled moments F = (Q_0, Q_2, ..., Q_{2N}) at the point x, and the
        exact Jacobian dF/dx, both by the same trapezoidal rule.

        Write g for the integrand of Q_m (see the module docstring), as a
        function of rho, rho' (at each node) and of k, kappa:

            g = cos(mt) rho [ v_r J_m - v kappa J_m' ]  -  m sin(mt) (rho'/rho) v J_m ,

        with v = J_0(k rho), v_r = -k J_1(k rho) and J_m, J_m' evaluated at
        z = kappa rho.  The Jacobian is obtained by differentiating g under
        the integral sign.  Second derivatives of Bessel functions are
        eliminated with Bessel's equation,

            J_m''(z) = -J_m'(z)/z - (1 - m^2/z^2) J_m(z).
        """
        N, m = self.N, self.m
        a, k, kappa = x[:N - 1], x[N - 1], x[N]
        rho, drho = self.radius(eps, a)

        # incident wave on the boundary: v, dv/dr, d^2v/dr^2
        J0, J1 = jv(0, k * rho), jv(1, k * rho)
        v = J0
        vr = -k * J1
        vrr = -k**2 * (J0 - J1 / (k * rho))

        # J_m, J_m', J_m'' at z = kappa rho, for all test orders (rows) and nodes
        z = kappa * rho
        Jm = jv(m, z)
        dJm = 0.5 * (jv(m - 1, z) - jv(m + 1, z))
        ddJm = -dJm / z - (1 - m**2 / z**2) * Jm

        c, s = self.cosm, self.sinm
        q = drho / rho

        # the integrand
        g = c * rho * (vr * Jm - v * kappa * dJm) - m * s * q * v * Jm

        # partial derivatives of g
        g_rho = (c * (vr * Jm - v * kappa * dJm)
                 + c * rho * (vrr * Jm - v * kappa**2 * ddJm)
                 - m * s * q * (-v * Jm / rho + vr * Jm + v * kappa * dJm))
        g_drho = -m * s * v * Jm / rho
        v_k = rho * vr / k                      # dv/dk
        vr_k = vr / k + rho * vrr / k           # d(v_r)/dk
        g_k = c * rho * (vr_k * Jm - v_k * kappa * dJm) - m * s * q * v_k * Jm
        g_kappa = (c * rho * (vr * rho * dJm - v * (dJm + z * ddJm))
                   - m * s * q * v * rho * dJm)

        # trapezoidal rule: int_0^{2 pi} f dt = (2 pi / M) * sum over nodes
        w = 2 * np.pi / self.M
        F = w * g.sum(axis=1)
        J = np.empty((N + 1, N + 1))
        # d rho / d a_j = cos(2jt),   d rho' / d a_j = -2j sin(2jt)
        J[:, :N - 1] = w * (g_rho @ self.cos2j.T - g_drho @ (2 * self.j[:, None] * self.sin2j).T)
        J[:, N - 1] = w * g_k.sum(axis=1)
        J[:, N] = w * g_kappa.sum(axis=1)

        # apply the row scaling
        return F * self.scale, J * self.scale[:, None]

    def F(self, eps, x):
        """Scaled moments only."""
        return self.F_and_jacobian(eps, x)[0]

    # -- Newton ---------------------------------------------------------------
    def newton(self, eps, x, tol=1e-13, maxit=30):
        """Solve F(eps, x) = 0 from the initial guess x.

        Returns (x, size of last Newton update, max |F|).
        """
        dx = np.inf
        for _ in range(maxit):
            F, J = self.F_and_jacobian(eps, x)
            step = np.linalg.solve(J, -F)
            x = x + step
            dx = np.abs(step).max()
            if dx < tol:
                break
        return x, dx, np.abs(self.F(eps, x)).max()

    # -- small-eps asymptotics, used only as the first initial guess ----------
    def asymptotic_guess(self, eps):
        """x from the expansion (1.15) of Proposition 1.4:

            a_2 = (3/4 + kappa_*^2/12) eps^2 + ...,

        with k and kappa at their seed values (they change at order eps^2,
        see (3.7)).  Higher a_j and the corrections to k, kappa are left to
        Newton.
        """
        x = np.zeros(self.N + 1)
        x[0] = (0.75 + self.kappa_star**2 / 12) * eps**2
        x[self.N - 1] = self.k_star
        x[self.N] = self.kappa_star
        return x


# ----------------------------------------------------------------------------
# 3. Continuation in eps
# ----------------------------------------------------------------------------

def continue_branch(sys_, eps_target, max_step=0.004, verbose=True):
    """Follow the branch from near the disk to eps_target.

    The interval (0, eps_target] is divided into equal steps of length at most
    max_step.  The first step starts from the asymptotic guess; later steps
    use a secant (linear extrapolation) predictor, then Newton as corrector.

    Returns an array with rows (eps, a_2, ..., a_N, k, kappa).
    """
    nsteps = max(1, int(np.ceil(eps_target / max_step - 1e-12)))
    eps_values = eps_target * np.arange(1, nsteps + 1) / nsteps
    eps_values[-1] = eps_target                      # avoid rounding in the last step

    rows, prev = [], None            # prev = (eps, x) of the previous step
    x = sys_.asymptotic_guess(eps_values[0])
    for eps in eps_values:
        x_new, dx, res = sys_.newton(eps, x)
        if not np.isfinite(dx) or dx > 1e-8:
            raise RuntimeError(f"Newton failed at eps = {eps:.5f} (last update {dx:.1e}); "
                               "the branch may have reached its end, or N, M are too small.")
        k, kappa = x_new[sys_.N - 1], x_new[sys_.N]
        if verbose:
            print(f"  eps = {eps:.5f}   k = {k:.8f}   kappa = {kappa:.8f}   "
                  f"sigma = {(kappa / k)**2:.8f}   max|F| = {res:.1e}")
        rows.append(np.concatenate([[eps], x_new]))

        # secant predictor for the next step
        if prev is None:
            x = x_new
        else:
            x = x_new + (x_new - prev[1])            # equal steps => slope * step
        prev = (eps, x_new)
    return np.array(rows)


# ----------------------------------------------------------------------------
# 4. Diagnostics of a computed domain
# ----------------------------------------------------------------------------

def boundary(eps, a, npts=2001):
    """Polar radius (*) and the curvature of the boundary on a fine grid.

    Returns (t, rho, curvature).  The boundary points are
    (x, y) = rho (cos t, sin t).
    """
    t = np.linspace(0, 2 * np.pi, npts)
    j = np.arange(2, len(a) + 2)
    c, s = np.cos(2 * np.outer(j, t)), np.sin(2 * np.outer(j, t))
    rho = 1 + eps * np.cos(2 * t) + a @ c
    r1 = -2 * eps * np.sin(2 * t) - (2 * j * a) @ s
    r2 = -4 * eps * np.cos(2 * t) - (4 * j**2 * a) @ c
    # curvature of a polar curve r = rho(t)
    curv = (rho**2 + 2 * r1**2 - rho * r2) / (rho**2 + r1**2)**1.5
    return t, rho, curv


def report(sys_, row):
    """Print a summary of the final domain."""
    N = sys_.N
    eps, a, k, kappa = row[0], row[1:N], row[N], row[N + 1]
    ks, kps = sys_.k_star, sys_.kappa_star
    sig, sig_s = (kappa / k)**2, (kps / ks)**2
    t, rho, curv = boundary(eps, a)

    print("\nResonant disk (seed), cf. (1.13):")
    print(f"  k_* = {ks:.12f}   kappa_* = j_(2,1) = {kps:.12f}   sigma_* = {sig_s:.12f}")
    print(f"\nDomain at eps = {eps}:")
    print(f"  k = {k:.10f}   kappa = {kappa:.10f}   sigma = {sig:.10f}")
    print(f"  rho(0) = {rho[0]:.6f} (long axis)   rho(pi/2) = {rho[len(rho) // 4]:.6f} (short axis)")
    print(f"  curvature: min = {curv.min():.4f}, max = {curv.max():.4f}"
          f"   ({'convex' if curv.min() > 0 else 'NOT convex'})")
    # positivity of the incident wave on the boundary is what the analyticity
    # argument (proof of Proposition 3.1) uses; it fails at the end of the branch
    print(f"  min of J_0(k rho) on the boundary = {jv(0, k * rho).min():.5f}")
    print("  leading coefficients a_2, a_3, a_4 = "
          + ", ".join(f"{v:.3e}" for v in a[:3]))
    print(f"  size of last retained coefficient |a_N| = {abs(a[-1]):.1e}"
          "   (should be negligible)")


def compare_with_expansions(sys_, row):
    """Compare a point of the branch at small eps with Proposition 1.4.

    This is a test of the code: the quotients below must tend to the constants
    in (1.15) and (1.16) as eps -> 0.  It is applied to the first continuation
    step, because the expansions are only accurate for eps well below eps_0
    (the coefficients grow like (7 eps)^j).
    """
    N = sys_.N
    eps, a2, k, kappa = row[0], row[1], row[N], row[N + 1]
    ks, kps = sys_.k_star, sys_.kappa_star
    sig, sig_s = (kappa / k)**2, (kps / ks)**2
    c_star = 2 * sig_s * (2 * kps**2 + 3) * (kps**2 - ks**2) / (3 * (kps**4 + 4 * ks**2))
    print(f"\nCheck against Proposition 1.4, at eps = {eps:.4g}:")
    print(f"  a_2 / eps^2               = {a2 / eps**2:.5f}"
          f"   predicted by (1.15): 3/4 + kappa_*^2/12 = {0.75 + kps**2 / 12:.5f}")
    print(f"  (sigma - sigma_*) / eps^2 = {(sig - sig_s) / eps**2:.5f}"
          f"   predicted by (1.16): c_*                = {c_star:.5f}")


# ----------------------------------------------------------------------------
# 5. Output
# ----------------------------------------------------------------------------

def save_data(sys_, branch, outdir):
    """Write the branch as a CSV file: eps, k, kappa, sigma, a_2, ..., a_N."""
    N = sys_.N
    eps, a, k, kappa = branch[:, 0], branch[:, 1:N], branch[:, N], branch[:, N + 1]
    table = np.column_stack([eps, k, kappa, (kappa / k)**2, a])
    header = "eps,k,kappa,sigma," + ",".join(f"a_{j}" for j in range(2, N + 1))
    fname = os.path.join(outdir, "nonscattering_branch.csv")
    np.savetxt(fname, table, delimiter=",", header=header, comments="", fmt="%.16e")
    return fname


# ----------------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Non-scattering branch near the resonant disk.")
    p.add_argument("--eps", type=float, default=0.04, help="target eps (default 0.04)")
    p.add_argument("--N", type=int, default=24, help="highest mode cos(2Nt) (default 24)")
    p.add_argument("--M", type=int, default=256, help="quadrature nodes (default 256)")
    p.add_argument("--outdir", default=".", help="output directory")
    p.add_argument("--check", action="store_true",
                   help="repeat with N+8 modes and 2M nodes and compare")
    args = p.parse_args()
    if args.eps <= 0:
        p.error("--eps must be positive (eps -> -eps is a rotation by pi/2, see (3.2))")
    if args.eps > 0.05:
        print("WARNING: eps is at or beyond the end of the smooth branch (~0.0505); "
              "results are unreliable.")
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Continuation to eps = {args.eps} with N = {args.N}, M = {args.M}:")
    sys_ = MomentSystem(args.N, args.M)
    branch = continue_branch(sys_, args.eps)
    report(sys_, branch[-1])
    compare_with_expansions(sys_, branch[0])

    f_branch = save_data(sys_, branch, args.outdir)
    print("\nWritten:")
    print("  " + f_branch)

    if args.check:
        # Truncation check: the same computation with more modes and more
        # nodes should reproduce k, kappa and the low coefficients.
        N2, M2 = args.N + 8, 2 * args.M
        print(f"\nTruncation check with N = {N2}, M = {M2}:")
        sys2 = MomentSystem(N2, M2)
        b2 = continue_branch(sys2, args.eps, verbose=False)[-1]
        b1 = branch[-1]
        print(f"  |k difference|       = {abs(b1[args.N] - b2[N2]):.1e}")
        print(f"  |kappa difference|   = {abs(b1[args.N + 1] - b2[N2 + 1]):.1e}")
        n = min(6, args.N - 1)
        print(f"  max |a_j difference|, j = 2..{n + 1} = "
              f"{np.abs(b1[1:1 + n] - b2[1:1 + n]).max():.1e}")
        t, r1, _ = boundary(b1[0], b1[1:args.N])
        _, r2, _ = boundary(b2[0], b2[1:N2])
        print(f"  max |rho difference| on the boundary = {np.abs(r1 - r2).max():.1e}")


if __name__ == "__main__":
    main()
