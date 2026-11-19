(* ::Package:: *)

(* ==========================================================================
   nonscattering_branch.wl
   ==========================================================================

   Numerical computation of the branch of non-circular non-scattering domains
   bifurcating from the resonant disk, as constructed in

       F. Cakoni and M. Levitin, "A family of homogeneous convex non-scattering
       domains with analytic non-circular boundaries".

   This is a Wolfram Language version of nonscattering_branch.py; it uses the
   same equations, discretisation and continuation, and writes the same file.

   The branch is that of Theorem 1.2; the script also checks the expansions
   (1.15), (1.16) of Proposition 1.4.  Theorem and equation numbers refer to
   the version of the paper dated 19 November 2026.

   WHAT IS COMPUTED
   ----------------
   For each eps we look for a domain

       Omega_eps = { r (cos t, sin t) : 0 <= r < rho(t) },
       rho(t)    = 1 + eps cos 2t + sum_{j=2}^{N} a_j cos 2jt ,          (#)

   and wave numbers k (exterior) and kappa (interior), coefficient
   sigma = kappa^2 / k^2, such that Omega_eps with constant coefficient sigma
   does not scatter the radial Herglotz wave (1.10), v(x) = J_0(k |x|).
   The normalisation in (#) is (1.14).

   THE EQUATIONS
   -------------
   Omega is non-scattering for v iff there is w satisfying (1.11).  By
   Lemma 3.2 this is equivalent to the integral identities (3.3), which for a
   domain symmetric in both axes take the form (3.5),

       M_m = < cos(mt) F_m(rho(t); k, kappa) > = 0,     m = 0, 2, 4, ...

   We use them in an equivalent boundary form: with
   phi_m(r, t) = J_m(kappa r) cos(m t), Green's formula gives

       Q_m := int_{dOmega} ( phi_m d_nu v - v d_nu phi_m ) ds
            = 2 pi (kappa^2 - k^2) M_m = 0 ,

   and on the polar curve r = rho(t)

       Q_m = int_0^{2 pi} {  cos(mt) rho [ v_r J_m(kappa rho) - v kappa J_m'(kappa rho) ]
                           - m sin(mt) (rho'/rho) v J_m(kappa rho) } dt ,

   with v = J_0(k rho), v_r = -k J_1(k rho).

   DISCRETISATION
   --------------
   Unknowns  : a_2, ..., a_N, k, kappa        (N + 1 numbers)
   Equations : Q_m = 0 for m = 0, 2, ..., 2N  (N + 1 equations)

   Trapezoidal rule with M equispaced nodes in t; Newton's method with the
   exact Jacobian of the discretised equations; continuation from the disk in
   eps with a secant predictor.

   THE SEED (eps = 0)
   ------------------
   The parameters (1.12), well defined by Lemma 2.1, with values (1.13):

       kappa_star = j_{2,1},
       k_star     in (0, j_{0,1})  with  d_0(k_star) = d_0(kappa_star),
       d_0(t)  = t J_0'(t) / J_0(t).

   CAVEATS
   -------
   * Floating-point continuation, NOT a certified computation.
   * The smooth convex branch ends near eps_0 ~ 0.0505 (cf. Remark 1.3), where
     the tip of the domain reaches the first nodal circle of J_0(k|x|).  For
     eps close to eps_0 increase N and M and distrust the output.

   USAGE
   -----
   Set the parameters in the block below, then either evaluate the file in a
   notebook ( Get["nonscattering_branch.wl"] ) or run

       wolframscript -file nonscattering_branch.wl

   OUTPUT (in outDir)
   ------------------
       nonscattering_branch.csv     one row per continuation step:
                                    eps, k, kappa, sigma, a_2, ..., a_N

   Each row determines the corresponding boundary through (#).
   ========================================================================== *)


(* --------------------------------------------------------------------------
   0. Parameters
   -------------------------------------------------------------------------- *)

epsTarget = 0.05;        (* target eps *)
nModes = 24;             (* N: highest mode cos(2 N t) in the polar radius *)
nNodes = 256;            (* M: number of trapezoidal nodes on [0, 2 pi) *)
outDir = Directory[];    (* output directory *)
doCheck = False;         (* True: repeat with N+8 modes, 2M nodes and compare *)


(* --------------------------------------------------------------------------
   1. The resonant disk from which the branch bifurcates
   -------------------------------------------------------------------------- *)

(* d_0(t) = t J_0'(t) / J_0(t) = -t J_1(t) / J_0(t), see (1.6) *)
d0[t_] := -t BesselJ[1, t]/BesselJ[0, t];

(* kappa_star = j_{2,1}; k_star is the unique root of d_0(k) = d_0(kappa_star) in
   (0, j_{0,1}), see (1.12) and Lemma 2.1.  The root is bracketed by 2 and
   2.39 and computed with 30 digits, then rounded to machine precision; we
   use d_0(kappa_star) = -kappa_star^2/2, see (2.2), with kappa_star taken to
   50 digits so that the equation is known to more than the working precision. *)
kappaStar = N[BesselJZero[2, 1]];
kStar = With[{kap = N[BesselJZero[2, 1], 50]},
   N[kk /. FindRoot[d0[kk] + kap^2/2, {kk, 2, 239/100},
       Method -> "Brent", WorkingPrecision -> 30]]];
sigmaStar = (kappaStar/kStar)^2;


(* --------------------------------------------------------------------------
   2. The discretised moment conditions Q_m = 0
   -------------------------------------------------------------------------- *)

(* Precomputed data for given N (nn) and M (mm).
   "scale": J_m(kappa rho) ~ (kappa/2)^m / m!, so the raw Q_m span many orders
   of magnitude; multiplying Q_m by m! / (kappa_star/2)^m makes all equations O(1)
   and does not change the solution set. *)
makeSystem[nn_Integer, mm_Integer] := Module[{t, j, orders},
   t = 2. Pi Range[0, mm - 1]/mm;          (* quadrature nodes *)
   j = Range[2, nn];                       (* indices of the unknown a_j *)
   orders = 2 Range[0, nn];                (* test orders m = 0, 2, ..., 2N *)
   <|"N" -> nn, "M" -> mm, "t" -> t, "j" -> j, "m" -> orders,
     "cos2" -> Cos[2 t], "sin2" -> Sin[2 t],
     "cos2j" -> Cos[2. Outer[Times, j, t]],        (* (N-1) x M *)
     "sin2j" -> Sin[2. Outer[Times, j, t]],
     "cosm" -> Cos[1. Outer[Times, orders, t]],    (* (N+1) x M *)
     "sinm" -> Sin[1. Outer[Times, orders, t]],
     "scale" -> Exp[LogGamma[orders + 1.] - orders Log[kappaStar/2]]|>];

(* Scaled moments F = (Q_0, Q_2, ..., Q_{2N}) at x = (a_2, ..., a_N, k, kappa),
   and the exact Jacobian dF/dx, both by the same trapezoidal rule.

   For each test order mu, g is the integrand of Q_mu as a function of rho,
   rho' (at the nodes) and of k, kappa.  The Jacobian is obtained by
   differentiating g under the integral sign; second derivatives of Bessel
   functions are eliminated with Bessel's equation,
       J_m''(z) = -J_m'(z)/z - (1 - m^2/z^2) J_m(z).
   Returns {F, J}. *)
residualAndJacobian[sys_, eps_, x_] := Module[
   {nn, mm, a, k, kap, rho, drho, bJ0, bJ1, v, vr, vrr, z, q, w, vk, vrk, rows},
   nn = sys["N"]; mm = sys["M"];
   a = x[[1 ;; nn - 1]]; k = x[[nn]]; kap = x[[nn + 1]];

   (* rho(t) and rho'(t) at the nodes *)
   rho = 1 + eps sys["cos2"] + a . sys["cos2j"];
   drho = -2 eps sys["sin2"] - (2 sys["j"] a) . sys["sin2j"];

   (* incident wave on the boundary: v, dv/dr, d^2v/dr^2, and k-derivatives *)
   bJ0 = BesselJ[0, k rho]; bJ1 = BesselJ[1, k rho];
   v = bJ0;
   vr = -k bJ1;
   vrr = -k^2 (bJ0 - bJ1/(k rho));
   vk = rho vr/k;                  (* dv/dk *)
   vrk = vr/k + rho vrr/k;         (* d(v_r)/dk *)

   z = kap rho;
   q = drho/rho;
   w = 2. Pi/mm;                   (* trapezoidal weight *)

   rows = Table[
     Module[{mu, c, s, bJm, dJm, ddJm, g, gRho, gDrho, gK, gKap},
      mu = sys["m"][[i]]; c = sys["cosm"][[i]]; s = sys["sinm"][[i]];
      bJm = BesselJ[mu, z];
      dJm = (BesselJ[mu - 1, z] - BesselJ[mu + 1, z])/2;
      ddJm = -dJm/z - (1 - mu^2/z^2) bJm;

      (* the integrand and its partial derivatives *)
      g = c rho (vr bJm - v kap dJm) - mu s q v bJm;
      gRho = c (vr bJm - v kap dJm) + c rho (vrr bJm - v kap^2 ddJm)
         - mu s q (-v bJm/rho + vr bJm + v kap dJm);
      gDrho = -mu s v bJm/rho;
      gK = c rho (vrk bJm - vk kap dJm) - mu s q vk bJm;
      gKap = c rho (vr rho dJm - v (dJm + z ddJm)) - mu s q v rho dJm;

      (* d rho / d a_j = cos(2jt),   d rho' / d a_j = -2j sin(2jt) *)
      sys["scale"][[i]] w Join[
        {Total[g]},
        sys["cos2j"] . gRho - (2 sys["j"] sys["sin2j"]) . gDrho,
        {Total[gK], Total[gKap]}]
      ],
     {i, nn + 1}];
   {rows[[All, 1]], rows[[All, 2 ;;]]}];

(* Newton's method from the initial guess x0.
   Returns {x, size of last Newton update, max |F|}. *)
newton[sys_, eps_, x0_, tol_: 1.*^-13, maxit_: 30] := Module[
   {x = x0, dx = Infinity, f, jac, step, it = 0},
   While[it < maxit && dx >= tol,
    {f, jac} = residualAndJacobian[sys, eps, x];
    step = LinearSolve[jac, -f];
    x = x + step;
    dx = Max[Abs[step]];
    it++];
   {x, dx, Max[Abs[First[residualAndJacobian[sys, eps, x]]]]}];

(* First initial guess, from the expansion (1.15) of Proposition 1.4:
   a_2 = (3/4 + kappa_star^2/12) eps^2, with k, kappa at their seed values. *)
asymptoticGuess[sys_, eps_] := Module[{nn = sys["N"], x},
   x = ConstantArray[0., nn + 1];
   x[[1]] = (0.75 + kappaStar^2/12) eps^2;
   x[[nn]] = kStar;
   x[[nn + 1]] = kappaStar;
   x];


(* --------------------------------------------------------------------------
   3. Continuation in eps
   -------------------------------------------------------------------------- *)

(* number formatting for the printed output *)
fix[x_, d_] := ToString[NumberForm[x, {16, d}]];
sci[x_] := If[x == 0, "0",
   With[{e = Floor[Log10[Abs[x]]]},
    ToString[NumberForm[x/10.^e, {3, 1}]] <> "e" <> ToString[e]]];

(* Follow the branch from near the disk to epsTo, in equal steps of length at
   most maxStep.  The first step starts from the asymptotic guess; later steps
   use a secant predictor, then Newton as corrector.
   Returns a list of rows {eps, a_2, ..., a_N, k, kappa}. *)
continueBranch[sys_, epsTo_, verbose_: True, maxStep_: 0.004] := Module[
   {nn = sys["N"], nsteps, epsValues, rows = {}, prev = None, x, xNew, dx, res},
   nsteps = Max[1, Ceiling[epsTo/maxStep - 1.*^-12]];
   epsValues = N[epsTo Range[nsteps]/nsteps];
   epsValues[[-1]] = epsTo;
   x = asymptoticGuess[sys, First[epsValues]];
   Do[
    {xNew, dx, res} = newton[sys, eps, x];
    If[! NumericQ[dx] || dx > 1.*^-8,
     Print["Newton failed at eps = ", eps,
      "; the branch may have reached its end, or N, M are too small."];
     Abort[]];
    If[verbose,
     Print["  eps = ", fix[eps, 5], "   k = ", fix[xNew[[nn]], 8],
      "   kappa = ", fix[xNew[[nn + 1]], 8],
      "   sigma = ", fix[(xNew[[nn + 1]]/xNew[[nn]])^2, 8],
      "   max|F| = ", sci[res]]];
    AppendTo[rows, Join[{eps}, xNew]];
    (* secant predictor for the next step (equal steps) *)
    x = If[prev === None, xNew, 2 xNew - prev];
    prev = xNew,
    {eps, epsValues}];
   rows];


(* --------------------------------------------------------------------------
   4. Diagnostics of a computed domain
   -------------------------------------------------------------------------- *)

(* Polar radius (#) and the curvature of the boundary on a fine grid.
   Returns {t, rho, curvature}; the boundary points are rho (cos t, sin t). *)
boundaryData[eps_, a_, npts_: 2001] := Module[{t, j, c, s, rho, r1, r2, curv},
   t = 2. Pi Range[0, npts - 1]/(npts - 1);
   j = Range[2, Length[a] + 1];
   c = Cos[2. Outer[Times, j, t]]; s = Sin[2. Outer[Times, j, t]];
   rho = 1 + eps Cos[2 t] + a . c;
   r1 = -2 eps Sin[2 t] - (2 j a) . s;
   r2 = -4 eps Cos[2 t] - (4 j^2 a) . c;
   curv = (rho^2 + 2 r1^2 - rho r2)/(rho^2 + r1^2)^1.5;
   {t, rho, curv}];

(* Print a summary of the domain in the given row. *)
report[sys_, row_] := Module[
   {nn = sys["N"], eps, a, k, kap, t, rho, curv},
   eps = row[[1]]; a = row[[2 ;; nn]]; k = row[[nn + 1]]; kap = row[[nn + 2]];
   {t, rho, curv} = boundaryData[eps, a];
   Print["\nResonant disk (seed), cf. (1.13):"];
   Print["  k_* = ", fix[kStar, 12], "   kappa_* = j_(2,1) = ",
    fix[kappaStar, 12], "   sigma_* = ", fix[sigmaStar, 12]];
   Print["\nDomain at eps = ", eps, ":"];
   Print["  k = ", fix[k, 10], "   kappa = ", fix[kap, 10],
    "   sigma = ", fix[(kap/k)^2, 10]];
   Print["  rho(0) = ", fix[rho[[1]], 6], " (long axis)   rho(pi/2) = ",
    fix[rho[[(Length[rho] - 1)/4 + 1]], 6], " (short axis)"];
   Print["  curvature: min = ", fix[Min[curv], 4], ", max = ", fix[Max[curv], 4],
    If[Min[curv] > 0, "   (convex)", "   (NOT convex)"]];
   (* positivity of the incident wave on the boundary is what the analyticity
      argument (proof of Proposition 3.1) uses; it fails at the end of the branch *)
   Print["  min of J_0(k rho) on the boundary = ", fix[Min[BesselJ[0, k rho]], 5]];
   Print["  leading coefficients a_2, a_3, a_4 = ",
    StringRiffle[sci /@ Take[a, UpTo[3]], ", "]];
   Print["  size of last retained coefficient |a_N| = ", sci[Abs[Last[a]]],
    "   (should be negligible)"]];

(* Compare a point of the branch at small eps with Proposition 1.4: the
   quotients must tend to the constants in (1.15), (1.16) as eps -> 0. *)
compareWithExpansions[sys_, row_] := Module[
   {nn = sys["N"], eps, a2, k, kap, sig, cStar},
   eps = row[[1]]; a2 = row[[2]]; k = row[[nn + 1]]; kap = row[[nn + 2]];
   sig = (kap/k)^2;
   cStar = 2 sigmaStar (2 kappaStar^2 + 3) (kappaStar^2 - kStar^2)/
     (3 (kappaStar^4 + 4 kStar^2));
   Print["\nCheck against Proposition 1.4, at eps = ", eps, ":"];
   Print["  a_2 / eps^2               = ", fix[a2/eps^2, 5],
    "   predicted by (1.15): 3/4 + kappa_*^2/12 = ", fix[0.75 + kappaStar^2/12, 5]];
   Print["  [sigma - sigma_*] / eps^2   = ", fix[(sig - sigmaStar)/eps^2, 5],
    "   predicted by (1.16): c_*                = ", fix[cStar, 5]]];


(* --------------------------------------------------------------------------
   5. Output
   -------------------------------------------------------------------------- *)

(* Write the branch as a CSV file: eps, k, kappa, sigma, a_2, ..., a_N,
   with all available (about 16) significant digits. *)
saveData[sys_, branch_, dir_] := Module[{nn = sys["N"], header, lines, file},
   header = "eps,k,kappa,sigma," <>
     StringRiffle[("a_" <> ToString[#]) & /@ Range[2, nn], ","];
   lines = Table[
     StringRiffle[
      ToString[CForm[#]] & /@
       Join[{row[[1]], row[[nn + 1]], row[[nn + 2]],
         (row[[nn + 2]]/row[[nn + 1]])^2}, row[[2 ;; nn]]], ","],
     {row, branch}];
   file = FileNameJoin[{dir, "nonscattering_branch.csv"}];
   Export[file, StringRiffle[Prepend[lines, header], "\n"] <> "\n", "Text"];
   "nonscattering_branch.csv"];


(* --------------------------------------------------------------------------
   6. Main
   -------------------------------------------------------------------------- *)

If[epsTarget <= 0,
  Print["epsTarget must be positive (eps -> -eps is a rotation by pi/2, see (3.2))"];
  Abort[]];
If[epsTarget > 0.05,
  Print["WARNING: eps is at or beyond the end of the smooth branch (~0.0505); ",
   "results are unreliable."]];
If[! DirectoryQ[outDir], CreateDirectory[outDir]];

Print["Continuation to eps = ", epsTarget, " with N = ", nModes, ", M = ", nNodes, ":"];
system = makeSystem[nModes, nNodes];
(* branch = continueBranch[system, epsTarget]; *)
branch = continueBranch[system, epsTarget, True, 0.005];
report[system, Last[branch]];
compareWithExpansions[system, First[branch]];

Print["\nWritten:"];
Print["  ", saveData[system, branch, outDir]];

If[doCheck,
  (* Truncation check: the same computation with more modes and more nodes
     should reproduce k, kappa and the low coefficients. *)
  Module[{n2 = nModes + 8, m2 = 2 nNodes, sys2, b1, b2, n, rho1, rho2},
   Print["\nTruncation check with N = ", n2, ", M = ", m2, ":"];
   sys2 = makeSystem[n2, m2];
   b2 = Last[continueBranch[sys2, epsTarget, False]];
   b1 = Last[branch];
   Print["  |k difference|       = ", sci[Abs[b1[[nModes + 1]] - b2[[n2 + 1]]]]];
   Print["  |kappa difference|   = ", sci[Abs[b1[[nModes + 2]] - b2[[n2 + 2]]]]];
   n = Min[6, nModes - 1];
   Print["  max |a_j difference|, j = 2..", n + 1, " = ",
    sci[Max[Abs[b1[[2 ;; n + 1]] - b2[[2 ;; n + 1]]]]]];
   rho1 = boundaryData[b1[[1]], b1[[2 ;; nModes]]][[2]];
   rho2 = boundaryData[b2[[1]], b2[[2 ;; n2]]][[2]];
   Print["  max |rho difference| on the boundary = ", sci[Max[Abs[rho1 - rho2]]]]]];
