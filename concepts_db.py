#!/usr/bin/env python3
"""
concepts_db
===========
Core concept of every chapter (Maths, Physics, Chemistry, Biology) for JEE / NEET style exams.
Each entry has: core idea, key formulas/facts, how to attack problems, and the common trap.

Use it three ways
-----------------
1. Command line:
       python concepts_db.py "area under curve"      # search
       python concepts_db.py --list                  # list all chapters
       python concepts_db.py --list --subject physics
2. From Python:
       from concepts_db import search, format_chapter
       for c in search("projectile motion"): print(format_chapter(c))
3. As a tool for the AI agent (exam_agent.py calls search() for you).

Formulas are in plain text so they print in any terminal: ^ means power, sqrt() square root.
Always confirm the syllabus of your own exam; chapters and weightage change over time.
"""

import argparse
import re
import sys

CHAPTERS = []


def C(subject, chapter, core, key, method, trap, tags=""):
    CHAPTERS.append({"subject": subject, "chapter": chapter, "core": core,
                     "key": key, "method": method, "trap": trap, "tags": tags})


# =============================== MATHEMATICS ===============================
C("maths", "Sets, Relations and Functions",
  "A function gives exactly one output for each input. One-one, onto, even and odd decide counting and symmetry.",
  "Functions from m elements to n elements: n^m. One-one functions: nPm. Even: f(-x)=f(x). Odd: f(-x)=-f(x). "
  "Composition (f o g)(x)=f(g(x)). Domain: denominator!=0, sqrt argument>=0, log argument>0.",
  "Find the domain first. Test symmetry with f(-x). For inverse, swap x and y and solve.",
  "Composition order is g first. Forgetting domain restrictions.",
  "function domain range one-one onto inverse composition even odd relation set")
C("maths", "Quadratic Equations",
  "Roots are linked to coefficients, and the discriminant tells the nature of the roots without solving.",
  "x=(-b+-sqrt(D))/(2a), D=b^2-4ac. Sum of roots=-b/a, product=c/a. D>0 two real, D=0 equal, D<0 complex conjugates. "
  "ax^2+bx+c>0 for all x if a>0 and D<0.",
  "Use symmetric functions of the roots (a^2+b^2=(a+b)^2-2ab) instead of solving. For roots in an interval check D, "
  "sign of f at the ends, and the vertex -b/(2a).",
  "Rejecting or keeping roots outside the allowed domain.",
  "quadratic roots discriminant sum product equation polynomial inequality")
C("maths", "Complex Numbers",
  "Numbers x+iy are points/vectors in a plane. Polar form turns powers and roots into angle arithmetic.",
  "|z|^2=z*conj(z). z=r(cos t+i sin t)=r e^(it). De Moivre: (cos t+i sin t)^n=cos nt+i sin nt. "
  "nth roots of unity sum to 0. 1+w+w^2=0, w^3=1. |z-z0|=r is a circle. |z1+z2|<=|z1|+|z2|.",
  "Convert to polar for powers/roots. Turn locus conditions into geometry (circle, perpendicular bisector).",
  "Argument in the wrong quadrant. (sqrt2)^8 is 16, not 2^8.",
  "complex number modulus argument polar de moivre roots of unity locus conjugate iota")
C("maths", "Sequences and Series",
  "Patterns with a constant difference (AP) or constant ratio (GP). AM>=GM>=HM gives quick maxima and minima.",
  "AP: a_n=a+(n-1)d, S_n=n/2[2a+(n-1)d]. GP: a_n=a r^(n-1), S_n=a(r^n-1)/(r-1), S_inf=a/(1-r) if |r|<1. "
  "Sum n=n(n+1)/2, sum n^2=n(n+1)(2n+1)/6, sum n^3=[n(n+1)/2]^2. AM>=GM, equality when terms equal.",
  "Telescoping: write the general term as a difference. For AGP multiply by r and subtract. Use AM-GM for extrema.",
  "Wrong term index (n vs n-1). Using S_inf when |r|>=1.",
  "AP GP HP arithmetic geometric progression series sum telescoping AM GM")
C("maths", "Permutations and Combinations",
  "Counting: permutations when order matters, combinations when it does not.",
  "nPr=n!/(n-r)!, nCr=n!/(r!(n-r)!). Circular arrangement of n: (n-1)!. With repeats: n!/(p!q!). "
  "Identical objects into distinct boxes: (n+r-1)C(r-1). Derangements: n!(1-1/1!+1/2!-...).",
  "Decide whether order matters, split into cases, or count the complement. Treat 'always together' items as one block.",
  "Overcounting when objects are identical.",
  "permutation combination counting arrangement selection factorial circular derangement")
C("maths", "Binomial Theorem",
  "Expansion of (a+b)^n with coefficients nCr. The general term finds any specific term.",
  "T(r+1)=nCr a^(n-r) b^r. Sum of coefficients: put x=1. Alternating: x=-1. nC0+...+nCn=2^n. "
  "Middle term: one if n even, two if n odd.",
  "Write the general term, set the power of x as required (zero for independent term), solve for r.",
  "Term number is r+1, not r.",
  "binomial theorem expansion coefficient general term middle term independent term")
C("maths", "Probability",
  "A measure of chance. Conditional probability updates the chance when new information arrives.",
  "P(A or B)=P(A)+P(B)-P(A and B). P(A|B)=P(A and B)/P(B). Independent: P(A and B)=P(A)P(B). "
  "Bayes: P(Ei|A)=P(Ei)P(A|Ei)/sum P(Ej)P(A|Ej). Binomial: nCr p^r q^(n-r), mean np, variance npq.",
  "Define the sample space. 'At least one' = 1 - P(none). Use tree diagrams for stages and Bayes for reversed conditions.",
  "Mixing up independent with mutually exclusive.",
  "probability bayes conditional independent binomial distribution dice coin cards random variable")
C("maths", "Matrices and Determinants",
  "Matrices store linear relationships. The determinant tells whether a unique solution or inverse exists.",
  "A^-1=adj(A)/|A|. |AB|=|A||B|. |kA|=k^n|A|. |adj A|=|A|^(n-1). (AB)^T=B^T A^T, (AB)^-1=B^-1 A^-1. "
  "Cramer's rule x_i=|A_i|/|A|.",
  "Row/column operations to simplify determinants. For AX=B: |A|!=0 gives a unique solution; if |A|=0 check consistency.",
  "Order reverses in (AB)^T and (AB)^-1.",
  "matrix determinant inverse adjoint system linear equations cramer rank symmetric skew")
C("maths", "Trigonometry",
  "Relations between angles and sides. Identities reduce expressions; general solutions list all roots.",
  "sin^2+cos^2=1. sin(A+-B)=sinAcosB+-cosAsinB. cos(A+-B)=cosAcosB-+sinAsinB. sin2A=2sinAcosA. "
  "cos2A=2cos^2A-1. sinx=sin a => x=n pi+(-1)^n a; cosx=cos a => x=2n pi+-a; tanx=tan a => x=n pi+a. "
  "Range of a sin x+b cos x: [-sqrt(a^2+b^2), sqrt(a^2+b^2)]. Sine rule a/sinA=2R. Cosine rule a^2=b^2+c^2-2bc cosA.",
  "Reduce to one function, solve, then check solutions in the given interval. Use auxiliary angle for a sin x+b cos x.",
  "Squaring can add extra roots; always verify. Inverse trig principal ranges.",
  "trigonometry identities equation inverse trig heights distances triangle sine rule cosine rule")
C("maths", "Straight Lines",
  "Lines in a plane are described by slope and intercept; distances and angles follow from simple formulas.",
  "y-y1=m(x-x1). Distance from (x1,y1) to ax+by+c=0: |ax1+by1+c|/sqrt(a^2+b^2). "
  "tan(theta)=|(m1-m2)/(1+m1 m2)|. Perpendicular: m1 m2=-1. Family of lines through an intersection: L1+kL2=0.",
  "Pick the form that fits the given data. Use the family L1+kL2=0 for lines through an intersection point.",
  "Vertical lines have no slope.",
  "straight line slope intercept distance angle locus pair of lines")
C("maths", "Circles",
  "A circle is all points at a fixed distance from the centre; tangents and chords reduce to distance conditions.",
  "(x-h)^2+(y-k)^2=r^2. General: x^2+y^2+2gx+2fy+c=0, centre (-g,-f), radius sqrt(g^2+f^2-c). "
  "y=mx+c touches x^2+y^2=r^2 if c^2=r^2(1+m^2). Tangent length=sqrt(S1). Radical axis: S1-S2=0. "
  "Circles touch externally if d=r1+r2, internally if d=|r1-r2|.",
  "Compare distance from centre to line with radius. Use radical axis for common chords.",
  "Radius formula needs g^2+f^2-c>0.",
  "circle tangent chord radical axis centre radius")
C("maths", "Conic Sections (Parabola, Ellipse, Hyperbola)",
  "Curves defined by distance to a focus and directrix (eccentricity e). Tangent conditions are standard.",
  "Parabola y^2=4ax: focus (a,0), directrix x=-a, latus rectum 4a, tangent y=mx+a/m, point (at^2,2at). "
  "Ellipse x^2/a^2+y^2/b^2=1: e^2=1-b^2/a^2, foci (+-ae,0), focal distances add to 2a, tangent c^2=a^2m^2+b^2. "
  "Hyperbola x^2/a^2-y^2/b^2=1: e^2=1+b^2/a^2, asymptotes y=+-(b/a)x, focal distances differ by 2a, c^2=a^2m^2-b^2.",
  "Use parametric forms for tangents/normals. Use focal property definitions for distance problems.",
  "Which axis is major; a>b for the standard ellipse.",
  "parabola ellipse hyperbola conic focus directrix eccentricity latus rectum asymptote tangent normal")
C("maths", "Limits, Continuity and Differentiability",
  "A limit is the value a function approaches. Continuity means no jump; differentiability means a unique tangent.",
  "sinx/x->1, tanx/x->1, (e^x-1)/x->1, ln(1+x)/x->1, (1+x)^(1/x)->e, (a^x-1)/x->ln a. "
  "Continuous at a if LHL=RHL=f(a). Differentiable implies continuous, not conversely (|x| at 0).",
  "Substitute first. For 0/0: factorise, rationalise, standard limits, series expansion, or L'Hopital.",
  "Match the argument inside sin with the denominator before using sinx/x.",
  "limit continuity differentiability lhopital indeterminate form sandwich greatest integer modulus")
C("maths", "Differentiation and Applications of Derivatives",
  "The derivative is the rate of change; it gives tangents, monotonic behaviour and extreme values.",
  "Product, quotient, chain rules. d(a^x)=a^x ln a. d(sin^-1 x)=1/sqrt(1-x^2). d(tan^-1 x)=1/(1+x^2). "
  "Tangent slope f'(x1); normal slope -1/f'(x1). f'>0 increasing. At f'(c)=0: f''<0 max, f''>0 min. "
  "Rolle: f(a)=f(b) => f'(c)=0. LMVT: f'(c)=(f(b)-f(a))/(b-a).",
  "For extrema on a closed interval compare critical points and both end points. For word problems form one-variable function.",
  "Maximum can be at an end point, not at the critical point.",
  "derivative tangent normal maxima minima increasing decreasing rolle lmvt rate of change optimisation")
C("maths", "Integration (Indefinite and Definite)",
  "Integration reverses differentiation and measures accumulated quantity. Symmetry properties save work.",
  "x^n -> x^(n+1)/(n+1); 1/x -> ln|x|; 1/(x^2+a^2) -> (1/a)tan^-1(x/a); 1/sqrt(a^2-x^2) -> sin^-1(x/a). "
  "By parts: ILATE. e^x[f+f']dx=e^x f. King: int_a^b f(x)=int_a^b f(a+b-x). Even: int_-a^a f=2 int_0^a f. Odd: 0. "
  "Leibniz: d/dx int_u^v f(t)dt=f(v)v'-f(u)u'.",
  "Try substitution, then by parts, then partial fractions. For definite integrals test symmetry and the King property first.",
  "Change limits when you substitute in a definite integral.",
  "integration integral definite indefinite by parts substitution partial fractions king property leibniz")
C("maths", "Area Under Curves",
  "Area is an integral of (upper curve - lower curve). Split where the curves cross.",
  "Area between y=f and x-axis: int|f|dx. Between two curves: int|f-g|dx. For min{f,g}: find f=g, test which is lower on each part, "
  "split. Even functions: double the half-area. Straight-line parts are triangles/trapezoids.",
  "Sketch, find intersections, decide the lower curve piecewise, integrate each piece. Use geometry for lines.",
  "Reading the function wrongly (e.g. (x^2+5)/12 versus x^2+5/12) gives ugly intersections.",
  "area under curve between curves min max bounded region integral parabola line")
C("maths", "Differential Equations",
  "Equations relating a function and its derivatives. Identify the type, then use its standard method.",
  "Order=highest derivative, degree=its power. Separable: dy/g(y)=f(x)dx. Homogeneous: y=vx. "
  "Linear dy/dx+Py=Q: IF=e^(int P dx), y*IF=int Q*IF dx.",
  "Classify (separable, homogeneous, linear, exact), apply the method, use initial conditions for the constant.",
  "Degree is defined only after clearing fractional powers of derivatives.",
  "differential equation order degree separable homogeneous linear integrating factor initial condition")
C("maths", "Vectors and 3D Geometry",
  "Vectors encode magnitude and direction; dot and cross products give angles, areas and volumes.",
  "a.b=|a||b|cos t. |axb|=|a||b|sin t. [a b c]=volume of parallelepiped (0 if coplanar). Line: r=a+lambda b. "
  "Plane ax+by+cz+d=0 has normal (a,b,c). Point-plane distance |ax1+by1+cz1+d|/sqrt(a^2+b^2+c^2). "
  "Skew-line distance |(a2-a1).(b1xb2)|/|b1xb2|.",
  "Convert geometry to vectors, use dot product for angles/perpendicularity and cross product for normals/areas.",
  "Write the plane equation in the form ...+d=0 before using the distance formula.",
  "vector dot product cross product 3d line plane distance angle skew lines direction cosines triple product")
C("maths", "Statistics",
  "Mean, variance and standard deviation describe the centre and spread of data.",
  "Mean=sum x/n. Variance=sum x^2/n-mean^2. SD=sqrt(variance). Adding a constant leaves variance unchanged; "
  "multiplying by k multiplies variance by k^2.",
  "Use the shortcut formula for variance. For grouped data use class marks.",
  "Variance is not changed by shifting the data.",
  "statistics mean variance standard deviation median mode data dispersion")

# ================================ PHYSICS =================================
C("physics", "Units, Measurements and Errors",
  "Dimensions check whether a formula can be right. Errors tell how reliable a result is.",
  "Sum/difference: absolute errors add. Product/quotient: relative errors add. x^n: relative error multiplies by n. "
  "Dimensional analysis: both sides of an equation have the same dimensions.",
  "Use dimensional analysis to derive or check relations; carry the right number of significant figures.",
  "Dimensional analysis cannot find dimensionless constants.",
  "units dimensions error significant figures vernier screw gauge measurement")
C("physics", "Kinematics",
  "Description of motion without forces. Constant acceleration gives three equations; projectile motion splits into x and y.",
  "v=u+at, s=ut+(1/2)at^2, v^2=u^2+2as. Projectile: T=2u sin t/g, H=u^2 sin^2 t/(2g), R=u^2 sin2t/g. "
  "Relative velocity v_AB=v_A-v_B. Area under v-t = displacement.",
  "Choose axes and a sign convention, resolve vectors, treat each axis separately.",
  "Use sin^2 for height and sin2t for range; sign of g.",
  "kinematics projectile motion velocity acceleration relative velocity river boat graph free fall")
C("physics", "Laws of Motion",
  "Net force equals mass times acceleration. Draw a free body diagram for every body.",
  "F=ma. Friction: static f<=mu_s N, kinetic f=mu_k N. Circular motion a=v^2/r. Banking (no friction) tan t=v^2/(rg). "
  "Atwood: a=(m1-m2)g/(m1+m2), T=2m1m2g/(m1+m2).",
  "Free body diagram, choose axes, write F=ma along each axis, add constraint relations (string length).",
  "Friction direction opposes relative motion; pseudo force in accelerating frames.",
  "newton laws force friction pulley atwood inclined plane circular motion banking free body diagram")
C("physics", "Work, Energy and Power",
  "Work changes kinetic energy; conservative forces conserve mechanical energy.",
  "W=F.d, W_net=change in KE, KE=(1/2)mv^2, spring PE=(1/2)kx^2, PE=mgh, P=F.v. "
  "Collisions: momentum conserved; e=speed of separation/speed of approach.",
  "Use energy for speed/height problems and momentum for collisions. Choose a zero of potential energy.",
  "Kinetic energy is not conserved in inelastic collisions.",
  "work energy power collision momentum conservation spring potential kinetic")
C("physics", "Rotational Motion",
  "Rotation obeys rules parallel to linear motion with I, torque and angular momentum.",
  "tau=I alpha, L=I w, KE=(1/2)I w^2. I: ring MR^2, disc MR^2/2, solid sphere 2MR^2/5, rod about centre ML^2/12. "
  "Parallel axis I=Icm+Md^2. Rolling: v=wR, KE=(1/2)mv^2+(1/2)I w^2.",
  "Choose the axis, find I (parallel axis if needed), use torque or angular momentum conservation.",
  "Using I about the wrong axis.",
  "rotation torque moment of inertia angular momentum rolling centre of mass")
C("physics", "Gravitation",
  "Every mass attracts every other. Orbits balance gravity with the needed centripetal force.",
  "F=Gm1m2/r^2, g=GM/R^2. Escape velocity sqrt(2GM/R)=sqrt(2gR). Orbital speed sqrt(GM/r). Kepler: T^2 proportional to r^3. "
  "U=-GMm/r. Orbit total energy -GMm/(2r).",
  "Use energy conservation for escape and variable-height problems, centripetal balance for orbits.",
  "Gravitational potential energy is negative.",
  "gravitation gravity escape velocity orbit satellite kepler planet")
C("physics", "Properties of Matter (Solids and Fluids)",
  "Elasticity, pressure, flow and surface tension.",
  "Y=stress/strain. Pressure P=P0+rho g h. Continuity A1v1=A2v2. Bernoulli P+(1/2)rho v^2+rho g h=constant. "
  "Excess pressure: drop 2T/R, soap bubble 4T/R. Capillary rise h=2T cos t/(rho g r). Stokes terminal velocity 2r^2(rho-sigma)g/(9 eta).",
  "Pick the correct law: hydrostatics, continuity, Bernoulli or surface tension.",
  "Soap bubble has two surfaces (4T/R).",
  "elasticity young modulus fluid pressure bernoulli viscosity surface tension capillary buoyancy stokes")
C("physics", "Thermal Physics and Thermodynamics",
  "Heat and work change internal energy. Processes differ by what stays constant.",
  "PV=nRT. First law dU=Q-W. Isothermal PV=const; adiabatic PV^gamma=const (Q=0). Cp-Cv=R. U=(f/2)nRT. "
  "Carnot efficiency 1-T2/T1 (kelvin). Conduction Q/t=kA dT/L. Stefan P=sigma A T^4.",
  "Identify the process, apply the first law, find work from the area on a P-V diagram.",
  "Use kelvin, and watch sign convention for work.",
  "thermodynamics heat temperature first law adiabatic isothermal carnot engine calorimetry conduction radiation")
C("physics", "Kinetic Theory of Gases",
  "Gas pressure and temperature come from random molecular motion.",
  "v_rms=sqrt(3RT/M). Average KE per molecule (3/2)kT. Energy per degree of freedom (1/2)kT. gamma=1+2/f.",
  "Relate speeds and energies to T and M; use degrees of freedom for specific heats.",
  "v_rms, average and most probable speeds differ.",
  "kinetic theory gas rms speed degrees of freedom mean free path specific heat")
C("physics", "Oscillations (SHM)",
  "Restoring force proportional to displacement gives simple harmonic motion.",
  "a=-w^2 x, x=A sin(wt+phi), v=w sqrt(A^2-x^2), E=(1/2)kA^2. Spring T=2 pi sqrt(m/k). Pendulum T=2 pi sqrt(l/g). "
  "Springs: series 1/k=sum 1/ki; parallel k=sum ki.",
  "Find the restoring force for a small displacement, identify the effective k, then T=2 pi sqrt(m/k).",
  "Pendulum formula is valid only for small angles.",
  "oscillation shm spring pendulum time period amplitude energy damped resonance")
C("physics", "Waves and Sound",
  "Waves carry energy; superposition gives beats and standing waves.",
  "v=f lambda. String v=sqrt(T/mu). Open pipe f=n v/(2L); closed pipe f=(2n-1)v/(4L). Beats=|f1-f2|. "
  "Doppler f'=f (v+-vo)/(v-+vs).",
  "Write boundary conditions for standing waves; apply Doppler with correct signs.",
  "Closed pipe has only odd harmonics.",
  "wave sound string standing wave beats doppler resonance pipe harmonic")
C("physics", "Electrostatics",
  "Charges create fields and potentials; Gauss's law handles symmetric charge distributions.",
  "F=kq1q2/r^2 (k=9e9). E=kq/r^2, V=kq/r. Gauss: flux=q_enc/eps0. Dipole: torque p x E, U=-p.E. "
  "Capacitor C=eps0 A/d, energy (1/2)CV^2. Series 1/C=sum 1/Ci, parallel C=sum Ci.",
  "Use symmetry and Gauss's law for fields; superposition for potentials (scalar).",
  "Field is a vector, potential is a scalar.",
  "electrostatics electric field potential gauss law capacitor dipole coulomb charge dielectric")
C("physics", "Current Electricity",
  "Charge flow in circuits obeys Ohm's law and Kirchhoff's rules.",
  "V=IR, R=rho l/A, P=VI=I^2R. Series R=sum; parallel 1/R=sum 1/Ri. Cell V=E-Ir. Max power when R=r. "
  "Balanced Wheatstone: P/Q=R/S.",
  "Simplify with series/parallel, else apply Kirchhoff's junction and loop rules.",
  "Internal resistance drops terminal voltage.",
  "current electricity resistance ohm kirchhoff wheatstone potentiometer cell emf power circuit")
C("physics", "Magnetic Effects of Current and Magnetism",
  "Moving charges feel magnetic force and currents create fields.",
  "F=qv x B, F=I L x B. r=mv/(qB). Long wire B=mu0 I/(2 pi r). Loop centre mu0 I/(2R). Solenoid mu0 n I. "
  "Ampere: closed integral B.dl=mu0 I_enc.",
  "Use right-hand rules for direction; Biot-Savart for loops; Ampere's law for symmetric currents.",
  "Magnetic force does no work on a charge.",
  "magnetism magnetic field biot savart ampere solenoid cyclotron force on wire moving charge")
C("physics", "Electromagnetic Induction and AC",
  "Changing flux induces emf (Lenz: opposes the change). AC circuits have reactance and resonance.",
  "emf=-d(flux)/dt, motional emf=Blv, inductor energy (1/2)LI^2. XL=wL, XC=1/(wC), Z=sqrt(R^2+(XL-XC)^2). "
  "Resonance w=1/sqrt(LC). Vrms=V0/sqrt2. Average power Vrms Irms cos(phi). Transformer Vs/Vp=Ns/Np.",
  "Find the flux change, apply Faraday and Lenz for direction; use phasors for AC.",
  "RMS not peak values for power; at resonance Z=R.",
  "induction faraday lenz ac alternating current inductor lcr resonance transformer eddy")
C("physics", "Electromagnetic Waves",
  "Oscillating electric and magnetic fields travelling at the speed of light.",
  "c=1/sqrt(mu0 eps0)=3e8 m/s. E0/B0=c. Spectrum (long to short wavelength): radio, microwave, infrared, visible, UV, X-ray, gamma.",
  "Recall the order of the spectrum and relation between E, B and c.",
  "All EM waves travel at c in vacuum.",
  "electromagnetic waves spectrum microwave infrared x-ray displacement current")
C("physics", "Ray Optics",
  "Reflection and refraction at mirrors, lenses and prisms.",
  "Mirror 1/v+1/u=1/f. Lens 1/v-1/u=1/f. Lens maker 1/f=(n-1)(1/R1-1/R2). Snell n1 sin t1=n2 sin t2. "
  "Critical angle sin c=n2/n1. Thin prism deviation (n-1)A. Power P=1/f (metres).",
  "Fix a sign convention first, then use the formula; draw the ray diagram to check.",
  "Sign of u and f for convex versus concave.",
  "optics ray lens mirror refraction prism total internal reflection telescope microscope")
C("physics", "Wave Optics",
  "Interference, diffraction and polarisation show the wave nature of light.",
  "YDSE fringe width beta=lambda D/d. Bright: path diff n lambda; dark: (n+1/2) lambda. Single slit first minimum a sin t=lambda. "
  "Brewster tan t=n. Malus I=I0 cos^2 t.",
  "Find path difference, apply the maxima/minima conditions, compute fringe width.",
  "Diffraction pattern width differs from interference fringe width.",
  "wave optics interference young double slit diffraction polarisation brewster malus coherence")
C("physics", "Dual Nature, Atoms and Nuclei",
  "Light behaves as photons; matter has wavelength; atoms have discrete levels; nuclei decay.",
  "E=h nu=hc/lambda (hc=1240 eV nm). Photoelectric KEmax=h nu-phi. de Broglie lambda=h/p. Bohr E_n=-13.6 Z^2/n^2 eV. "
  "N=N0 e^(-lambda t), t_half=0.693/lambda. E=delta m c^2 (1 u=931.5 MeV).",
  "Use energy conservation for photon problems; count half-lives for decay.",
  "Stopping potential relates to KEmax, not to intensity.",
  "photoelectric work function threshold frequency stopping potential de broglie bohr atom hydrogen spectrum nucleus radioactivity binding energy fission fusion half life")
C("physics", "Semiconductors and Logic Gates",
  "p-n junctions conduct in one direction; logic gates implement Boolean logic.",
  "Forward bias conducts, reverse blocks. Rectifier. NAND and NOR are universal gates. Zener works in reverse breakdown.",
  "Identify bias, then the conducting path; build truth tables for gates.",
  "Forward bias lowers the barrier.",
  "semiconductor diode transistor logic gate rectifier zener p-n junction")

# =============================== CHEMISTRY ================================
C("chemistry", "Some Basic Concepts (Mole Concept)",
  "The mole links mass, particles and volume; stoichiometry follows balanced equations.",
  "n=mass/M=N/NA. NA=6.022e23. Molarity=mol/L, molality=mol/kg solvent. Gas at STP: 22.4 L/mol. "
  "Limiting reagent decides yield. Percentage yield=actual/theoretical x100.",
  "Convert to moles, use mole ratios from the balanced equation, find the limiting reagent.",
  "Molarity uses volume of solution; molality uses mass of solvent.",
  "mole concept stoichiometry molarity molality limiting reagent empirical formula equivalent")
C("chemistry", "Atomic Structure",
  "Electrons occupy quantised levels described by four quantum numbers.",
  "Bohr r=0.529 n^2/Z A, E=-13.6 Z^2/n^2 eV. Rydberg 1/lambda=R Z^2(1/n1^2-1/n2^2). Orbitals per shell n^2. "
  "Radial nodes n-l-1, angular nodes l. Aufbau, Pauli, Hund. Exceptions: Cr 3d5 4s1, Cu 3d10 4s1.",
  "Write configurations, count nodes and unpaired electrons, use Rydberg for spectral lines.",
  "Cr and Cu are exceptions; 4s is filled before 3d but removed first.",
  "atomic structure bohr quantum numbers orbital electronic configuration hydrogen spectrum de broglie")
C("chemistry", "Periodic Table and Periodicity",
  "Properties repeat with atomic number and follow trends.",
  "Across a period: radius decreases, IE and electronegativity increase. Down a group: radius increases, IE decreases. "
  "IE exceptions: Be>B, N>O. Electron gain enthalpy: Cl more negative than F.",
  "Rank by effective nuclear charge and size; check stable half/fully filled subshell exceptions.",
  "General trends have exceptions (N vs O).",
  "periodic table trends ionisation energy electronegativity atomic radius electron affinity")
C("chemistry", "Chemical Bonding",
  "Atoms bond to reach stability; shape follows electron pair repulsion.",
  "Hybridisation: sp linear, sp2 trigonal planar, sp3 tetrahedral, sp3d trigonal bipyramidal, sp3d2 octahedral. "
  "MOT bond order=(Nb-Na)/2; O2 is paramagnetic. Fajans: more polarisation, more covalent. Dipole moment is vector sum.",
  "Count bond pairs + lone pairs for shape (VSEPR); use MOT for bond order and magnetism.",
  "Lone pairs change bond angles (NH3, H2O).",
  "chemical bonding vsepr hybridisation molecular orbital bond order dipole moment hydrogen bond ionic covalent")
C("chemistry", "States of Matter and Solid State",
  "Gas laws link P, V, T; crystals are described by unit cells.",
  "PV=nRT. Graham: rate proportional to 1/sqrt(M). Dalton: P_total=sum partial pressures. "
  "Unit cell atoms: sc 1, bcc 2, fcc 4. Density=Z M/(a^3 NA). Packing efficiency: sc 52%, bcc 68%, fcc/hcp 74%.",
  "Identify the unit cell, count atoms (Z), then use density and radius relations.",
  "Convert units of a (pm or Angstrom to cm) before computing density.",
  "gas laws ideal gas real gas solid state unit cell crystal packing efficiency density lattice")
C("chemistry", "Chemical Thermodynamics",
  "Energy changes decide spontaneity.",
  "dU=q+w. dH=dU+dn_g RT. dG=dH-T dS; spontaneous if dG<0. dG0=-RT ln K. Hess's law. Bond enthalpy: reactants minus products.",
  "Use Hess's law for enthalpy; check sign of dG; keep units consistent (kJ vs J).",
  "Convert dS from J/K to kJ/K when combining with dH in kJ.",
  "thermodynamics enthalpy entropy gibbs energy spontaneity hess law heat capacity calorimetry")
C("chemistry", "Chemical and Ionic Equilibrium",
  "Reactions reach a dynamic balance described by K; acids and bases follow ionic equilibria.",
  "Kp=Kc(RT)^dn. Q<K goes forward. pH=-log[H+], pH+pOH=14. Buffer pH=pKa+log([salt]/[acid]). Ksp for AxBy=(xs)^x (ys)^y. "
  "Weak acid: [H+]=sqrt(Ka C).",
  "Write the ICE table, apply K; use approximations only when the dissociation is small.",
  "Solid and pure liquid concentrations are not in K.",
  "equilibrium kc kp le chatelier ph buffer solubility product ksp acid base hydrolysis common ion")
C("chemistry", "Redox and Electrochemistry",
  "Electron transfer produces electrical energy; Nernst links potential to concentration.",
  "E=E0-(0.059/n)log Q at 298 K. dG0=-nFE0, F=96500. Mass deposited=M I t/(nF). Anode oxidation, cathode reduction. "
  "Kohlrausch law for molar conductivity.",
  "Balance redox by half reactions; identify anode and cathode; use Nernst for non-standard conditions.",
  "E0 values are not multiplied by coefficients.",
  "electrochemistry redox cell emf nernst electrolysis faraday conductivity kohlrausch galvanic")
C("chemistry", "Chemical Kinetics",
  "Rate of reaction depends on concentration, temperature and catalyst.",
  "Rate=k[A]^m[B]^n. First order: k=(2.303/t)log(a/(a-x)), t_half=0.693/k. Zero order: t_half=a/(2k). "
  "Arrhenius k=A e^(-Ea/RT); ln(k2/k1)=(Ea/R)(1/T1-1/T2).",
  "Find the order from data, then use the integrated rate law for that order.",
  "Order is experimental, not from stoichiometric coefficients.",
  "kinetics rate law order half life arrhenius activation energy catalyst molecularity")
C("chemistry", "Solutions",
  "Colligative properties depend only on the number of solute particles.",
  "Raoult: P_A=x_A P_A0. dTb=i Kb m, dTf=i Kf m, osmotic pressure pi=i C R T. Relative lowering of vapour pressure=x_solute. "
  "Van't Hoff i>1 dissociation, i<1 association.",
  "Find moles of solute, molality or molarity, apply the colligative formula with i.",
  "Do not forget i for electrolytes.",
  "solutions colligative raoult boiling point elevation freezing point depression osmotic pressure van't hoff")
C("chemistry", "s-Block, p-Block and Hydrogen",
  "Group trends in reactivity, oxidation states and compounds.",
  "s-block: strong reducing metals; solubility of sulphates decreases down group 2. p-block: inert pair effect, oxoacids, "
  "interhalogens, noble gas compounds (XeF2 linear, XeF4 square planar, XeF6 distorted octahedral).",
  "Use periodic trends and known structures; learn exceptions and special compounds.",
  "Anomalous behaviour of the first element in each group.",
  "s block p block alkali alkaline earth halogen noble gas nitrogen oxygen sulphur boron carbon hydrogen")
C("chemistry", "d- and f-Block Elements",
  "Transition metals show variable oxidation states, colour and catalytic activity.",
  "Colour from d-d transitions; magnetic moment sqrt(n(n+2)) BM; lanthanoid contraction; KMnO4 and K2Cr2O7 oxidising reactions.",
  "Write the d-configuration, count unpaired electrons, predict colour and magnetism.",
  "Zn2+, Cd2+, Sc3+ are colourless (no unpaired d electrons).",
  "d block f block transition metal lanthanoid actinoid potassium permanganate dichromate magnetic moment")
C("chemistry", "Coordination Compounds",
  "Metal ions bind ligands to form complexes with shapes and isomers.",
  "Crystal field: octahedral t2g and eg. Strong-field ligands (CN-, CO) low spin. CFSE. Isomerism: structural (linkage, ionisation, "
  "coordination) and stereo (geometrical, optical). Werner theory. EAN.",
  "Find oxidation state, d-count, geometry and unpaired electrons; name using IUPAC rules.",
  "Ligand order in names is alphabetical, not by charge.",
  "coordination compounds ligand crystal field splitting isomerism werner nomenclature complex")
C("chemistry", "Metallurgy and Environmental Chemistry",
  "Extraction of metals and impact of pollutants.",
  "Concentration, roasting/calcination, reduction, refining. Ellingham diagram: lower dG line reduces the other oxide. "
  "Greenhouse gases CO2, CH4; ozone depletion by CFCs; acid rain pH<5.6.",
  "Match method to ore type (sulphide: roasting; carbonate: calcination).",
  "Reduction by carbon works only where the carbon line lies below the metal line.",
  "metallurgy ore extraction ellingham environmental pollution ozone greenhouse")
C("chemistry", "General Organic Chemistry",
  "Electronic effects decide stability and reactivity of organic species.",
  "Inductive, resonance, hyperconjugation. Carbocation stability 3>2>1>methyl; benzylic/allylic extra stable. "
  "Acidity: more stable conjugate base, stronger acid. R/S by CIP rules. Isomerism types.",
  "Draw resonance structures, compare stability of intermediates, then decide the mechanism.",
  "Resonance beats inductive effect when both act.",
  "general organic chemistry inductive resonance hyperconjugation carbocation stability isomerism stereochemistry nomenclature")
C("chemistry", "Hydrocarbons and Haloalkanes",
  "Addition, substitution and elimination in carbon compounds.",
  "Markovnikov addition; peroxide effect with HBr gives anti-Markovnikov. SN1 (carbocation, racemisation, 3 degree) vs SN2 "
  "(inversion, 1 degree). Zaitsev elimination. Wurtz, Friedel-Crafts. Benzene substituents: activating o/p, deactivating meta.",
  "Classify substrate (1,2,3 degree), nucleophile and solvent to pick SN1/SN2/E1/E2.",
  "Halogens deactivate the ring but are o/p directing.",
  "alkane alkene alkyne benzene haloalkane substitution elimination sn1 sn2 markovnikov electrophilic aromatic")
C("chemistry", "Alcohols, Phenols, Aldehydes, Ketones, Acids and Amines",
  "Functional group reactions and named reactions.",
  "Aldol (alpha hydrogen), Cannizzaro (no alpha hydrogen), Clemmensen and Wolff-Kishner reductions, Rosenmund, Reimer-Tiemann, Kolbe, "
  "Haloform, Hofmann bromamide, Gabriel, Sandmeyer, Gattermann, Hell-Volhard-Zelinsky. Basicity of amines depends on solvent and substituents.",
  "Identify the functional group and reagent, recall the named reaction and product.",
  "Aldol needs alpha hydrogen; Cannizzaro needs none.",
  "alcohol phenol ether aldehyde ketone carboxylic acid amine named reaction aldol cannizzaro diazonium")
C("chemistry", "Biomolecules and Polymers",
  "Structure and function of carbohydrates, proteins, nucleic acids and polymers.",
  "Glucose, fructose, sucrose (non-reducing), starch, cellulose. Peptide bond. DNA/RNA bases. "
  "Addition polymers: polyethylene, PVC, teflon. Condensation: nylon, polyester, bakelite. Vulcanisation of rubber.",
  "Recall monomers and linkage types; classify polymers.",
  "Sucrose is non-reducing; glucose and fructose are reducing.",
  "biomolecules carbohydrate protein amino acid vitamin nucleic acid polymer nylon rubber")

# ================================ BIOLOGY =================================
C("biology", "Cell: Structure and Cell Cycle",
  "The cell is the basic unit of life; the cell cycle ensures faithful division.",
  "Prokaryote vs eukaryote. Mitochondria: ATP; Golgi: packaging; ribosome: protein synthesis. Cell cycle G1, S, G2, M. "
  "Meiosis reduces chromosome number; crossing over in prophase I.",
  "Link each organelle to its function and each cell cycle phase to DNA content.",
  "Mitosis keeps the chromosome number; meiosis halves it.",
  "cell organelle mitosis meiosis cell cycle membrane nucleus prokaryote eukaryote")
C("biology", "Biomolecules and Enzymes",
  "Biomolecules build cells; enzymes speed reactions by lowering activation energy.",
  "Carbohydrates, lipids, proteins (primary to quaternary), nucleic acids. Enzyme: Km, Vmax; competitive inhibitor raises apparent Km.",
  "Relate structure to function; read enzyme graphs.",
  "Enzymes are not consumed in reactions.",
  "biomolecules enzyme protein lipid carbohydrate nucleic acid km vmax inhibitor")
C("biology", "Molecular Basis of Inheritance",
  "DNA stores information that flows DNA to RNA to protein.",
  "Replication is semi-conservative. Transcription then translation. Genetic code: triplet, degenerate, universal, non-overlapping. "
  "Lac operon regulation. Human Genome Project.",
  "Trace the flow of information; identify template and coding strands.",
  "mRNA is complementary to the template strand, same as the coding strand (with U).",
  "dna rna replication transcription translation genetic code operon gene expression")
C("biology", "Principles of Inheritance and Variation",
  "Traits are inherited by Mendelian rules plus exceptions.",
  "Monohybrid 3:1, dihybrid 9:3:3:1, test cross. Incomplete dominance 1:2:1. ABO blood groups (multiple alleles + codominance). "
  "Sex-linked traits. Down (21 trisomy), Turner (XO), Klinefelter (XXY). Hardy-Weinberg p^2+2pq+q^2=1.",
  "Draw Punnett squares, track linked genes, and use Hardy-Weinberg for populations.",
  "Carriers have frequency 2pq, not q^2.",
  "genetics mendel punnett dihybrid linkage sex linked pedigree blood group hardy weinberg chromosomal disorder")
C("biology", "Evolution and Biodiversity",
  "Populations change over time by natural selection and other forces.",
  "Homologous vs analogous organs. Natural selection, genetic drift, speciation. Five-kingdom classification. "
  "Conservation: in situ (national parks) and ex situ (zoos, seed banks). Biodiversity hotspots.",
  "Match evidence (fossils, homology) to the evolutionary idea.",
  "Analogous organs show convergent evolution, not common ancestry.",
  "evolution natural selection speciation darwin biodiversity classification kingdom conservation")
C("biology", "Plant Physiology",
  "Plants make food by photosynthesis, respire, transport water and respond to hormones.",
  "Light reactions in thylakoid (ATP+NADPH); Calvin cycle in stroma. C3 vs C4 plants. Glycolysis, Krebs cycle, ETC. "
  "Transpiration pull and cohesion-tension. Hormones: auxin, gibberellin, cytokinin, ethylene, ABA.",
  "Locate each process in its organelle and track inputs and outputs.",
  "Calvin cycle does not need light directly but needs ATP and NADPH.",
  "photosynthesis respiration transpiration plant hormone auxin mineral nutrition xylem phloem")
C("biology", "Plant Morphology and Reproduction",
  "Structure of flowering plants and their sexual reproduction.",
  "Pollination, double fertilisation (triple fusion gives endosperm), seed and fruit development. Monocot vs dicot features. "
  "Classification: algae, bryophytes, pteridophytes, gymnosperms, angiosperms.",
  "Draw and label flower parts; follow pollen tube to double fertilisation.",
  "Double fertilisation is unique to angiosperms.",
  "flower pollination fertilisation seed fruit monocot dicot morphology anatomy plant kingdom")
C("biology", "Human Physiology",
  "Organ systems coordinate to keep the body stable.",
  "Digestion and absorption; breathing and haemoglobin; double circulation and cardiac cycle; nephron (filtration, reabsorption, secretion); "
  "neuron and action potential; endocrine glands; muscle sliding filament theory.",
  "Trace the pathway of food, air, blood, urine or signal step by step.",
  "Most absorption occurs in the small intestine; most water reabsorption in the PCT.",
  "human physiology digestion respiration circulation excretion neural control hormone muscle nephron heart")
C("biology", "Human Reproduction and Health",
  "Gamete formation, fertilisation, development and common diseases.",
  "Spermatogenesis and oogenesis. Menstrual cycle phases. Fertilisation, implantation. Contraception methods. "
  "Immunity (innate, acquired), vaccines, AIDS, cancer, common diseases.",
  "Sequence the events (menstrual cycle hormones, fertilisation) and match each disease to its cause.",
  "Fertilisation is in the ampulla of the fallopian tube.",
  "reproduction gamete menstrual cycle fertilisation embryo contraception immunity disease aids cancer")
C("biology", "Biotechnology and Ecology",
  "Tools to manipulate DNA, and the study of organisms and environment.",
  "Restriction enzymes, vectors, PCR, gel electrophoresis, Bt crops. Ecosystem energy flow (10 percent law). Nutrient cycles. "
  "Ecological succession. Population interactions: mutualism, competition, predation. Pollution, greenhouse effect, ozone depletion.",
  "Describe the technique step by step; use trophic levels for energy questions.",
  "Only about 10 percent of energy passes to the next trophic level.",
  "biotechnology recombinant dna pcr ecology ecosystem food chain energy flow succession pollution population")

# ------------------------------------------------------------------------


def _words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


_STOP = {"the", "a", "an", "of", "and", "in", "for", "to", "is", "are", "what", "find", "how", "value",
         "theorem", "equation", "law", "effect", "function", "formula", "problem", "question", "numerical"}


def _doc_words(c):
    return set(_words(c["chapter"] + " " + c["tags"] + " " + c["core"] + " " + c["key"]
                      + " " + c["method"] + " " + c["trap"]))


def search(query, k=3, subject=None):
    """Return the k chapters that best match the query (list of dicts).

    Rare words count more than common ones (so 'projectile' beats 'motion').
    """
    import math
    q = set(_words(query)) - _STOP
    if not q:
        q = set(_words(query))
    n_docs = len(CHAPTERS)
    df = {w: sum(1 for c in CHAPTERS if w in _doc_words(c)) for w in q}
    scored = []
    for c in CHAPTERS:
        if subject and c["subject"] != subject:
            continue
        title = set(_words(c["chapter"]))
        tags = set(_words(c["tags"]))
        body = set(_words(c["core"] + " " + c["key"] + " " + c["method"] + " " + c["trap"]))
        score = 0.0
        for w in q:
            if df[w] == 0:
                continue
            idf = math.log(1 + n_docs / df[w])
            weight = (3 if w in title else 0) + (4 if w in tags else 0) + (1 if w in body else 0)
            score += idf * weight
        if score > 0:
            scored.append((score, c))
    scored.sort(key=lambda t: -t[0])
    return [c for _, c in scored[:k]]


def format_chapter(c):
    return (f"[{c['subject'].upper()}] {c['chapter']}\n"
            f"  Core idea : {c['core']}\n"
            f"  Key facts : {c['key']}\n"
            f"  Method    : {c['method']}\n"
            f"  Trap      : {c['trap']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Core concept of every chapter.")
    ap.add_argument("query", nargs="*", help="words to search for")
    ap.add_argument("--list", action="store_true", help="list all chapters")
    ap.add_argument("--subject", choices=["maths", "physics", "chemistry", "biology"])
    ap.add_argument("-k", type=int, default=3, help="number of results")
    a = ap.parse_args(argv)

    if a.list:
        for c in CHAPTERS:
            if not a.subject or c["subject"] == a.subject:
                print(f"[{c['subject']}] {c['chapter']}")
        print(f"\n{len([c for c in CHAPTERS if not a.subject or c['subject'] == a.subject])} chapters")
        return 0
    if not a.query:
        ap.print_help()
        return 1
    results = search(" ".join(a.query), a.k, a.subject)
    if not results:
        print("No matching chapter. Try other words, or use --list.")
        return 1
    print("\n\n".join(format_chapter(c) for c in results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
