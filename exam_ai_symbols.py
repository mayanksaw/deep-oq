#!/usr/bin/env python3
"""
exam_ai.py - Concept builder + question solver for competitive exams.

Covers: JEE, JEE Advanced, NEET, IPhO, NSEP, NSEA, IChO, NSEC, NSEB, IBO, IJSO/NSEJS,
IOQM/INMO/IMO, IOI, GATE, CAT, CLAT, NDA, Banking, NTSE, CUET, CBSE Boards, UPSC, SSC.
You can add any exam by editing the SYLLABUS dictionary below.

HOW IT WORKS
  1. build  -> For every exam/subject/topic, the LLM writes a structured concept note
               (definitions, formulas, theorems, traps, worked examples) saved to
               exam_ai_data/concepts.json. This is your AI's "knowledge base".
  2. ask    -> For a question: classify it, retrieve the best concept notes,
               solve step by step several times, majority-vote the answer,
               run a verifier pass, and save the solved question for future reuse.

SETUP (Gemini - default)
  pip install google-genai
  Get a free key at https://aistudio.google.com/apikey, then EITHER:
    - paste the key into a file named gemini_key.txt next to this script, OR
    - export GEMINI_API_KEY="your-key"      (Windows: set GEMINI_API_KEY=your-key)
  Optional: choose a model with EXAM_AI_GEMINI_MODEL (default gemini-2.5-flash)

SETUP (Claude - alternative)
  pip install anthropic
  export ANTHROPIC_API_KEY="your-key"
  (Used automatically if no Gemini key is found, or force with EXAM_AI_PROVIDER=claude)

USAGE
  python exam_ai.py build                      # build concepts for ALL exams (long, costs API credits)
  python exam_ai.py build --exam JEE NEET      # only some exams
  python exam_ai.py build --exam IPhO --limit 5
  python exam_ai.py ask "A ball is thrown at 30 m/s at 45 degrees. Find the range."
  python exam_ai.py ask "Q text" --exam JEE --samples 5
  python exam_ai.py chat                       # interactive mode
  python exam_ai.py stats                      # how much is built so far
  python exam_ai.py fix                        # convert already-built notes to symbols (no API calls)

SYMBOLS VERSION (exam_ai_symbols.py)
  Same as exam_ai.py, but every answer and note is written with maths symbols:
  x⁶ not "x to the power 6", sin²θ not "sin squared theta", 90° not "90 degrees",
  ± not "plus or minus", ∫₀¹ not "integral from 0 to 1 of", a/b not "a divided by b".
  Leftover LaTeX ($, \frac, \begin{aligned}) is also converted. Code blocks are never changed.

Honest note: no system can solve *every* question correctly. This pipeline
(retrieval + multiple attempts + voting + verification) raises accuracy, but
always double-check answers for important decisions.
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
DATA_DIR = Path("exam_ai_data")
CONCEPT_FILE = DATA_DIR / "concepts.json"
SOLVED_FILE = DATA_DIR / "solved.json"
KEY_FILE = Path("gemini_key.txt")  # optional: paste your Gemini API key in this file
GEMINI_MODEL = os.environ.get("EXAM_AI_GEMINI_MODEL", "gemini-2.5-flash")
CLAUDE_MODEL = os.environ.get("EXAM_AI_MODEL", "claude-sonnet-5-5")

# --------------------------------------------------------------------------
# SYLLABUS  (edit / extend freely: exam -> subject -> list of topics)
# --------------------------------------------------------------------------
PHYSICS = [
    "Units, dimensions and measurement errors", "Kinematics (1D, 2D, projectile, relative motion)",
    "Newton's laws, friction and constraint motion", "Work, energy and power",
    "Center of mass, collisions and momentum", "Rotational motion and rigid body dynamics",
    "Gravitation and orbital motion", "Elasticity, fluid mechanics and surface tension",
    "Simple harmonic motion and oscillations", "Waves, sound and Doppler effect",
    "Thermal properties, calorimetry and heat transfer", "Kinetic theory and thermodynamics",
    "Electrostatics and Gauss's law", "Capacitors and dielectrics",
    "Current electricity and circuits", "Magnetic effects of current and magnetism",
    "Electromagnetic induction and AC circuits", "Electromagnetic waves",
    "Ray optics and optical instruments", "Wave optics (interference, diffraction, polarization)",
    "Dual nature of matter and photoelectric effect", "Atoms, nuclei and radioactivity",
    "Semiconductors and digital electronics", "Special relativity and modern physics",
]
CHEMISTRY = [
    "Mole concept and stoichiometry", "Atomic structure and quantum numbers",
    "Periodic table and periodic trends", "Chemical bonding and molecular structure",
    "States of matter (gases, liquids, solids)", "Chemical thermodynamics and thermochemistry",
    "Chemical equilibrium", "Ionic equilibrium, pH and buffers", "Redox reactions and electrochemistry",
    "Chemical kinetics", "Solutions and colligative properties", "Surface chemistry",
    "s-block and p-block elements", "d- and f-block elements and coordination compounds",
    "Metallurgy and qualitative salt analysis", "General organic chemistry (inductive, resonance, hyperconjugation)",
    "Stereochemistry and isomerism", "Hydrocarbons", "Haloalkanes and haloarenes",
    "Alcohols, phenols and ethers", "Aldehydes, ketones and carboxylic acids",
    "Amines and nitrogen compounds", "Biomolecules, polymers and everyday chemistry",
    "Named reactions and reaction mechanisms",
]
MATHS = [
    "Sets, relations and functions", "Complex numbers", "Quadratic equations and inequalities",
    "Sequences and series", "Permutations and combinations", "Binomial theorem",
    "Matrices and determinants", "Probability and statistics", "Trigonometry and inverse trigonometric functions",
    "Straight lines and circles", "Conic sections (parabola, ellipse, hyperbola)",
    "Limits, continuity and differentiability", "Applications of derivatives",
    "Indefinite and definite integration", "Differential equations", "Area under curves",
    "Vectors and 3D geometry", "Mathematical reasoning",
]
BIOLOGY = [
    "Diversity of living organisms and classification", "Cell structure and cell division",
    "Biomolecules and enzymes", "Plant physiology (photosynthesis, respiration, transport)",
    "Human physiology (digestion, breathing, circulation, excretion)",
    "Neural control, endocrine system and locomotion", "Reproduction in plants and humans",
    "Genetics and heredity (Mendel, linkage, pedigrees)", "Molecular biology (DNA, RNA, protein synthesis)",
    "Evolution", "Biotechnology and its applications", "Ecology, biodiversity and environment",
    "Human health, disease and immunity", "Microbes in human welfare",
]
OLYMPIAD_MATH = [
    "Number theory (divisibility, modular arithmetic, Diophantine equations)",
    "Algebra (inequalities, polynomials, functional equations)",
    "Combinatorics (counting, pigeonhole, invariants, graph theory)",
    "Geometry (Euclidean, triangle centres, circles, transformations)",
    "Proof writing techniques (induction, contradiction, extremal principle)",
]
CS = [
    "Complexity analysis and recursion", "Sorting and searching", "Dynamic programming",
    "Graphs (BFS, DFS, shortest paths, MST)", "Trees and segment trees", "Greedy algorithms",
    "Number theory and combinatorics for coding", "String algorithms", "Data structures (heap, stack, queue, DSU)",
]
UPSC = {
    "History": ["Ancient India", "Medieval India", "Modern India and freedom struggle", "World history", "Art and culture"],
    "Geography": ["Physical geography", "Indian geography", "World geography", "Climatology and oceanography", "Resources and agriculture"],
    "Polity": ["Constitution and its features", "Fundamental Rights, DPSP and Duties", "Union and State executive/legislature",
               "Judiciary", "Federalism, local government and elections", "Constitutional and statutory bodies"],
    "Economy": ["Basic macroeconomics and national income", "Banking, RBI and monetary policy", "Fiscal policy, budget and taxation",
                "Poverty, inflation and unemployment", "External sector and trade", "Agriculture and industry"],
    "Environment": ["Ecology and ecosystems", "Biodiversity and conservation", "Climate change and international agreements"],
    "Science and Technology": ["Space, defence and nuclear technology", "Biotechnology and health", "IT, AI and emerging tech"],
    "Current Affairs and Governance": ["Government schemes", "International relations", "Social issues and governance"],
    "CSAT": ["Reading comprehension", "Logical reasoning and analytical ability", "Basic numeracy and data interpretation"],
    "Ethics": ["Ethics theories and attitude", "Case studies and integrity"],
}
SSC = {
    "Quantitative Aptitude": ["Number system and simplification", "Percentage, profit and loss, discount",
                              "Ratio, proportion and mixtures", "Time, work, speed and distance",
                              "Algebra, geometry and mensuration", "Trigonometry", "Data interpretation"],
    "Reasoning": ["Verbal reasoning (series, coding, analogies, syllogisms)", "Non-verbal reasoning",
                  "Puzzles, seating and blood relations"],
    "English": ["Grammar and error spotting", "Vocabulary, synonyms and idioms", "Reading comprehension and cloze test"],
    "General Awareness": ["Static GK", "Indian polity and history basics", "General science", "Current affairs"],
}

# ---------------------- MORE CONCEPTS: advanced / olympiad level ----------------------
def uniq(items):
    """Remove duplicates but keep order."""
    return list(dict.fromkeys(items))


PHYSICS_ADV = [
    "Vectors, calculus and mathematical methods for physics", "Dimensional analysis and order-of-magnitude estimation",
    "Non-inertial frames, pseudo forces, Coriolis force", "Lagrangian and Hamiltonian mechanics basics",
    "Central force motion, Kepler's laws and orbit energetics", "Moment of inertia, rolling motion and angular momentum conservation",
    "Variable mass systems and rocket motion", "Fluid dynamics: Bernoulli, viscosity, Stokes law, Reynolds number",
    "Surface tension, capillarity and drops/bubbles", "Damped, forced and coupled oscillations, normal modes",
    "Wave motion: standing waves, beats, dispersion, group velocity", "Entropy, Carnot cycle, heat engines and refrigerators",
    "Phase change, latent heat and van der Waals gas", "Blackbody radiation, Wien and Stefan-Boltzmann laws",
    "Heat conduction, convection and radiation problems", "Electrostatics: multipole expansion and method of images",
    "Dielectrics, polarization and energy in fields", "Magnetostatics: Biot-Savart, Ampere's law, magnetic materials",
    "Maxwell's equations, displacement current and Poynting vector", "LCR circuits, resonance, transformers and power factor",
    "Transient circuits (RC, RL, LC) and time constants", "Geometrical optics: lens systems, prisms, aberrations",
    "Physical optics: thin films, Fraunhofer diffraction, gratings, Brewster angle", "Photons, X-rays, Compton effect and de Broglie waves",
    "Bohr model, hydrogen spectrum and basic quantum mechanics", "Nuclear physics: binding energy, decay chains, fission and fusion",
    "Particle physics and conservation laws", "Superconductivity, plasma and solid state basics",
    "Astrophysics: magnitudes, HR diagram, Hubble law, cosmology", "Experimental physics: error propagation, graphs and curve fitting",
    "Thermal expansion, thermal stresses and calorimetry puzzles", "Statistical physics and Maxwell-Boltzmann distribution",
]
CHEM_ADV = [
    "Quantum chemistry: orbitals, LCAO and molecular orbital theory", "Spectroscopy: UV-Vis, IR, NMR and mass spectrometry",
    "Gibbs energy, spontaneity, Hess's law and Kirchhoff's law", "Phase equilibria, phase diagrams and Gibbs phase rule",
    "Nernst equation, electrochemical cells and Faraday's laws", "Kinetics: Arrhenius equation, mechanisms and steady-state approximation",
    "Catalysis and enzyme kinetics (Michaelis-Menten)", "Coordination chemistry: crystal field theory, isomerism, magnetism",
    "Organometallic and bioinorganic chemistry basics", "Substitution and elimination: SN1, SN2, E1, E2 and carbocation rearrangements",
    "Aromaticity, electrophilic and nucleophilic aromatic substitution", "Pericyclic reactions and rearrangements",
    "Retrosynthesis and multi-step organic synthesis", "Carbohydrates, amino acids, proteins and nucleic acids",
    "Polymers: types, mechanisms and properties", "Analytical chemistry: titrations, gravimetric analysis and errors",
    "Qualitative organic analysis and functional group tests", "Solid state: unit cells, packing efficiency and defects",
    "Nuclear chemistry and radioactive decay", "Green and environmental chemistry",
    "Acid-base theories (Lewis, HSAB) and non-aqueous solvents", "Statistical thermodynamics basics and partition functions",
    "Real gases, virial equation and critical constants", "Chemistry of industrial processes (Haber, Contact, Ostwald)",
]
MATHS_ADV = [
    "Roots of unity and advanced complex number geometry", "Polynomials: Vieta, remainder theorem, rational root, Newton sums",
    "Inequalities: AM-GM, Cauchy-Schwarz, Jensen, rearrangement", "Functional equations and recurrence relations",
    "Number theory: CRT, Fermat, Euler, Wilson, orders, quadratic residues", "Combinatorics: inclusion-exclusion, generating functions, Catalan numbers",
    "Graph theory, pigeonhole and extremal arguments", "Geometry: Ceva, Menelaus, inversion, power of a point, coordinate and complex methods",
    "Calculus: Taylor series, improper integrals, Leibniz rule, Gamma/Beta functions", "Linear algebra: vector spaces, eigenvalues, rank and systems of equations",
    "Probability: conditional probability, Bayes, distributions and expectation", "Statistics: mean, variance, regression and hypothesis basics",
    "Real analysis basics: sequences, series convergence, continuity", "Locus, transformations and advanced coordinate geometry",
    "Game theory and invariants in olympiad problems", "Floor/ceiling functions and Diophantine equations",
]
BIO_ADV = [
    "Metabolism: glycolysis, Krebs cycle, electron transport chain", "Cell signalling, membrane transport and cell cycle control",
    "Microbiology and virology", "Immunology: innate, adaptive, antibodies and vaccines",
    "Plant anatomy, morphology and tissue systems", "Plant hormones, tropisms and photoperiodism",
    "Advanced animal physiology: nephron, cardiac cycle, action potential", "Developmental biology and embryology",
    "Population genetics, Hardy-Weinberg and speciation", "Genetic engineering: PCR, restriction enzymes, cloning, CRISPR",
    "Bioinformatics and biostatistics basics", "Taxonomy of animal phyla and plant divisions",
    "Animal behaviour and ethology", "Advanced ecology: succession, energy flow, population dynamics",
    "Endocrinology and homeostasis", "Biotechnology in medicine, agriculture and industry",
]
CS_ADV = [
    "Computer organization and architecture", "Operating systems (scheduling, memory, deadlock)", "Databases (SQL, normalization, transactions)",
    "Computer networks", "Theory of computation (automata, grammars, decidability)", "Compiler design",
    "Discrete mathematics and logic", "Digital logic and number systems", "Probability and linear algebra for CS",
    "Machine learning and data science basics", "Advanced DP (bitmask, digit, tree DP)", "Computational geometry basics",
    "Flows and matching", "Advanced data structures (BIT, sparse table, treaps)",
]

# More depth for existing exams
UPSC["History"] += ["Indian art, architecture and literature", "Post-independence India", "Modern world events (revolutions, world wars, decolonisation)"]
UPSC["Geography"] += ["Geomorphology and plate tectonics", "Population, urbanisation and human geography", "Mapping and important places"]
UPSC["Polity"] += ["Parliamentary procedures and committees", "Amendments and landmark Supreme Court judgments", "Governance, RTI and e-governance"]
UPSC["Economy"] += ["Capital markets, insurance and financial inclusion", "Sustainable development and economic surveys", "Infrastructure, energy and transport"]
UPSC["Science and Technology"] += ["Everyday science: physics, chemistry and biology basics", "Nanotechnology, robotics and quantum tech"]
UPSC["Current Affairs and Governance"] += ["Important summits, reports and indices", "Internal security and disaster management"]
SSC["Quantitative Aptitude"] += ["Averages, ages and partnership", "Simple and compound interest", "Mensuration 2D and 3D", "Coordinate geometry and statistics"]
SSC["Reasoning"] += ["Direction sense, ranking and calendar/clock", "Venn diagrams, matrices and Data sufficiency"]
SSC["English"] += ["One word substitution, spelling and sentence improvement", "Para jumbles and active/passive voice"]
SSC["General Awareness"] += ["Indian geography and economy basics", "Awards, books, sports and important days"]

# New exams
CAT = {
    "Quantitative Aptitude": ["Arithmetic (percentages, ratios, TSD, work)", "Algebra (equations, inequalities, functions, logarithms)",
                              "Geometry and mensuration", "Number system (remainders, factors, base systems)",
                              "Permutation, combination and probability", "Progressions and modern maths"],
    "DILR": ["Tables, bar graphs and caselets", "Arrangements and seating puzzles", "Games, tournaments and networks",
             "Venn diagrams, sets and logic puzzles"],
    "VARC": ["Reading comprehension strategies", "Para jumbles and para summary", "Odd sentence out and sentence insertion", "Vocabulary in context"],
}
GATE = {
    "Engineering Mathematics": ["Linear algebra", "Calculus", "Differential equations", "Probability and statistics", "Complex variables and transforms", "Numerical methods"],
    "Computer Science": CS + CS_ADV,
    "General Aptitude": ["Verbal ability", "Numerical ability", "Spatial and logical reasoning"],
}
CLAT = {
    "Legal Reasoning": ["Legal principles and fact application", "Constitutional law basics", "Law of torts, contracts and criminal law basics"],
    "Logical Reasoning": ["Arguments and inference", "Analogies and syllogisms", "Critical reasoning"],
    "English": ["Reading comprehension", "Grammar and vocabulary"],
    "Current Affairs": ["National and international events", "Legal news and judgments"],
    "Quantitative Techniques": ["Data interpretation", "Basic maths"],
}
NDA = {
    "Mathematics": ["Algebra", "Trigonometry", "Matrices and determinants", "Calculus", "Analytical geometry", "Statistics and probability"],
    "General Ability": ["Physics basics", "Chemistry basics", "Biology basics", "History, geography and polity", "Current affairs", "English grammar and vocabulary"],
}
BANKING = {
    "Quantitative Aptitude": ["Number series and simplification", "Data interpretation", "Arithmetic word problems", "Quadratic equations"],
    "Reasoning": ["Puzzles and seating arrangement", "Syllogism and inequalities", "Coding-decoding and blood relations", "Input-output and data sufficiency"],
    "English": ["Reading comprehension", "Cloze test and error spotting", "Para jumbles and fillers"],
    "Banking and General Awareness": ["Banking terms, RBI and financial institutions", "Economy and current affairs", "Static GK"],
    "Computer Aptitude": ["Computer fundamentals and networking", "MS Office and internet basics"],
}
NTSE_JUNIOR = {
    "Mental Ability": ["Series, analogy, classification", "Coding-decoding and puzzles", "Figure-based reasoning"],
    "Science": ["Class 9-10 Physics", "Class 9-10 Chemistry", "Class 9-10 Biology"],
    "Mathematics": ["Class 9-10 Algebra and number system", "Class 9-10 Geometry and mensuration", "Class 9-10 Trigonometry and statistics"],
    "Social Science": ["History", "Geography", "Civics and economics"],
}
CUET = {
    "Domain Subjects": ["Physics", "Chemistry", "Mathematics", "Biology", "Economics", "Accountancy", "History", "Political Science", "Geography"],
    "General Test": ["General knowledge and current affairs", "Logical and analytical reasoning", "Quantitative reasoning"],
    "Language": ["English comprehension and grammar", "Hindi comprehension and grammar"],
}
CBSE_BOARDS = {
    "Class 10 Science": ["Chemical reactions", "Acids, bases and salts", "Metals and non-metals", "Carbon compounds", "Life processes", "Control and coordination",
                         "Heredity", "Light", "Electricity", "Magnetic effects of current", "Environment"],
    "Class 10 Maths": ["Real numbers", "Polynomials", "Linear equations", "Quadratics", "AP", "Triangles", "Coordinate geometry", "Trigonometry", "Circles",
                       "Mensuration", "Statistics and probability"],
    "Class 12 Physics": PHYSICS[12:23], "Class 12 Chemistry": CHEMISTRY[5:24], "Class 12 Maths": MATHS[:17],
}

SYLLABUS = {
    "JEE":   {"Physics": PHYSICS, "Chemistry": CHEMISTRY, "Mathematics": MATHS},
    "JEE Advanced": {"Physics": uniq(PHYSICS + PHYSICS_ADV), "Chemistry": uniq(CHEMISTRY + CHEM_ADV),
                     "Mathematics": uniq(MATHS + MATHS_ADV)},
    "NEET":  {"Physics": PHYSICS[:22], "Chemistry": CHEMISTRY, "Biology": uniq(BIOLOGY + BIO_ADV)},
    "IPhO":  {"Physics": uniq(PHYSICS + PHYSICS_ADV)},
    "NSEP":  {"Physics": uniq(PHYSICS + PHYSICS_ADV)},
    "NSEA":  {"Astronomy": ["Celestial sphere and coordinate systems", "Stellar properties, magnitudes and HR diagram",
                            "Solar system and orbital mechanics", "Telescopes and observational astronomy",
                            "Galaxies, cosmology and Hubble law", "Stellar evolution, black holes and neutron stars"]},
    "IChO":  {"Chemistry": uniq(CHEMISTRY + CHEM_ADV + ["Laboratory techniques and titrations"])},
    "NSEC":  {"Chemistry": uniq(CHEMISTRY + CHEM_ADV)},
    "NSEB":  {"Biology": uniq(BIOLOGY + BIO_ADV)},
    "IBO":   {"Biology": uniq(BIOLOGY + BIO_ADV)},
    "IJSO/NSEJS": {"Physics": PHYSICS[:14], "Chemistry": CHEMISTRY[:12], "Biology": BIOLOGY[:10], "Mathematics": MATHS[:10]},
    "IOQM/INMO/IMO": {"Mathematics": uniq(OLYMPIAD_MATH + MATHS + MATHS_ADV)},
    "IOI":   {"Computer Science": uniq(CS + CS_ADV[10:])},
    "GATE":  GATE,
    "CAT":   CAT,
    "CLAT":  CLAT,
    "NDA":   NDA,
    "Banking (IBPS/SBI)": BANKING,
    "NTSE":  NTSE_JUNIOR,
    "CUET":  CUET,
    "CBSE Boards": CBSE_BOARDS,
    "UPSC":  UPSC,
    "SSC":   SSC,
}

# --------------------------------------------------------------------------
# MATH SYMBOLS
# The AI must write maths with symbols, never with words such as "squared",
# "theta", "plus", "minus", "to the power", "degrees", "frac" or raw LaTeX.
#   1) MATH_RULES is added to every prompt so the model writes symbols itself.
#   2) symbolize() is a safety net: it converts any leftover LaTeX or math words
#      into symbols. Code blocks (``` ... ``` and `...`) are never touched.
# --------------------------------------------------------------------------
MATH_RULES = """
MATH WRITING RULES (strict, apply to every answer):
- Write all maths with symbols, never with words. Use ² ³ ⁴ ⁿ for powers (x⁶, sin²θ, 10⁻³),
  √ ∛ for roots, π θ α β γ δ λ μ σ φ ω Δ Ω for Greek letters, ° for degrees (90°),
  + − × ÷ = ≠ ≈ ≤ ≥ < > ± ∓ → ⇒ ⇔ ∞ ∑ ∏ ∫ ∂ ∇ ∈ ∪ ∩ ∠ ⊥ ∥ ∴ ∵ for operations and relations.
- Fractions as a/b or (a+b)/(c+d). Integrals as ∫₀¹ (x⁶ − 4x⁵ + 5x⁴) dx. Subscripts as x₁, aₙ, log₂.
- Trigonometry as sin²θ, cos 2A, tan⁻¹(x), sin 30° = 1/2.
- NEVER write the words: squared, cubed, power, degree(s), theta, alpha, beta, pi (for π), plus, minus,
  times, divided by, frac, integral, sqrt, infinity, "less than or equal to", "plus or minus".
- NEVER use LaTeX: no $ signs, no \\frac, \\sqrt, \\theta, \\begin{aligned}, \\text{...}, \\( \\).
  Plain Unicode text only, so it displays correctly everywhere.
"""

_SUP = {**{str(i): c for i, c in enumerate("⁰¹²³⁴⁵⁶⁷⁸⁹")},
        "+": "⁺", "-": "⁻", "−": "⁻", "=": "⁼", "(": "⁽", ")": "⁾", "n": "ⁿ", "i": "ⁱ", "x": "ˣ", "y": "ʸ"}
_SUB = {**{str(i): c for i, c in enumerate("₀₁₂₃₄₅₆₇₈₉")},
        "+": "₊", "-": "₋", "−": "₋", "=": "₌", "(": "₍", ")": "₎", "n": "ₙ", "x": "ₓ"}

_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ",
    "eta": "η", "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν",
    "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ", "varphi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}
_CMD = {
    "times": "×", "div": "÷", "cdot": "·", "pm": "±", "mp": "∓", "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥",
    "neq": "≠", "ne": "≠", "approx": "≈", "equiv": "≡", "sim": "∼", "cong": "≅", "infty": "∞", "to": "→",
    "rightarrow": "→", "longrightarrow": "→", "Rightarrow": "⇒", "implies": "⇒", "Leftrightarrow": "⇔",
    "iff": "⇔", "leftarrow": "←", "leftrightarrow": "↔", "circ": "°", "degree": "°", "int": "∫", "iint": "∬",
    "oint": "∮", "sum": "∑", "prod": "∏", "partial": "∂", "nabla": "∇", "in": "∈", "notin": "∉", "subset": "⊂",
    "subseteq": "⊆", "cup": "∪", "cap": "∩", "forall": "∀", "exists": "∃", "angle": "∠", "triangle": "△",
    "perp": "⊥", "parallel": "∥", "propto": "∝", "ldots": "…", "dots": "…", "cdots": "⋯", "because": "∵",
    "therefore": "∴", "hbar": "ħ", "ell": "ℓ", "emptyset": "∅", "neg": "¬", "land": "∧", "lor": "∨",
    "ll": "≪", "gg": "≫", "lt": "<", "gt": ">", "quad": " ", "qquad": "  ", "prime": "′", "mid": "|",
}
_FUNCS = {"sin", "cos", "tan", "cot", "sec", "csc", "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan",
          "log", "ln", "exp", "min", "max", "det", "lim", "sup", "inf", "gcd", "lcm", "mod", "deg", "dim", "ker"}
_DROP = {"left", "right", "big", "Big", "bigg", "Bigg", "displaystyle", "limits", "nolimits", "textstyle",
         "scriptstyle", "tag", "nonumber", "label"}
_STRUCT = {"frac", "dfrac", "tfrac", "binom", "sqrt", "text", "textbf", "textit", "textrm", "mathrm", "mathbf",
           "mathit", "mathcal", "mathbb", "operatorname", "mbox", "boldsymbol", "bm", "vec", "hat", "bar",
           "tilde", "dot", "ddot", "overline", "begin", "end"}
_ACC = {"vec": "\u20d7", "hat": "\u0302", "bar": "\u0304", "tilde": "\u0303", "dot": "\u0307", "ddot": "\u0308"}
_BB = {"R": "ℝ", "N": "ℕ", "Z": "ℤ", "Q": "ℚ", "C": "ℂ", "P": "ℙ"}
_LB, _RB = "\ue000", "\ue001"  # placeholders for literal \{ and \}
_TRIG = "sin|cos|tan|cot|sec|csc|sinh|cosh|tanh|log|ln"


def _script(s, table, mark):
    """Convert s to superscript/subscript characters; fall back to ^x or ^(x+1) if a char has no symbol."""
    s = s.strip()
    if s and all(c in table for c in s):
        return "".join(table[c] for c in s)
    return mark + (s if re.fullmatch(r"\w+", s or "") else f"({s})")


def _outer_parens(s):
    if not (s.startswith("(") and s.endswith(")")):
        return False
    depth = 0
    for i, c in enumerate(s):
        depth += (c == "(") - (c == ")")
        if depth == 0 and i < len(s) - 1:
            return False
    return depth == 0


def _wrap(s):
    """Put brackets around s only when it is more than a single term."""
    s = s.strip()
    if not s or _outer_parens(s) or re.fullmatch(r"[−-]?[^\s+\-−×÷=/*]+", s):
        return s
    return f"({s})"


def _frac(m):
    return f"{_wrap(m.group(1))}/{_wrap(m.group(2))}"


def _root(m):
    n = m.group(1).strip()
    sign = {"2": "√", "3": "∛", "4": "∜"}.get(n) or (_script(n, _SUP, "^") + "√")
    return sign + _wrap(m.group(2))


def _strip_delims(s):
    """Remove $...$, $$...$$, \\( \\), \\[ \\] but keep real money such as '$5 and $6'."""
    def inline(m):
        inner = m.group(1)
        is_math = (inner == inner.strip()) and (re.search(r"[\\^_{}=]", inner) or not re.search(r"\s", inner))
        return inner if is_math else m.group(0)
    s = re.sub(r"\$\$(.+?)\$\$", lambda m: "\n" + m.group(1).strip() + "\n", s, flags=re.S)
    s = re.sub(r"\$([^$\n]+?)\$", inline, s)
    s = re.sub(r"\\\((.+?)\\\)", lambda m: m.group(1), s, flags=re.S)
    s = re.sub(r"\\\[(.+?)\\\]", lambda m: "\n" + m.group(1).strip() + "\n", s, flags=re.S)
    return s


def _commands(s, final=False):
    def rep(m):
        name = m.group(1)
        if name in _GREEK:
            return _GREEK[name]
        if name in _CMD:
            return _CMD[name]
        if name in _FUNCS:
            return name
        if name in _DROP:
            return ""
        if name in _STRUCT and not final:
            return m.group(0)
        return "" if name in _STRUCT else name
    return re.sub(r"\\([A-Za-z]+)", rep, s)


def _structure(s):
    """frac, sqrt, text, accents and ^{..} _{..}; innermost braces first, repeated until nothing changes."""
    prev = None
    for _ in range(12):
        if s == prev:
            break
        prev = s
        s = re.sub(r"\\(?:text|textbf|textit|textrm|mathrm|mathbf|mathit|mathcal|operatorname|mbox|boldsymbol|bm)"
                   r"\s*\{([^{}]*)\}", r"\1", s)
        s = re.sub(r"\\mathbb\s*\{([A-Z])\}", lambda m: _BB.get(m.group(1), m.group(1)), s)
        s = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", _frac, s)
        s = re.sub(r"\\[dt]?frac\s*(\d)\s*(\d)", lambda m: f"{m.group(1)}/{m.group(2)}", s)
        s = re.sub(r"\\binom\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"C(\1, \2)", s)
        s = re.sub(r"\\sqrt\s*\[([^\]]*)\]\s*\{([^{}]*)\}", _root, s)
        s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", lambda m: "√" + _wrap(m.group(1)), s)
        s = re.sub(r"\\sqrt\s*([A-Za-z0-9])", r"√\1", s)
        s = re.sub(r"\\(vec|hat|bar|tilde|dot|ddot)\s*\{([^{}]*)\}", lambda m: m.group(2) + _ACC[m.group(1)], s)
        s = re.sub(r"\\overline\s*\{([^{}]*)\}", lambda m: "".join(c + "\u0305" for c in m.group(1)), s)
        s = re.sub(r"\^\{([^{}]*)\}", lambda m: _script(m.group(1), _SUP, "^"), s)
        s = re.sub(r"_\{([^{}]*)\}", lambda m: _script(m.group(1), _SUB, "_"), s)
    return s


def _latex(seg):
    if not re.search(r"[\\$^_]", seg):
        return seg
    s = _strip_delims(seg)
    s = s.replace("\\{", _LB).replace("\\}", _RB)
    has_env = bool(re.search(r"\\begin\{|\\\\", s))
    s = s.replace("\\\\", "\n")
    s = re.sub(r"\\(?:left|right)\s*\.", "", s)
    s = re.sub(r"\^\s*\{\s*\\circ\s*\}|\^\s*\\circ", "°", s)
    s = re.sub(r"\\(?:begin|end)\{[A-Za-z*]+\}(?:\{[^{}]*\})?", "", s)
    if has_env:
        s = re.sub(r"[ \t]*&[ \t]*", " ", s)
    s = re.sub(r"\\[,;:]", " ", s)
    s = re.sub(r"\\[! ]", "", s)
    s = _commands(s)
    s = _structure(s)
    s = _commands(s, final=True)
    s = re.sub(r"\^(-?\d+|[A-Za-z])", lambda m: _script(m.group(1), _SUP, "^"), s)
    s = re.sub(r"\^\(([^()]+)\)", lambda m: _script(m.group(1), _SUP, "^")
               if all(c in _SUP for c in m.group(1).strip()) else m.group(0), s)
    s = re.sub(r"(?<=[A-Za-z)\]∫∑∏])_(-?\d+)(?![A-Za-z])", lambda m: _script(m.group(1), _SUB, "_"), s)
    s = re.sub(r"(?<=[A-Za-z)\]∫∑∏])_(n)(?![A-Za-z0-9])", lambda m: _script(m.group(1), _SUB, "_"), s)
    for a, b in (("\\%", "%"), ("\\&", "&"), ("\\_", "_"), ("\\$", "$"), ("\\#", "#")):
        s = s.replace(a, b)
    return s.replace(_LB, "{").replace(_RB, "}")


_MATHY = re.compile(r"\d|=|[+×÷²³]|\b(?:sin|cos|tan|cot|sec|csc|log|ln|lim|dx|dy)\b", re.I)
_OPERAND = r"(\([^()]*\)|[\w.²³ⁿ°θπ]+)"
_BEHIND_MATH = (r"(?:(?<=\d)|(?<=\d[A-Za-z])|(?<=\))|(?<=\])|(?<=\b[A-Za-z])|(?<=[θαβγδπφωλμσρτ])|(?<=[⁰¹²³⁴-⁹ⁿ])|"
                + "|".join(f"(?<={t})" for t in _TRIG.split("|")) + ")")


def _words(line):
    """Replace maths written in words by symbols, one line at a time."""
    s = line
    mathy = bool(_MATHY.search(s))
    I = re.I
    s = re.sub(r"\b(?:begin|end)(?:aligned|align|cases|array|[pb]?matrix)\b\s*", "", s)
    s = re.sub(r"\s*&=\s*", " = ", s)  # leftover from broken aligned blocks
    s = re.sub(r"\bdelta\s+(?=[A-Z])|\bDelta\s+(?=[A-Z])", "Δ", s)  # delta H, delta G -> ΔH, ΔG
    s = re.sub(r"\bplus[ -]or[ -]minus\b|\bplus[ -]minus\b", "±", s, flags=I)
    s = re.sub(r"\bminus[ -]or[ -]plus\b", "∓", s, flags=I)
    s = re.sub(r"\b(?:is\s+)?less than or equal to\b", "≤", s, flags=I)
    s = re.sub(r"\b(?:is\s+)?greater than or equal to\b", "≥", s, flags=I)
    s = re.sub(r"\b(?:is\s+)?not equal to\b", "≠", s, flags=I)
    s = re.sub(r"\b(?:is\s+)?approximately equal to\b", "≈", s, flags=I)
    s = re.sub(r"\binfinity\b", "∞", s, flags=I)
    if mathy:
        s = re.sub(r"\b(?:is\s+)?less than\b", "<", s, flags=I)
        s = re.sub(r"\b(?:is\s+)?greater than\b", ">", s, flags=I)
        s = re.sub(r"\bOmega\b", "Ω", s)
        s = re.sub(r"\b(theta|alpha|beta|gamma|delta|lambda|omega|sigma|phi|pi|mu|rho|tau|epsilon)\b",
                   lambda m: _GREEK[m.group(1)], s)
    # powers
    s = re.sub(r"\s*\b(?:raised to the power(?: of)?|to the power(?: of)?)\s+(-?\d+|\([^()]*\)|[A-Za-z]\b)",
               lambda m: _script(m.group(1).strip("()"), _SUP, "^"), s, flags=I)
    s = re.sub(_BEHIND_MATH + r"\s*\bsquared\b", "²", s, flags=I)
    s = re.sub(_BEHIND_MATH + r"\s*\bcubed\b", "³", s, flags=I)
    # roots, integrals, sums, limits
    s = re.sub(r"\bsquare root of\s+" + _OPERAND, r"√\1", s, flags=I)
    s = re.sub(r"\bcube root of\s+" + _OPERAND, r"∛\1", s, flags=I)
    s = re.sub(r"\bintegral from\s+(\S+)\s+to\s+(\S+?)[,]?\s+of\b\s*",
               lambda m: "∫" + _script(m.group(1), _SUB, "_") + _script(m.group(2), _SUP, "^") + " ", s, flags=I)
    s = re.sub(r"\b(?:sum|summation) from\s+(\S+)\s+to\s+(\S+?)[,]?\s+of\b\s*",
               lambda m: "∑" + _script(m.group(1), _SUB, "_") + _script(m.group(2), _SUP, "^") + " ", s, flags=I)
    s = re.sub(r"\bintegral of\b\s*", "∫ ", s, flags=I)
    s = re.sub(r"\blimit as\s+(\w+)\s+(?:approaches|tends to|goes to)\s+(\S+?)[,]?\s+of\b",
               r"lim(\1→\2)", s, flags=I)
    # arithmetic words
    for _ in range(4):
        s = re.sub(_OPERAND + r"\s+divided by\s+" + _OPERAND, r"\1/\2", s, flags=I)
    s = re.sub(r"(?<=[\d)])\s+(?:times|multiplied by)\s+(?=[\d(])", " × ", s, flags=I)
    if mathy:
        s = re.sub(r"\s+plus\s+", " + ", s, flags=I)
        s = re.sub(r"\s+minus\s+", " − ", s, flags=I)
        s = re.sub(r"\bplus\s+(?=[\d(])", "+", s, flags=I)
        s = re.sub(r"\bminus\s+(?=[\d(])", "−", s, flags=I)
        s = re.sub(r"\b(?:equals|is equal to)\b", "=", s, flags=I)
    s = re.sub(r"(?<=\d)\s*degrees?\b", "°", s, flags=I)
    s = re.sub(r"\b(" + _TRIG + r")([²³⁴ⁿ⁻¹]+)\s+(?=[A-Za-zθαβγφωπ(])", r"\1\2", s)
    return s


_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.S)


def symbolize(text):
    """Turn LaTeX and maths-in-words into symbols. Code blocks are left exactly as they are."""
    if not text:
        return text
    parts = _CODE.split(text)
    for i in range(0, len(parts), 2):
        seg = _latex(parts[i])
        parts[i] = "\n".join(_words(line) for line in seg.split("\n"))
    return "".join(parts)


# --------------------------------------------------------------------------
# LLM WRAPPER
# --------------------------------------------------------------------------
_client = None
_provider = None


def _gemini_key():
    """Find the Gemini key: env var first, then gemini_key.txt next to the script."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
    return key


def _init_client():
    global _client, _provider
    wanted = os.environ.get("EXAM_AI_PROVIDER", "").lower()
    gkey = _gemini_key()
    if wanted == "gemini" or (not wanted and gkey):
        if not gkey:
            sys.exit("No Gemini key found. Put it in gemini_key.txt or set GEMINI_API_KEY.")
        try:
            from google import genai
        except ImportError:
            sys.exit("Install the SDK first:  pip install google-genai")
        _client, _provider = genai.Client(api_key=gkey), "gemini"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
        except ImportError:
            sys.exit("Install the SDK first:  pip install anthropic")
        _client, _provider = anthropic.Anthropic(), "claude"
    else:
        sys.exit("No API key found. Put your Gemini key in gemini_key.txt "
                 "(or set GEMINI_API_KEY / ANTHROPIC_API_KEY).")


def llm(system, prompt, max_tokens=3000, temperature=0.0, math=True):
    """Call Gemini (default if a Gemini key exists) or Claude and return text.
    math=True adds the symbol rules to the prompt and converts leftover words/LaTeX to symbols."""
    if _client is None:
        _init_client()
    if math:
        system = system + "\n" + MATH_RULES
    if _provider == "gemini":
        from google.genai import types
        resp = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_tokens * 4,  # Gemini 2.5 thinking tokens count toward this
            ),
        )
        text = resp.text or ""
    else:
        resp = _client.messages.create(
            model=CLAUDE_MODEL, max_tokens=max_tokens, temperature=temperature,
            system=system, messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
    return symbolize(text) if math else text


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}

# --------------------------------------------------------------------------
# STORAGE + SEARCH  (dependency-free TF-IDF style retrieval)
# --------------------------------------------------------------------------
def load(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def save(path, data):
    DATA_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def search(query, items, key, k=4, exam=None):
    """Rank items by TF-IDF overlap between query and item[key]."""
    pool = [it for it in items if not exam or it.get("exam") == exam] or items
    if not pool:
        return []
    docs = [Counter(tokens(it[key])) for it in pool]
    df = Counter(t for d in docs for t in d)
    n = len(pool)
    q = tokens(query)
    scored = []
    for it, d in zip(pool, docs):
        total = sum(d.values()) or 1
        score = sum((d[t] / total) * math.log(1 + n / df[t]) for t in q if t in d)
        scored.append((score, it))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for s, it in scored[:k] if s > 0]

# --------------------------------------------------------------------------
# 1) BUILD CONCEPT BASE
# --------------------------------------------------------------------------
BUILD_SYSTEM = (
    "You are a senior teacher and exam-setter who has trained toppers for Indian and "
    "international competitive exams. Write precise, correct, exam-oriented concept notes."
)

BUILD_PROMPT = """Write a complete concept note.

Exam: {exam}
Subject: {subject}
Topic: {topic}

Use exactly these sections:
1. CORE IDEAS - every key definition/principle in short points
2. FORMULAS / THEOREMS / RULES - with symbols explained and conditions of validity
3. METHODS - step-by-step problem-solving strategies for this topic
4. COMMON TRAPS - mistakes students make and how to avoid them
5. HOW {exam} ASKS THIS - typical question styles and difficulty level
6. TWO WORKED EXAMPLES - one standard, one hard, fully solved

Be accurate. If the exam has negative marking or multiple-correct format, mention it."""


def build(exams=None, limit=None):
    store = load(CONCEPT_FILE)
    done = {(c["exam"], c["subject"], c["topic"]) for c in store}
    count = 0
    for exam, subjects in SYLLABUS.items():
        if exams and exam not in exams:
            continue
        for subject, topics in subjects.items():
            for topic in topics:
                if (exam, subject, topic) in done:
                    continue
                if limit and count >= limit:
                    save(CONCEPT_FILE, store)
                    print(f"Reached limit of {limit}. Saved.")
                    return
                print(f"[build] {exam} > {subject} > {topic}")
                try:
                    note = llm(BUILD_SYSTEM, BUILD_PROMPT.format(exam=exam, subject=subject, topic=topic), 3500)
                except Exception as e:  # keep going if one call fails
                    print("   failed:", e)
                    continue
                store.append({"exam": exam, "subject": subject, "topic": topic,
                              "text": f"{exam} {subject} {topic}\n{note}"})
                save(CONCEPT_FILE, store)  # save after each so progress is never lost
                count += 1
    print(f"Done. {count} new concept notes. Total: {len(store)}")

# --------------------------------------------------------------------------
# 2) SOLVE
# --------------------------------------------------------------------------
CLASSIFY_PROMPT = """Classify this exam question. Reply with JSON only:
{{"exam": "one of {exams} or Other", "subject": "...", "topics": ["2-4 specific topics needed"], "type": "numerical|mcq|multi-correct|proof|theory|coding|descriptive"}}

Question:
{q}"""

SOLVE_SYSTEM = (
    "You are an expert exam solver. Think carefully and solve step by step. "
    "Use the reference notes only if they help; they may be incomplete. "
    "Check units, signs, edge cases and every option for MCQs. "
    "End with one line exactly in the form:  FINAL ANSWER: <answer>"
)

SOLVE_PROMPT = """Exam: {exam} | Subject: {subject} | Type: {qtype}

REFERENCE NOTES:
{notes}

SIMILAR SOLVED QUESTIONS:
{examples}

QUESTION:
{q}

Solve completely. For proofs, give a rigorous proof. For descriptive/UPSC answers, give a structured answer.
End with: FINAL ANSWER: <answer>"""

VERIFY_PROMPT = """Check this solution independently. Re-derive key steps, check arithmetic, units and logic.

QUESTION:
{q}

PROPOSED SOLUTION:
{sol}

Reply in this format:
VERDICT: CORRECT or WRONG
CORRECTED ANSWER: <only if wrong, else 'same'>
REASON: <one or two lines>"""


def final_answer(text):
    m = re.findall(r"FINAL ANSWER:\s*(.+)", text, re.I)
    return m[-1].strip() if m else ""


def norm(ans):
    # keep symbols such as ² θ √ ±, otherwise "x²" and "x" would count as the same answer
    return re.sub(r"[\s.]+$", "", re.sub(r"\s+", "", ans.lower()))


def solve(question, exam=None, samples=3, verbose=True):
    concepts = load(CONCEPT_FILE)
    solved = load(SOLVED_FILE)

    # a) classify
    meta = parse_json(llm("You classify exam questions.", CLASSIFY_PROMPT.format(
        exams=", ".join(SYLLABUS), q=question), 400, math=False))
    exam = exam or meta.get("exam", "Other")
    subject = meta.get("subject", "General")
    qtype = meta.get("type", "numerical")
    topics = " ".join(meta.get("topics", []))
    if verbose:
        print(f"[classified] exam={exam} subject={subject} type={qtype} topics={topics}")

    # b) retrieve concepts + similar solved questions
    hits = search(f"{question} {topics}", concepts, "text", k=3, exam=exam if exam in SYLLABUS else None)
    notes = "\n\n---\n".join(h["text"][:3500] for h in hits) or "(no notes built yet - run `build`)"
    sims = search(question, solved, "question", k=2)
    examples = "\n\n".join(f"Q: {s['question']}\nA: {s['answer']}" for s in sims) or "(none)"

    # c) solve several times, then vote
    prompt = SOLVE_PROMPT.format(exam=exam, subject=subject, qtype=qtype, notes=notes, examples=examples, q=question)
    attempts = []
    for i in range(samples):
        temp = 0.0 if i == 0 else 0.7
        sol = llm(SOLVE_SYSTEM, prompt, 4000, temp)
        attempts.append((final_answer(sol), sol))
        if verbose:
            print(f"[attempt {i + 1}] {attempts[-1][0] or '(no answer line)'}")

    votes = Counter(norm(a) for a, _ in attempts if a)
    if votes:
        win = votes.most_common(1)[0][0]
        best_ans, best_sol = next((a, s) for a, s in attempts if norm(a) == win)
        agreement = votes[win] / len(attempts)
    else:
        best_ans, best_sol = attempts[0]
        agreement = 0.0

    # d) verify
    ver = llm("You are a strict examiner checking a solution.", VERIFY_PROMPT.format(q=question, sol=best_sol), 1200)
    wrong = re.search(r"VERDICT:\s*WRONG", ver, re.I)
    if wrong:
        m = re.search(r"CORRECTED ANSWER:\s*(.+)", ver, re.I)
        if m and m.group(1).strip().lower() != "same":
            best_ans = m.group(1).strip()

    result = {"question": question, "exam": exam, "subject": subject, "answer": best_ans,
              "solution": best_sol, "agreement": agreement, "verifier": ver.strip()}

    # e) remember
    solved.append({"question": question, "answer": best_ans, "exam": exam})
    save(SOLVED_FILE, solved)
    return result


def show(r):
    print("\n" + "=" * 70)
    print(r["solution"])
    print("=" * 70)
    print(f"ANSWER: {r['answer']}")
    print(f"Agreement across attempts: {r['agreement']:.0%}")
    print(f"Verifier: {r['verifier']}")
    if r["agreement"] < 0.67:
        print("WARNING: attempts disagreed - double-check this answer.")

# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def fix_saved():
    """Convert math words/LaTeX in already-built notes and solved questions to symbols (no API calls)."""
    for path in (CONCEPT_FILE, SOLVED_FILE):
        if not path.exists():
            continue
        path.with_suffix(".json.bak").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        data = load(path)
        for item in data:
            for k in ("text", "answer", "solution"):
                if isinstance(item.get(k), str):
                    item[k] = symbolize(item[k])
        save(path, data)
        print(f"Converted {len(data)} items in {path} (backup: {path.with_suffix('.json.bak')})")


def stats():
    store = load(CONCEPT_FILE)
    total = sum(len(t) for s in SYLLABUS.values() for t in s.values())
    print(f"Concept notes built: {len(store)} / {total} topics")
    for exam, subjects in SYLLABUS.items():
        n = sum(1 for c in store if c["exam"] == exam)
        t = sum(len(x) for x in subjects.values())
        print(f"  {exam:16s} {n}/{t}")
    print(f"Solved questions remembered: {len(load(SOLVED_FILE))}")


def main():
    p = argparse.ArgumentParser(description="Exam concept builder and solver")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build the concept base")
    b.add_argument("--exam", nargs="*", help="only these exams, e.g. JEE NEET")
    b.add_argument("--limit", type=int, help="max new notes this run")

    a = sub.add_parser("ask", help="solve one question")
    a.add_argument("question")
    a.add_argument("--exam")
    a.add_argument("--samples", type=int, default=3)

    c = sub.add_parser("chat", help="interactive solver")
    c.add_argument("--samples", type=int, default=3)

    sub.add_parser("stats", help="show build progress")
    sub.add_parser("fix", help="convert words/LaTeX in saved notes to symbols (no API calls)")

    try:  # make sure symbols print correctly on Windows terminals and when output is piped
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = p.parse_args()
    if args.cmd == "build":
        build(args.exam, args.limit)
    elif args.cmd == "ask":
        show(solve(args.question, args.exam, args.samples))
    elif args.cmd == "chat":
        print("Exam AI ready. Paste a question (type 'exit' to quit).")
        while True:
            q = input("\nQuestion> ").strip()
            if q.lower() in {"exit", "quit"}:
                break
            if q:
                show(solve(q, samples=args.samples))
    elif args.cmd == "stats":
        stats()
    elif args.cmd == "fix":
        fix_saved()


if __name__ == "__main__":
    main()
