"""
AI Detective: A Knowledge-Based Mystery Solving Agent
CSE518 Artificial Intelligence, Monsoon 2026 - Streamlit interface

Run locally:   streamlit run app.py
The solver code below is copied unchanged from the project notebook (Sections 1-8 and 10).
The interface code at the end only calls those functions through small wrappers.
Editable clues: new wrapper code builds fresh constraint lists and a fresh KB for each script run.
"""
import hashlib
import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# =====================================================================
# SOLVER CODE (copied unchanged from the notebook, no prints or tests)
# =====================================================================
# ----- from notebook cell 1 -----
from itertools import product

# ----- from notebook cell 3 -----
SUSPECTS  = ["Anita", "Rohan", "Meera", "Kabir"]

WEAPONS   = ["Knife", "Poison", "Trophy", "Scarf"]

LOCATIONS = ["Library", "Kitchen", "Garden", "Balcony"]

TIMES     = ["8 PM", "9 PM", "10 PM"]

ROLES = {"Anita": "business partner", "Rohan": "nephew",
         "Meera": "secretary", "Kabir": "chef"}

TOTAL_HYPOTHESES = len(SUSPECTS) * len(WEAPONS) * len(LOCATIONS) * len(TIMES)

# ----- from notebook cell 5 -----
variables = ["Culprit", "Weapon", "Location", "Time"]

# list() makes a copy so the original lists are not changed
domains = {
    "Culprit":  list(SUSPECTS),
    "Weapon":   list(WEAPONS),
    "Location": list(LOCATIONS),
    "Time":     list(TIMES),
}

# ----- from notebook cell 6 -----
# IF a THEN b  is the same as  (not a) or b
def implies(a, b):
    return (not a) or b

# (variable, condition, clue)
unary_constraints = [
    ("Weapon",   lambda w: w != "Poison",     "Clue 1: No poison found in the body."),
    ("Location", lambda loc: loc != "Garden", "Clue 2: No footprints in the garden."),
    ("Time",     lambda t: t != "8 PM",       "Clue 3: Vikram was alive at 8 PM."),
]

# (variable 1, variable 2, condition, clue)
binary_constraints = [
    ("Weapon", "Location", lambda w, loc: implies(w == "Knife", loc == "Kitchen"),
     "Clue 4: No knife ever left the kitchen."),
    ("Location", "Time", lambda loc, t: implies(loc == "Kitchen", t == "8 PM"),
     "Clue 5: Kitchen helpers were in the kitchen all evening except at 8 PM, "
     "so nobody could commit a murder there at 9 PM or 10 PM without being seen."),
    ("Culprit", "Location", lambda c, loc: implies(c == "Kabir", loc == "Kitchen"),
     "Clue 6: Kabir never left the kitchen."),
    ("Weapon", "Location", lambda w, loc: implies(w == "Trophy", loc == "Library"),
     "Clue 7: The trophy never left the library."),
    ("Location", "Time", lambda loc, t: implies(loc == "Library", t != "10 PM"),
     "Clue 8: The library was locked at 9:30 PM."),
    ("Location", "Time", lambda loc, t: implies(loc == "Balcony", t != "9 PM"),
     "Clue 9: The balcony was crowded at 9 PM."),
    ("Culprit", "Weapon", lambda c, w: implies(w == "Scarf", c in ["Anita", "Meera"]),
     "Clue 10: Only Anita and Meera could get the scarf."),
    ("Culprit", "Weapon", lambda c, w: implies(c == "Anita", w != "Trophy"),
     "Clue 11: Anita cannot lift the trophy."),
    ("Culprit", "Time", lambda c, t: implies(c == "Meera", t != "9 PM"),
     "Clue 12: Meera was on a phone call at 9 PM."),
    ("Culprit", "Time", lambda c, t: implies(c == "Anita", t != "10 PM"),
     "Clue 13: Anita left the party at 9:45 PM."),
]

# ----- from notebook cell 7 -----
# returns the clues broken by a (full or partial) assignment
# a clue is only checked when all of its variables have a value
def broken_clues(assignment):
    broken = []
    for var, rule, clue in unary_constraints:
        if var in assignment and not rule(assignment[var]):
            broken.append(clue)
    for var1, var2, rule, clue in binary_constraints:
        if var1 in assignment and var2 in assignment:
            if not rule(assignment[var1], assignment[var2]):
                broken.append(clue)
    return broken

def is_consistent(assignment):
    return len(broken_clues(assignment)) == 0

# hypothesis as a readable sentence
def describe(h):
    return f"{h['Culprit']} ({ROLES[h['Culprit']]}) with the {h['Weapon']} in the {h['Location']} at {h['Time']}"

# ----- from notebook cell 9 -----
def apply_unary_constraints(doms, verbose=False):
    new_doms = {var: list(values) for var, values in doms.items()}
    for var, rule, clue in unary_constraints:
        for value in list(new_doms[var]):
            if not rule(value):
                new_doms[var].remove(value)
                if verbose:
                    print(f"  removed {value} from {var}  ({clue})")
    return new_doms

# variables that share a binary constraint with var
def neighbours(var):
    result = []
    for var1, var2, rule, clue in binary_constraints:
        if var1 == var and var2 not in result:
            result.append(var2)
        if var2 == var and var1 not in result:
            result.append(var1)
    return result

# removes values of x that have no supporting value in y
def revise(doms, x, y, verbose=False):
    removed = False
    for vx in list(doms[x]):
        has_support = False
        for vy in doms[y]:
            if is_consistent({x: vx, y: vy}):
                has_support = True
                break
        if not has_support:
            doms[x].remove(vx)
            removed = True
            if verbose:
                print(f"  removed {vx} from {x}  (no value of {y} works with it)")
    return removed

def ac3(doms, verbose=False):
    doms = {var: list(values) for var, values in doms.items()}

    queue = []
    for var1, var2, rule, clue in binary_constraints:
        for arc in [(var1, var2), (var2, var1)]:
            if arc not in queue:
                queue.append(arc)

    while queue:
        x, y = queue.pop(0)
        if revise(doms, x, y, verbose):
            if len(doms[x]) == 0:
                return None   # no solution
            for z in neighbours(x):
                if z != y and (z, x) not in queue:
                    queue.append((z, x))
    return doms

def print_domains(doms):
    size = 1
    for var in variables:
        print(f"  {var:<9}: {doms[var]}")
        size = size * len(doms[var])
    print(f"  combinations left: {size}")

# ----- from notebook cell 11 -----
def select_unassigned_variable(assignment):
    for var in variables:
        if var not in assignment:
            return var

def backtrack(assignment, doms, solutions, stats):
    if len(assignment) == len(variables):
        solutions.append(dict(assignment))
        return

    var = select_unassigned_variable(assignment)
    for value in doms[var]:
        stats["nodes"] += 1
        assignment[var] = value
        if is_consistent(assignment):
            backtrack(assignment, doms, solutions, stats)
        del assignment[var]   # undo and try the next value

def backtracking_search(doms):
    solutions = []
    stats = {"nodes": 0}
    backtrack({}, doms, solutions, stats)
    return solutions, stats["nodes"]

# ----- from notebook cell 17 -----
# one symbol per value, e.g. sym("Culprit", "Rohan") -> "C_Rohan", sym("Time", "9 PM") -> "T_9PM"
def sym(var, value):
    return var[0] + "_" + value.replace(" ", "")

SYMBOLS = [sym(var, value) for var in variables for value in domains[var]]

def exactly_one(var):
    return lambda m: sum(m[sym(var, v)] for v in domains[var]) == 1

# knowledge base: list of (name, sentence), a sentence is a function of the model
KB = [
    ("Exactly one culprit",  exactly_one("Culprit")),
    ("Exactly one weapon",   exactly_one("Weapon")),
    ("Exactly one location", exactly_one("Location")),
    ("Exactly one time",     exactly_one("Time")),
    ("Clue 1",  lambda m: not m["W_Poison"]),
    ("Clue 2",  lambda m: not m["L_Garden"]),
    ("Clue 3",  lambda m: not m["T_8PM"]),
    ("Clue 4",  lambda m: implies(m["W_Knife"], m["L_Kitchen"])),
    ("Clue 5",  lambda m: implies(m["L_Kitchen"], m["T_8PM"])),
    ("Clue 6",  lambda m: implies(m["C_Kabir"], m["L_Kitchen"])),
    ("Clue 7",  lambda m: implies(m["W_Trophy"], m["L_Library"])),
    ("Clue 8",  lambda m: implies(m["L_Library"], not m["T_10PM"])),
    ("Clue 9",  lambda m: implies(m["L_Balcony"], not m["T_9PM"])),
    ("Clue 10", lambda m: implies(m["W_Scarf"], m["C_Anita"] or m["C_Meera"])),
    ("Clue 11", lambda m: implies(m["C_Anita"], not m["W_Trophy"])),
    ("Clue 12", lambda m: implies(m["C_Meera"], not m["T_9PM"])),
    ("Clue 13", lambda m: implies(m["C_Anita"], not m["T_10PM"])),
]

def kb_true(m):
    return all(sentence(m) for name, sentence in KB)

# KB entails alpha if alpha is true in every model of the KB
def entails(alpha):
    return all(alpha(m) for m in kb_models)

# answers a yes / no question about one symbol
def ask(s):
    if entails(lambda m: m[s]):
        return "TRUE (proven)"
    if entails(lambda m: not m[s]):
        return "FALSE (proven)"
    return "UNKNOWN"

# ----- from notebook cell 18 -----
# symbols that are true in a hypothesis
def h_symbols(h):
    return [sym(var, h[var]) for var in variables]

# ----- from notebook cell 20 -----
BN_NODES = [
    "H",
    "Motive",
    "Fingerprint",
    "Argument",
    "Sighting"
]

# Directed edges: H -> each evidence node
BN_EDGES = [
    ("H", "Motive"),
    ("H", "Fingerprint"),
    ("H", "Argument"),
    ("H", "Sighting")
]

EVIDENCE_NODES = ["Motive", "Fingerprint", "Argument", "Sighting"]

# 2. Prior probability of each feasible hypothesis
def uniform_prior(hyps):
    if len(hyps) == 0:
        return []
    return [1.0 / len(hyps)] * len(hyps)

EVIDENCE = {
    "Motive": {
        "text": "Rohan has gambling debts and gets money from the will",
        "p_true": lambda h: (
            0.80 if h["Culprit"] == "Rohan" else 0.30
        ),
    },

    "Fingerprint": {
        "text": "Partial fingerprint on the trophy, about 70% match with Rohan",
        "p_true": lambda h: (
            0.70
            if h["Culprit"] == "Rohan"
            and h["Weapon"] == "Trophy"
            else 0.15
        ),
    },

    "Argument": {
        "text": "Anita and Vikram argued in the morning",
        "p_true": lambda h: (
            0.60 if h["Culprit"] == "Anita" else 0.25
        ),
    },

    "Sighting": {
        "text": "A guest saw someone on the balcony around 10 PM",
        "p_true": lambda h: (
            0.60
            if h["Location"] == "Balcony"
            and h["Time"] == "10 PM"
            else 0.15
        ),
    },
}

# 4. Validate network structure and CPT probabilities
def validate_bn():
    assert ("H", "Motive") in BN_EDGES
    assert ("H", "Fingerprint") in BN_EDGES
    assert ("H", "Argument") in BN_EDGES
    assert ("H", "Sighting") in BN_EDGES

    assert set(EVIDENCE) == set(EVIDENCE_NODES)

    for h in logic_hypotheses:
        for name in EVIDENCE_NODES:
            p = EVIDENCE[name]["p_true"](h)
            assert 0.0 <= p <= 1.0

def cpt_probability(h, node, value):
    p_true = EVIDENCE[node]["p_true"](h)
    return p_true if value else 1.0 - p_true

def joint_probability(h, observed, prior=None):
    if prior is None:
        prior = uniform_prior(logic_hypotheses)

    index = logic_hypotheses.index(h)
    p = prior[index]

    for name, value in observed.items():
        if name not in EVIDENCE:
            raise ValueError(f"Unknown evidence node: {name}")

        p *= cpt_probability(h, name, value)

    return p

def posterior(hyps, observed, prior=None):
    if len(hyps) == 0:
        return []

    if prior is None:
        prior = uniform_prior(hyps)

    if len(prior) != len(hyps):
        raise ValueError("Prior length must match hypothesis count.")

    if any(p < 0 for p in prior):
        raise ValueError("Prior probabilities cannot be negative.")

    if abs(sum(prior) - 1.0) > 1e-9:
        raise ValueError("Prior probabilities must sum to 1.")

    joint = []

    for i, h in enumerate(hyps):
        p = prior[i]

        for name, value in observed.items():
            if name not in EVIDENCE:
                raise ValueError(f"Unknown evidence node: {name}")

            p *= cpt_probability(h, name, value)

        joint.append(p)

    total = sum(joint)

    if total <= 0:
        raise ValueError(
            "Observed evidence has zero probability under all hypotheses."
        )

    return [p / total for p in joint]

# 8. Rank hypotheses by posterior probability
def rank_hypotheses(hyps, observed, prior=None):
    probs = posterior(hyps, observed, prior)

    return sorted(
        zip(hyps, probs),
        key=lambda pair: pair[1],
        reverse=True
    )

# 9. Bayes factor for comparing two hypotheses
def bayes_factor(h, other, name, value=True):
    a = cpt_probability(h, name, value)
    b = cpt_probability(other, name, value)

    if b == 0:
        return float("inf") if a > 0 else 1.0

    return a / b

# 10. All four uncertain clues are initially reported
observed = {name: True for name in EVIDENCE_NODES}

# ----- from notebook cell 22 -----
STATES = ["Alive", "Killed", "Moved"]

STEPS = ["8 PM", "9 PM", "10 PM", "Morning"]

# P(first state)
START = {"Alive": 1.0, "Killed": 0.0, "Moved": 0.0}      # clue 3: alive at 8 PM

# P(next state | state)  (one step = one clock hour, last step = overnight)
TRANS = {
    "Alive":  {"Alive": 0.50, "Killed": 0.50, "Moved": 0.00},
    "Killed": {"Alive": 0.00, "Killed": 0.40, "Moved": 0.60},
    "Moved":  {"Alive": 0.00, "Killed": 0.00, "Moved": 1.00},
}

# P(observation | state)
OBSERVATIONS = ["SeenAlive", "Normal", "Suspicious", "BodyFound"]

EMIT = {
    "Alive":  {"SeenAlive": 0.60, "Normal": 0.37, "Suspicious": 0.03, "BodyFound": 0.00},
    "Killed": {"SeenAlive": 0.00, "Normal": 0.45, "Suspicious": 0.50, "BodyFound": 0.05},
    "Moved":  {"SeenAlive": 0.00, "Normal": 0.55, "Suspicious": 0.20, "BodyFound": 0.25},
}

# clues in the order they arrive
CLUE_STREAM = ["SeenAlive", "Normal", "Suspicious", "BodyFound"]

def forward(obs, start=START, trans=TRANS, emit=EMIT):
    """filtering: belief over the current state after each clue (normalised)."""
    beliefs, alpha = [], {s: start[s] * emit[s][obs[0]] for s in STATES}
    for t in range(len(obs)):
        if t > 0:
            alpha = {s: sum(beliefs[-1][p] * trans[p][s] for p in STATES) * emit[s][obs[t]] for s in STATES}
        total = sum(alpha.values())
        beliefs.append({s: alpha[s] / total for s in STATES})
    return beliefs

def forward_backward(obs, start=START, trans=TRANS, emit=EMIT):
    """smoothing: belief over the state at every step using ALL clues."""
    n = len(obs)
    fwd = []
    a = {s: start[s] * emit[s][obs[0]] for s in STATES}
    fwd.append(a)
    for t in range(1, n):
        a = {s: sum(fwd[-1][p] * trans[p][s] for p in STATES) * emit[s][obs[t]] for s in STATES}
        fwd.append(a)
    bwd = [None] * n
    bwd[n - 1] = {s: 1.0 for s in STATES}
    for t in range(n - 2, -1, -1):
        bwd[t] = {s: sum(trans[s][q] * emit[q][obs[t + 1]] * bwd[t + 1][q] for q in STATES) for s in STATES}
    out = []
    for t in range(n):
        g = {s: fwd[t][s] * bwd[t][s] for s in STATES}
        z = sum(g.values())
        out.append({s: g[s] / z for s in STATES})
    return out

def viterbi(obs, start=START, trans=TRANS, emit=EMIT):
    """most likely hidden sequence."""
    v = [{s: start[s] * emit[s][obs[0]] for s in STATES}]
    back = []
    for t in range(1, len(obs)):
        v.append({}); back.append({})
        for s in STATES:
            best_prev = max(STATES, key=lambda p: v[t - 1][p] * trans[p][s])
            v[t][s] = v[t - 1][best_prev] * trans[best_prev][s] * emit[s][obs[t]]
            back[t - 1][s] = best_prev
    last = max(STATES, key=lambda s: v[-1][s])
    path = [last]
    for t in range(len(obs) - 2, -1, -1):
        path.insert(0, back[t][path[0]])
    return path, v[-1][last]

def murder_time_distribution(obs, allowed=("9 PM", "10 PM")):
    """P(murder at each time): sum the probability of every hidden path where the first
    Killed state is at that time. Only the times the CSP allows are kept, then renormalised."""
    n = len(obs); result = {}
    for k in range(1, n):          # 8 PM is impossible (clue 3: alive)
        if STEPS[k] not in allowed:
            continue
        # path prob = alive until k-1, killed at k, anything after
        p = START["Alive"] * EMIT["Alive"][obs[0]]
        for t in range(1, k):
            p *= TRANS["Alive"]["Alive"] * EMIT["Alive"][obs[t]]
        p *= TRANS["Alive"]["Killed"] * EMIT["Killed"][obs[k]]
        # sum over continuations from Killed at step k
        cont = {s: (1.0 if s == "Killed" else 0.0) for s in STATES}
        for t in range(k + 1, n):
            cont = {s: sum(cont[q] * TRANS[q][s] for q in STATES) * EMIT[s][obs[t]] for s in STATES}
        result[STEPS[k]] = p * sum(cont.values())
    z = sum(result.values())
    return {k: v / z for k, v in result.items()}

# ----- from notebook cell 26 -----
from math import log2

# ---------- 1. Bayesian Network posterior ranking ----------
# Uses exact inference from the explicitly defined Bayesian Network.
# For each observed clue, compare how likely the clue is under this hypothesis
# with how likely it is under the other hypotheses (ratio > 1 supports, < 1 conflicts).
SUPPORT_LIMIT = 1.2      # ratio above this counts as supporting

CONFLICT_LIMIT = 0.8     # ratio below this counts as conflicting

def clue_ratio(h, others, name, value):
    p_h = EVIDENCE[name]["p_true"](h)
    p_o = sum(EVIDENCE[name]["p_true"](o) for o in others) / len(others)
    if not value:
        p_h, p_o = 1 - p_h, 1 - p_o
    return p_h / p_o

def build_ranking(hyps, observed, clue_stream=None):
    """Returns a list of dicts (best first) with everything the interface needs."""
    probs = posterior(hyps, observed)
    time_probs = murder_time_distribution(clue_stream) if clue_stream else {}
    rows = []
    for i, h in enumerate(hyps):
        others = [o for j, o in enumerate(hyps) if j != i]
        supports, conflicts = [], []
        if others:
            for name, value in observed.items():
                ratio = clue_ratio(h, others, name, value)
                if ratio > SUPPORT_LIMIT:
                    supports.append((name, ratio))
                elif ratio < CONFLICT_LIMIT:
                    conflicts.append((name, ratio))
        rows.append({
            "hypothesis": h,
            "probability": probs[i],
            "supports": sorted(supports, key=lambda x: -x[1]),
            "conflicts": sorted(conflicts, key=lambda x: x[1]),
            "hmm_time_prob": time_probs.get(h["Time"]),
        })
    return sorted(rows, key=lambda r: r["probability"], reverse=True)

# ---------- 2. confidence: say when the evidence is not enough ----------
STRONG_TOP, STRONG_MARGIN = 0.80, 0.30

LIKELY_TOP, LIKELY_MARGIN = 0.50, 0.20

def verdict(rows):
    if len(rows) == 0:
        return "NO HYPOTHESIS", "The clues contradict each other."
    if len(rows) == 1:
        return "CONCLUSIVE", "Only one hypothesis is consistent with the clues."
    top, second = rows[0]["probability"], rows[1]["probability"]
    margin = top - second
    if top >= STRONG_TOP and margin >= STRONG_MARGIN:
        return "STRONG", f"Top hypothesis leads by {margin:.2f}."
    if top >= LIKELY_TOP and margin >= LIKELY_MARGIN:
        return "LIKELY", f"Top hypothesis leads by {margin:.2f}, but it is not conclusive."
    return "INCONCLUSIVE", f"Only {margin:.2f} between the top two, more evidence is needed."

# ---------- 3. decision layer: which clue should we look for next? ----------
def entropy(probs):
    return -sum(p * log2(p) for p in probs if p > 0)

def information_gain(hyps, observed, name):
    """Expected drop in entropy if we find out whether clue `name` is True or False."""
    now = posterior(hyps, observed)
    expected = 0.0
    for value in (True, False):
        p_value = sum(now[i] * (EVIDENCE[name]["p_true"](h) if value else 1 - EVIDENCE[name]["p_true"](h))
                      for i, h in enumerate(hyps))
        if p_value > 0:
            after = posterior(hyps, {**observed, name: value})
            expected += p_value * entropy(after)
    return entropy(now) - expected

def next_clue(hyps, observed):
    """The not-yet-known clue with the highest information gain (None if all are known)."""
    options = [(name, information_gain(hyps, observed, name)) for name in EVIDENCE if name not in observed]
    if len(options) == 0:
        return None
    return sorted(options, key=lambda x: -x[1])


# =====================================================================
# WRAPPERS FOR THE INTERFACE (new code, the solver functions are not changed)
# =====================================================================
EVIDENCE_CHOICES = ["True", "False", "Unknown"]
DEFAULT_OBSERVED = dict(observed)          # notebook default: all four items reported True
ASK_LABELS = {"TRUE (proven)": "Proven true", "FALSE (proven)": "Proven false", "UNKNOWN": "Unknown"}
CASE_CLUES = [
    "The autopsy found no poison in the body.",
    "It rained all evening and there were no footprints in the garden.",
    "Vikram gave a speech to all the guests at 8 PM, so he was alive at 8 PM.",
    "No knife ever left the kitchen.",
    "Kitchen helpers were in the kitchen all evening except at 8 PM, so nobody could commit a murder "
    "there at 9 PM or 10 PM without being seen.",
    "CCTV shows that Kabir never left the kitchen.",
    "The trophy is heavy and never left the library.",
    "The library was locked at 9:30 PM.",
    "At 9 PM a lot of guests were on the balcony watching fireworks.",
    "The scarf was kept in the office and only Anita and Meera had the office key.",
    "Anita has a broken wrist and cannot lift the trophy.",
    "Meera was on a recorded phone call at 9 PM.",
    "Anita left the party at 9:45 PM.",
]
ENCODED_CLUES = [c for _, _, c in unary_constraints] + [c for _, _, _, c in binary_constraints]
ENCODED_NUMBERS = sorted(int(re.search(r"Clue (\d+)", c).group(1)) for c in ENCODED_CLUES)


# =====================================================================
# EDITABLE CLUES (new wrapper code, the solver functions are not changed)
# =====================================================================
# The solver functions read the globals unary_constraints, binary_constraints, KB and kb_models.
# Streamlit runs this script in a fresh module for every rerun of every session, so these globals
# belong to one run of one user only. use_clues() points them at NEW lists built from the clue
# settings. The notebook lists below and the cached models are never changed in place.
BASE_UNARY = unary_constraints        # the notebook's lists = the original case
BASE_BINARY = binary_constraints
BASE_KB = KB

# dropdown values of the original case
DEFAULT_CLUES = {
    "c1_found": "found no", "c1_weapon": "Poison",
    "c3_time": "8 PM",
    "c6_suspect": "Kabir", "c6_location": "Kitchen",
    "c7_weapon": "Trophy", "c7_location": "Library",
    "c11_suspect": "Anita",
}
CLUE_NUMBERS = list(range(1, len(CASE_CLUES) + 1))
DEFAULT_CLUES.update({f"on_{n}": True for n in CLUE_NUMBERS})   # checkbox of each clue: ticked = used
EDITABLE = {1: ["c1_found", "c1_weapon"], 3: ["c3_time"],
            6: ["c6_suspect", "c6_location"], 7: ["c7_weapon", "c7_location"],
            11: ["c11_suspect"]}
FOUND_CHOICES = ["found", "found no"]
ALIVE_TIMES = ["8 PM", "9 PM"]     # alive at 10 PM would leave no possible murder time
# readable autopsy finding for each weapon (the dropdown still stores the weapon name)
WEAPON_SIGNS = {"Poison": "signs of poison", "Knife": "knife wounds",
                "Trophy": "a heavy blow from the trophy", "Scarf": "strangulation marks from the scarf"}
STALE_TEXT = "Clues changed, press Solve Case to update."
MODIFIED_NOTE = ("The Bayesian CPTs and HMM were written for the original story, "
                 "so probabilities here are illustrative only.")

# propositional form of each clue in the original case (display text only, ~ is NOT)
FORMULAS = {
    1: "~W_Poison", 2: "~L_Garden", 3: "~T_8PM",
    4: "W_Knife ⇒ L_Kitchen", 5: "L_Kitchen ⇒ T_8PM", 6: "C_Kabir ⇒ L_Kitchen",
    7: "W_Trophy ⇒ L_Library", 8: "L_Library ⇒ ~T_10PM", 9: "L_Balcony ⇒ ~T_9PM",
    10: "W_Scarf ⇒ (C_Anita ∨ C_Meera)", 11: "C_Anita ⇒ ~W_Trophy",
    12: "C_Meera ⇒ ~T_9PM", 13: "C_Anita ⇒ ~T_10PM",
}


def clue_number(text):
    found = re.search(r"Clue (\d+)", text)
    return int(found.group(1)) if found else None


def is_original(c, n):
    """True if editable clue n has the same dropdown values as the original case."""
    return all(c[k] == DEFAULT_CLUES[k] for k in EDITABLE[n])


def is_on(c, n):
    """True if clue n is ticked (used in the CSP and the KB)."""
    return c.get(f"on_{n}", True)


def changed_clues(c):
    """Ticked dropdown clues whose values differ from the original case (unticked ones are not used)."""
    return [n for n in EDITABLE if is_on(c, n) and not is_original(c, n)]


def off_clues(c):
    return [n for n in CLUE_NUMBERS if not is_on(c, n)]


def is_modified(c):
    return bool(changed_clues(c) or off_clues(c))


def keep_original(n, c):
    """Use the notebook's own entry unless n is an editable clue that was changed."""
    return n not in EDITABLE or is_original(c, n)


def times_up_to(t):
    """t and every earlier time: if Vikram was alive at t, the murder was not at any of these."""
    return TIMES[:TIMES.index(t) + 1]


def clue_sentence(n, c):
    if n == 1:
        return f"The autopsy {c['c1_found']} {WEAPON_SIGNS[c['c1_weapon']]}."
    if n == 3:
        return f"Vikram was still alive at {c['c3_time']}, so the murder happened later."
    if n == 6:
        return f"CCTV shows {c['c6_suspect']} never left the {c['c6_location'].lower()}."
    if n == 7:
        return f"The {c['c7_weapon'].lower()} never left the {c['c7_location'].lower()}."
    if n == 11:
        return f"{c['c11_suspect']} has a broken wrist and cannot lift the trophy."
    return CASE_CLUES[n - 1]


def clue_formula(n, c):
    if n == 1:
        s = sym("Weapon", c["c1_weapon"])
        return s if c["c1_found"] == "found" else "~" + s
    if n == 3:
        return " ∧ ".join("~" + sym("Time", t) for t in times_up_to(c["c3_time"]))
    if n == 6:
        return sym("Culprit", c["c6_suspect"]) + " ⇒ " + sym("Location", c["c6_location"])
    if n == 7:
        return sym("Weapon", c["c7_weapon"]) + " ⇒ " + sym("Location", c["c7_location"])
    if n == 11:
        return sym("Culprit", c["c11_suspect"]) + " ⇒ ~W_Trophy"
    return FORMULAS[n]


def hypothesis_formula(h):
    return " ∧ ".join(h_symbols(h))


def new_constraint(n, c):
    """CSP constraint for a changed clue, in the same tuple format as the notebook lists."""
    text = f"Clue {n}: {clue_sentence(n, c)}"
    if n == 1:
        w = c["c1_weapon"]
        if c["c1_found"] == "found":
            return ("Weapon", lambda x: x == w, text)
        return ("Weapon", lambda x: x != w, text)
    if n == 3:
        gone = times_up_to(c["c3_time"])
        return ("Time", lambda x: x not in gone, text)      # the murder happened after the chosen time
    if n == 6:
        s, loc = c["c6_suspect"], c["c6_location"]
        return ("Culprit", "Location", lambda x, y: implies(x == s, y == loc), text)
    if n == 11:
        s = c["c11_suspect"]
        return ("Culprit", "Weapon", lambda x, w: implies(x == s, w != "Trophy"), text)
    w, loc = c["c7_weapon"], c["c7_location"]
    return ("Weapon", "Location", lambda x, y: implies(x == w, y == loc), text)


def new_sentence(n, c):
    """KB sentence for a changed clue."""
    if n == 1:
        s = sym("Weapon", c["c1_weapon"])
        if c["c1_found"] == "found":
            return lambda m: m[s]
        return lambda m: not m[s]
    if n == 3:
        gone = [sym("Time", t) for t in times_up_to(c["c3_time"])]
        return lambda m: not any(m[s] for s in gone)
    if n == 11:
        s = sym("Culprit", c["c11_suspect"])
        return lambda m: implies(m[s], not m["W_Trophy"])
    if n == 6:
        a, b = sym("Culprit", c["c6_suspect"]), sym("Location", c["c6_location"])
    else:
        a, b = sym("Weapon", c["c7_weapon"]), sym("Location", c["c7_location"])
    return lambda m: implies(m[a], m[b])


def build_constraints(c):
    """New unary and binary lists for these clue settings (clue order is kept, unticked clues are left out)."""
    unary, binary = [], []
    for item in BASE_UNARY:
        n = clue_number(item[-1])
        if is_on(c, n):
            unary.append(item if keep_original(n, c) else new_constraint(n, c))
    for item in BASE_BINARY:
        n = clue_number(item[-1])
        if is_on(c, n):
            binary.append(item if keep_original(n, c) else new_constraint(n, c))
    return unary, binary


def build_kb(c):
    """New KB for these clue settings. The 'exactly one' sentences are always kept; unticked clues are left out."""
    kb = []
    for name, sentence in BASE_KB:
        n = clue_number(name)
        if n is None or is_on(c, n):
            kb.append((name, sentence) if keep_original(n, c) else (name, new_sentence(n, c)))
    return kb


@st.cache_data(show_spinner=False)
def load_kb_models(source_version, clue_key):
    """Same truth-table loop as the notebook, run on this run's KB. Returns plain dicts only.
    The key is a hash of this file plus the clue settings (dropdowns and ticks), so each setting has its own entry."""
    models = []
    for values in product([True, False], repeat=len(SYMBOLS)):
        m = dict(zip(SYMBOLS, values))
        if kb_true(m):
            models.append(m)
    return models


SOURCE_VERSION = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def use_clues(c):
    """Points this run's solver globals at new constraint lists and a new KB for the clue settings c."""
    global unary_constraints, binary_constraints, KB, kb_models
    unary_constraints, binary_constraints = build_constraints(c)
    KB = build_kb(c)
    kb_models = load_kb_models(SOURCE_VERSION, tuple(sorted(c.items())))   # used by entails() and ask()


def run_case(c):
    use_clues(c)
    return solve_case()


def current_clues():
    """Clue settings shown in the dropdowns and checkboxes right now."""
    return {k: st.session_state.get(f"clue_{k}", v) for k, v in DEFAULT_CLUES.items()}


def active_clues():
    """Clue settings of the last Solve (the original case before the first Solve)."""
    return st.session_state.get("solved_clues", DEFAULT_CLUES)


def reset_clues():
    """Restores the dropdowns and ticks every clue."""
    for k, v in DEFAULT_CLUES.items():
        st.session_state[f"clue_{k}"] = v


def tick_all(value):
    for n in CLUE_NUMBERS:
        st.session_state[f"clue_on_{n}"] = value


use_clues(active_clues())   # every tab below uses the clue settings of the last Solve


def domain_combinations(doms):
    size = 1
    for var in variables:
        size = size * len(doms[var])
    return size


def logic_filter(hyps):
    """Same rule as the notebook: keep h if the KB does not entail NOT h."""
    kept = []
    for h in hyps:
        hs = h_symbols(h)
        if not entails(lambda m: not all(m[s] for s in hs)):
            kept.append(h)
    return kept


def cpts_valid(hyps):
    """Runs the notebook's validate_bn() on these hypotheses (global is per script run)."""
    global logic_hypotheses
    logic_hypotheses = hyps
    try:
        validate_bn()
        return True
    except AssertionError:
        return False


def solve_case():
    """CSP -> logical inference. Returns only plain data for st.session_state."""
    log = io.StringIO()
    with redirect_stdout(log):
        print("Initial domains:")
        print_domains(domains)
        print("\nNode consistency:")
        d_unary = apply_unary_constraints(domains, verbose=True)
        print_domains(d_unary)
        print("\nAC-3:")
        d_ac3 = ac3(d_unary, verbose=True)
        if d_ac3 is None:
            print("AC-3 emptied a domain, so the clues contradict each other.")
            csp, nodes = [], 0
        else:
            print_domains(d_ac3)
            csp, nodes = backtracking_search(d_ac3)
            print("\nBacktracking nodes explored:", nodes)
    kb_ok = len(kb_models) > 0
    logic = logic_filter(csp) if kb_ok else []
    return {
        "initial": {v: list(domains[v]) for v in variables},
        "unary": d_unary,
        "ac3": d_ac3,
        "csp": csp,
        "nodes": nodes,
        "kb_ok": kb_ok,
        "kb_count": len(kb_models),
        "logic": logic,
        "cpt_ok": cpts_valid(logic) if logic else True,
        "log": log.getvalue(),
    }


def default_choice(name):
    if name not in DEFAULT_OBSERVED:
        return "Unknown"
    return "True" if DEFAULT_OBSERVED[name] else "False"


def current_observed():
    """Unknown means the item is left out of the dictionary (it is not False)."""
    obs = {}
    for name in EVIDENCE_NODES:
        choice = st.session_state.get(f"ev_{name}", default_choice(name))
        if choice == "True":
            obs[name] = True
        elif choice == "False":
            obs[name] = False
    return obs


def safe_ranking(hyps, obs):
    if len(hyps) == 0:
        return [], None
    try:
        return build_ranking(hyps, obs), None
    except (ValueError, ZeroDivisionError) as e:
        return [], str(e)


def safe_next_clue(hyps, obs):
    if len(hyps) == 0:
        return None, None
    try:
        return next_clue(hyps, obs), None
    except (ValueError, ZeroDivisionError) as e:
        return None, str(e)


def safe_hmm():
    try:
        path, path_prob = viterbi(CLUE_STREAM)
        return {
            "time_probs": murder_time_distribution(CLUE_STREAM),
            "path": path,
            "path_prob": path_prob,
            "smooth": forward_backward(CLUE_STREAM),
            "filter": forward(CLUE_STREAM),
        }, None
    except (ValueError, ZeroDivisionError, KeyError) as e:
        return None, str(e)


def label(h, rank=None):
    text = f"{h['Culprit']} · {h['Weapon']} · {h['Location']} · {h['Time']}"
    return f"#{rank} {text}" if rank is not None else text


def proven_by_logic(h):
    """True only if the KB entails every part of h."""
    return len(kb_models) > 0 and all(ask(s) == "TRUE (proven)" for s in h_symbols(h))


def reset_all():
    st.session_state.pop("results", None)
    st.session_state.pop("check_result", None)
    st.session_state.pop("solved_clues", None)
    for name in EVIDENCE_NODES:
        st.session_state[f"ev_{name}"] = default_choice(name)
    reset_clues()


# =====================================================================
# PAGE
# =====================================================================
st.set_page_config(page_title="AI Detective", page_icon="🔍", layout="centered")

st.markdown(
    """
    <style>
    h1 { letter-spacing: 0.5px; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

for _name in EVIDENCE_NODES:
    if f"ev_{_name}" not in st.session_state:
        st.session_state[f"ev_{_name}"] = default_choice(_name)
for _k, _v in DEFAULT_CLUES.items():
    if f"clue_{_k}" not in st.session_state:
        st.session_state[f"clue_{_k}"] = _v

st.title("🔍 AI Detective")
st.markdown(
    "**Who killed Vikram Shah at Riverside Manor?** The agent removes every possibility that breaks a clue "
    "(CSP + logic), then ranks the rest with a Bayesian network."
)
st.caption(
    "All deductions follow the puzzle's encoded rules and modelling assumptions. "
    "They are not real-world facts."
)

# ---------------- 1. Case overview ----------------
with st.expander("📜 The case"):
    st.markdown(
        "Vikram Shah was killed on the night of his 60th birthday party at Riverside Manor. "
        "His body was found the next morning near the staircase, but it had been moved there after the crime.\n\n"
        f"- **Suspects:** " + ", ".join(f"{s} ({ROLES[s]})" for s in SUSPECTS) + "\n"
        f"- **Weapons:** {', '.join(WEAPONS)}\n"
        f"- **Locations:** {', '.join(LOCATIONS)}\n"
        f"- **Times:** {', '.join(TIMES)}"
    )

with st.expander("❓ Uncertain evidence (used only by the Bayesian network)"):
    for name in EVIDENCE_NODES:
        st.markdown(f"- **{name}:** {EVIDENCE[name]['text']}")
    st.caption("These items are not confirmed facts. They change probabilities, never the hard constraints.")

# ---------------- Confirmed clues (clues 1, 3, 6 and 7 are editable) ----------------
def formula_text(n, formula):
    """Formula of clue n, marked 'off' (in grey) when the clue is unticked."""
    if is_on(current_clues(), n):
        return f"`{formula}`"
    return f":gray[`{formula}` (off)]"


def number_label(text):
    """Checkbox label: escape the first '.' so '5. ...' is not read as a markdown list (keeps the number)."""
    return text.replace(".", "\\.", 1)


def fixed_clue_row(n):
    a, b = st.columns([3, 1.5], vertical_alignment="center")
    a.checkbox(number_label(f"{n}. {CASE_CLUES[n - 1]}"), key=f"clue_on_{n}")
    b.markdown(formula_text(n, FORMULAS[n]))


def editable_clue_row(n, parts):
    """parts: the text before the first dropdown (checkbox label), then text pieces and
    (label, options, key[, format_func]) dropdowns, shown left to right, then the live formula.
    format_func only changes what the dropdown shows; the stored value stays the plain name.
    Dropdowns of an unticked clue are greyed out. A 'Reads as' line shows the whole sentence."""
    cols = st.columns([1.3 if isinstance(p, str) else 1.2 for p in parts] + [1.5], vertical_alignment="center")
    cols[0].checkbox(number_label(parts[0]), key=f"clue_on_{n}")
    off = not is_on(current_clues(), n)
    for col, p in zip(cols[1:], parts[1:]):
        if isinstance(p, str):
            col.markdown(f":gray[{p}]" if off else p)
        else:
            fmt = p[3] if len(p) > 3 else str
            col.selectbox(p[0], p[1], key=p[2], format_func=fmt, label_visibility="collapsed", disabled=off)
    cols[-1].markdown(formula_text(n, clue_formula(n, current_clues())))
    st.caption(f"Reads as: {clue_sentence(n, current_clues())}" + (" (off)" if off else ""))


with st.expander("✅ Confirmed clues", expanded=True):
    st.caption("Each clue is shown with its propositional form (~ NOT, ∧ AND, ∨ OR, ⇒ IMPLIES). "
               "Clues 1, 3, 6, 7 and 11 can be changed with the dropdowns; the other clues are fixed. "
               "Untick a clue to leave it out of both the CSP and the KB.")
    for n in CLUE_NUMBERS:
        if n == 1:
            editable_clue_row(1, ["1. The autopsy", ("Clue 1: found or not", FOUND_CHOICES, "clue_c1_found"),
                                  ("Clue 1: finding", WEAPONS, "clue_c1_weapon", WEAPON_SIGNS.get)])
        elif n == 3:
            editable_clue_row(3, ["3. Vikram was still alive at", ("Clue 3: time", ALIVE_TIMES, "clue_c3_time"),
                                  ", so the murder happened later."])
        elif n == 6:
            editable_clue_row(6, ["6. CCTV shows", ("Clue 6: suspect", SUSPECTS, "clue_c6_suspect"),
                                  "never left the", ("Clue 6: location", LOCATIONS, "clue_c6_location", str.lower)])
        elif n == 7:
            editable_clue_row(7, ["7. The", ("Clue 7: weapon", WEAPONS, "clue_c7_weapon", str.lower),
                                  "never left the", ("Clue 7: location", LOCATIONS, "clue_c7_location", str.lower)])
        elif n == 11:
            editable_clue_row(11, ["11.", ("Clue 11: suspect", SUSPECTS, "clue_c11_suspect"),
                                   "has a broken wrist and cannot lift the trophy."])
        else:
            fixed_clue_row(n)

    b1, b2, b3 = st.columns(3)
    b1.button("Tick all", on_click=tick_all, args=(True,), width="stretch")
    b2.button("Untick all", on_click=tick_all, args=(False,), width="stretch")
    b3.button("Reset clues", on_click=reset_clues, width="stretch")
    st.caption(f"{len(ENCODED_NUMBERS)} of {len(CASE_CLUES)} clues are encoded as constraints in the solver. "
               f"{len(CLUE_NUMBERS) - len(off_clues(current_clues()))} are ticked. "
               "Reset clues restores the dropdowns of clues 1, 3, 6, 7 and 11 and ticks every clue.")

# ---------------- 2. Solve workflow ----------------
c1, c2 = st.columns(2)
if c1.button("Solve Case", type="primary", width="stretch"):
    clues_now = current_clues()
    with st.spinner("Reasoning through the clues..."):
        original_count = None
        if is_modified(clues_now):
            original_count = len(run_case(DEFAULT_CLUES)["logic"])   # original case, for the comparison
        results = run_case(clues_now)        # run last, so the globals stay on the solved clues
        results["original_count"] = original_count
    st.session_state["results"] = results
    st.session_state["solved_clues"] = dict(clues_now)
c2.button("Reset", on_click=reset_all, width="stretch")

res = st.session_state.get("results")
solved = active_clues()
stale = current_clues() != solved     # dropdowns or ticks differ from the clues the results were computed with

if stale:
    st.warning(STALE_TEXT)
elif res is None:
    st.info("Press **Solve Case** to run CSP → logical inference → Bayesian ranking.")
else:
    if is_modified(solved):
        changes = "\n".join(
            f"- Clue {n}: {clue_sentence(n, DEFAULT_CLUES)} `{clue_formula(n, DEFAULT_CLUES)}` → "
            f"{clue_sentence(n, solved)} `{clue_formula(n, solved)}`" for n in changed_clues(solved))
        if off_clues(solved):
            changes += "\n\nOff: " + ", ".join(f"Clue {n}" for n in off_clues(solved)) + "."
        st.info(f"**Modified case**\n\n{changes.strip()}\n\n"
                f"Original: {res['original_count']} hypotheses → Modified: {len(res['logic'])}\n\n{MODIFIED_NOTE}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Initial hypotheses", TOTAL_HYPOTHESES)
    m2.metric("Domain combinations after AC-3", domain_combinations(res["ac3"]) if res["ac3"] else 0)
    m3.metric("CSP solutions", len(res["csp"]))
    m4.metric("After logical inference", len(res["logic"]))
    st.caption(
        "Domain combinations are only an upper bound (AC-3 checks two variables at a time). "
        "CSP solutions are complete hypotheses that break no encoded clue."
    )
    reasons = []
    if res["ac3"] is None:
        reasons.append("AC-3 emptied a domain")
    elif len(res["csp"]) == 0:
        reasons.append("backtracking found no complete hypothesis that satisfies every clue")
    if not res["kb_ok"]:
        reasons.append("the KB has no satisfying model")
    if reasons:
        st.error("**These clues contradict each other** (" + "; ".join(reasons) + "). There is no valid hypothesis.")

tab_rank, tab_logic, tab_check, tab_hmm, tab_diag = st.tabs(
    ["Verdict", "Logic queries", "Check a hypothesis", "Timeline (HMM)", "Diagnostics"]
)

# ---------------- 3 + 4. Results, verdict, evidence controls ----------------
with tab_rank:
    st.subheader("Uncertain evidence")
    st.caption("Set each item to True, False or Unknown. **Unknown** means it is not observed and is left out; "
               "it does not mean False. Defaults are the notebook's observed evidence.")
    cols = st.columns(2)
    for i, name in enumerate(EVIDENCE_NODES):
        with cols[i % 2]:
            st.radio(f"**{name}**", EVIDENCE_CHOICES, key=f"ev_{name}", horizontal=True,
                     help=EVIDENCE[name]["text"])

    obs = current_observed()

    if stale:
        st.warning(STALE_TEXT)
    elif res is None:
        st.info("Solve the case first to see the ranking.")
    elif len(res["logic"]) == 0:
        st.warning("No hypothesis is left after CSP and logic, so there is nothing to rank.")
    else:
        if is_modified(solved):
            st.warning(MODIFIED_NOTE)
        rows, err = safe_ranking(res["logic"], obs)
        if err:
            st.error(f"The ranking could not be computed with this evidence: {err}")
        if rows:
            if not res["cpt_ok"]:
                st.warning("A CPT value is outside [0, 1]; treat these probabilities with care.")
            st.subheader("Ranking")
            df = pd.DataFrame([{
                "Rank": i,
                "Culprit": r["hypothesis"]["Culprit"],
                "Weapon": r["hypothesis"]["Weapon"],
                "Location": r["hypothesis"]["Location"],
                "Time": r["hypothesis"]["Time"],
                "Posterior": r["probability"],
            } for i, r in enumerate(rows, start=1)])
            st.dataframe(
                df, hide_index=True, width="stretch",
                column_config={"Posterior": st.column_config.ProgressColumn(
                    "Posterior", format="%.3f", min_value=0.0, max_value=1.0)},
            )
            st.caption(f"Sum of posteriors = {sum(r['probability'] for r in rows):.3f}")

            chart_df = pd.DataFrame({
                "Hypothesis": [label(r["hypothesis"], i) for i, r in enumerate(rows, start=1)],
                "Posterior": [r["probability"] for r in rows],
            })
            chart = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    x=alt.X("Posterior:Q", scale=alt.Scale(domain=[0, 1]), title="Posterior probability"),
                    y=alt.Y("Hypothesis:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
                    tooltip=["Hypothesis", alt.Tooltip("Posterior:Q", format=".3f")],
                )
                .properties(height=28 * len(chart_df) + 40)
            )
            st.altair_chart(chart, width="stretch")

            # verdict
            st.subheader("Verdict")
            v_label, v_reason = verdict(rows)
            top = rows[0]
            tied = [r for r in rows if abs(r["probability"] - top["probability"]) < 1e-12]
            if len(tied) > 1:
                st.warning(f"**INCONCLUSIVE (tie).** {len(tied)} hypotheses share the top probability "
                           f"P = {top['probability']:.3f}, so no single answer can be picked.")
                for r in tied:
                    st.markdown(f"- {describe(r['hypothesis'])}")
            else:
                show = {"STRONG": st.success, "LIKELY": st.info, "CONCLUSIVE": st.success}.get(v_label, st.warning)
                show(f"**{v_label}.** {v_reason}")
                if proven_by_logic(top["hypothesis"]):
                    st.markdown(f"**Logically determined:** {describe(top['hypothesis'])}")
                else:
                    st.markdown(f"**Most probable (not proven):** {describe(top['hypothesis'])} "
                                f"(P = {top['probability']:.2f})")
            st.caption(
                "Probabilities depend on the assumed CPT values in the notebook and are conditional on the "
                f"{len(rows)} hypotheses that survived CSP and logic. Only logic can prove something; "
                "a high probability is not a proof."
            )

            with st.expander("Supporting and conflicting evidence"):
                st.caption("Ratio = how likely the observed item is under this hypothesis compared with the "
                           f"average of the others. Above {SUPPORT_LIMIT} supports, below {CONFLICT_LIMIT} conflicts.")
                if len(obs) == 0:
                    st.markdown("No evidence is observed, so nothing supports or conflicts yet.")
                for i, r in enumerate(rows, start=1):
                    sup = ", ".join(f"{n} (×{x:.1f})" for n, x in r["supports"]) or "none"
                    con = ", ".join(f"{n} (×{x:.1f})" for n, x in r["conflicts"]) or "none"
                    st.markdown(f"**{label(r['hypothesis'], i)}** · P = {r['probability']:.3f}  \n"
                                f"Supported by: {sup}  \nConflicts with: {con}")

        st.subheader("Which evidence to check next")
        options, err2 = safe_next_clue(res["logic"], obs)
        if err2:
            st.error(f"Information gain could not be computed: {err2}")
        elif options is None:
            st.markdown("Every evidence item is set to True or False, so no unobserved evidence items remain.")
        else:
            best_name, best_gain = options[0]
            st.markdown(f"Find out **{best_name}** next: expected information gain {best_gain:.3f} bits.")
            st.dataframe(pd.DataFrame(options, columns=["Evidence item", "Information gain (bits)"]),
                         hide_index=True, width="stretch")

# ---------------- 5. Logical queries ----------------
with tab_logic:
    st.subheader("Ask the knowledge base")
    q1, q2 = st.columns(2)
    q_var = q1.selectbox("Variable", variables, key="q_var")
    q_val = q2.selectbox("Value", domains[q_var], key=f"q_val_{q_var}")
    if stale:
        st.warning(STALE_TEXT)
    elif len(kb_models) == 0:
        st.error("**These clues contradict each other:** no model satisfies the KB, so no truth claims are shown.")
    else:
        s = sym(q_var, q_val)
        st.markdown(f"`KB ⊨ {s} ?` **{'Yes' if entails(lambda m: m[s]) else 'No'}**  \n"
                    f"`KB ⊨ ~{s} ?` **{'Yes' if entails(lambda m: not m[s]) else 'No'}**")
        answer = ASK_LABELS[ask(s)]
        if q_var == "Culprit":
            text = {"Proven true": f"The KB proves {q_val} is the culprit.",
                    "Proven false": f"The KB proves {q_val} is not the culprit.",
                    "Unknown": f"Neither proven guilty nor proven innocent. Guilt not proven does not mean innocence proven."}
        else:
            text = {"Proven true": f"The KB proves {q_var} = {q_val}.",
                    "Proven false": f"The KB proves {q_var} ≠ {q_val}.",
                    "Unknown": f"{q_var} = {q_val} is neither proven nor disproven."}
        show = {"Proven true": st.success, "Proven false": st.error, "Unknown": st.info}[answer]
        show(f"**{answer}.** {text[answer]}")
        st.caption(f"Model checking over {2 ** len(SYMBOLS)} truth assignments; {len(kb_models)} satisfy the KB. "
                   "**Unknown** means some KB models make it true and others make it false.")
        with st.expander("All symbols"):
            st.dataframe(pd.DataFrame([{"Symbol": s, "Result": ASK_LABELS[ask(s)]} for s in SYMBOLS]),
                         hide_index=True, width="stretch")

# ---------------- 6. Hypothesis checker ----------------
with tab_check:
    st.subheader("Check your own theory")
    with st.form("checker"):
        f1, f2 = st.columns(2)
        hc = f1.selectbox("Culprit", SUSPECTS)
        hw = f2.selectbox("Weapon", WEAPONS)
        f3, f4 = st.columns(2)
        hl = f3.selectbox("Location", LOCATIONS)
        ht = f4.selectbox("Time", TIMES)
        if st.form_submit_button("Check hypothesis", width="stretch"):
            st.session_state["check_result"] = {"h": {"Culprit": hc, "Weapon": hw, "Location": hl, "Time": ht}}
    result = st.session_state.get("check_result")
    if result and stale:
        st.warning(STALE_TEXT)
    elif result:
        broken = broken_clues(result["h"])     # checked against the clues of the last Solve
        st.markdown(f"**{describe(result['h'])}**  \n`{hypothesis_formula(result['h'])}`")
        if broken:
            st.error("Not possible. It breaks these encoded constraints:")
            for clue in broken:
                st.markdown(f"- {clue} `{clue_formula(clue_number(clue), solved)}`")
        else:
            st.success("Consistent with the clues.")
            st.caption("No encoded constraint rules it out. That does not mean it is proven.")

# ---------------- 7. Temporal reasoning ----------------
with tab_hmm:
    st.subheader("When did it happen?")
    if res is not None and not stale and is_modified(solved):
        st.warning(MODIFIED_NOTE + " The HMM is not recalculated for the modified clues.")
    hmm, hmm_err = safe_hmm()
    if hmm_err:
        st.error(f"The HMM could not be computed: {hmm_err}")
    else:
        tp = hmm["time_probs"]
        if tp:
            cols = st.columns(len(tp))
            for col, (t, p) in zip(cols, tp.items()):
                col.metric(f"P(murder at {t})", f"{p:.2f}")
        st.dataframe(pd.DataFrame([{
            "Step": STEPS[i],
            "Observation": CLUE_STREAM[i],
            "Viterbi state": hmm["path"][i],
            "P(Alive)": round(hmm["smooth"][i]["Alive"], 3),
            "P(Killed)": round(hmm["smooth"][i]["Killed"], 3),
            "P(Moved)": round(hmm["smooth"][i]["Moved"], 3),
        } for i in range(len(STEPS))]), hide_index=True, width="stretch")
        st.caption("State probabilities are smoothed (forward-backward, using all observations). "
                   "Viterbi is the single most likely sequence; P(murder time) sums over all sequences, "
                   "so the two can point to different times.")

        best_p = max(tp.values()) if tp else None
        hmm_times = [t for t, p in tp.items() if best_p is not None and abs(p - best_p) < 1e-12]
        if stale:
            st.caption(STALE_TEXT)
        elif res and res["logic"]:
            rows_now, _ = safe_ranking(res["logic"], current_observed())
            if rows_now and hmm_times:
                top_time = rows_now[0]["hypothesis"]["Time"]
                if len(hmm_times) > 1:
                    st.info(f"The HMM is tied between {', '.join(hmm_times)}.")
                elif hmm_times[0] == top_time:
                    st.success(f"HMM most likely time ({hmm_times[0]}) agrees with the top hypothesis time ({top_time}).")
                else:
                    st.warning(f"HMM most likely time ({hmm_times[0]}) disagrees with the top hypothesis time "
                               f"({top_time}). Treat the verdict with care.")
        else:
            st.caption("Solve the case to compare the HMM with the top hypothesis.")
        st.caption("The HMM uses its own fixed observation stream and is kept separate from the Bayesian ranking "
                   "(the 10 PM sighting is already used there), so evidence is not counted twice. "
                   "Changing the evidence controls does not change the HMM.")

# ---------------- 8. Diagnostics ----------------
with tab_diag:
    if stale:
        st.warning(STALE_TEXT)
    elif res is None:
        st.info("Solve the case to see the diagnostics.")
    else:
        st.markdown("**Encoded clues used in the last Solve** (ticked clues only)")
        if off_clues(solved):
            st.caption("Off: " + ", ".join(f"Clue {n}" for n in off_clues(solved)))
        active_labels = [c for _, _, c in unary_constraints] + [c for _, _, _, c in binary_constraints]
        st.dataframe(pd.DataFrame([{
            "Clue": clue_number(c),
            "Constraint": c.split(": ", 1)[1],
            "Formula": clue_formula(clue_number(c), solved),
        } for c in sorted(active_labels, key=clue_number)]), hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame([{
            "Variable": v,
            "Initial": ", ".join(res["initial"][v]),
            "After node consistency": ", ".join(res["unary"][v]),
            "After AC-3": ", ".join(res["ac3"][v]) if res["ac3"] else "(empty)",
        } for v in variables]), hide_index=True, width="stretch")
        st.markdown(
            f"- Combinations: {domain_combinations(res['initial'])} → {domain_combinations(res['unary'])} "
            f"→ {domain_combinations(res['ac3']) if res['ac3'] else 0}\n"
            f"- Backtracking nodes explored: {res['nodes']}\n"
            f"- KB models: {res['kb_count']}"
        )
        ruled_out = [s for s in SUSPECTS if all(h["Culprit"] != s for h in res["logic"])]
        st.markdown("- Suspects ruled out: " + (", ".join(ruled_out) if ruled_out else "none"))
        for v in variables[1:]:
            gone = [x for x in domains[v] if all(h[v] != x for h in res["logic"])]
            st.markdown(f"- {v} values ruled out: " + (", ".join(gone) if gone else "none"))
        with st.expander("Solver log"):
            st.code(res["log"] or "(empty)", language=None)
