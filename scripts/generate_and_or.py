"""Generate compound and/or statements from the cities dataset.

Pairs true/false cities statements with "and"/"or" into compound statements,
labeled by the logical truth table of the connective, so probes can be
tested on whether they track compositional truth rather than just
per-statement truth.
"""

import random
import re

import pandas as pd

from src.data import load_statements

random.seed(0)

df = load_statements("cities")
true_stmts  = df[df.label == 1]["statement"].tolist()
false_stmts = df[df.label == 0]["statement"].tolist()

def city_of(s):
    """Extract the city name so we can avoid repeating a city within a compound."""
    m = re.search(r"city of (.+?) is in", s)
    return m.group(1).strip() if m else s[:20]

def make_compound(a, b, connective):
    """'The city of X is in Y.' + 'The city of Z is in W.' -> 'The city of X is in Y and the city of Z is in W.'"""
    a = a.strip().rstrip(".")
    b = b.strip().rstrip(".")
    b = b[0].lower() + b[1:]          # 'The city' -> 'the city'
    return f"{a} {connective} {b}."

TRUTH_TABLE = {
    ("and", "TT"): 1, ("and", "TF"): 0, ("and", "FT"): 0, ("and", "FF"): 0,
    ("or",  "TT"): 1, ("or",  "TF"): 1, ("or",  "FT"): 1, ("or",  "FF"): 0,
}

def sample_conjunct(kind):
    return random.choice(true_stmts if kind == "T" else false_stmts)

rows = []
N_PER_CELL = 200
for connective in ["and", "or"]:
    for pattern in ["TT", "TF", "FT", "FF"]:
        made = 0
        while made < N_PER_CELL:
            a = sample_conjunct(pattern[0])
            b = sample_conjunct(pattern[1])
            if city_of(a) == city_of(b):
                continue                       # reject same-city compounds
            rows.append({
                "statement":  make_compound(a, b, connective),
                "connective": connective,
                "pattern":    pattern,
                "label":      TRUTH_TABLE[(connective, pattern)],
                "conj_a":     a,
                "conj_b":     b,
            })
            made += 1

comp = pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)
comp.to_csv("data/compound_cities.csv", index=False)