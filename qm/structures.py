"""Structure handling: SDF/XYZ parsing, the chorismate->prephenate atom
bijection, bond-diff verification, connectivity, and key distances.

Parse mode (no ORCA execution): everything here reads the locked world
(data/ is read-only) and fails hard on any inconsistency.

The bijection is the recorded v1 result (step1), embedded here as the
1-based table; it is verified two ways against
data/derived/atom_mappings.json (see verify_mapping): the artifact's own
directions + verification.ok flag, and an independent bond-diff of the
two SDF connectivity tables mapped through the bijection.
"""

import math
import re

# Chorismate -> prephenate atom bijection, 1-based (recorded in
# data/derived/atom_mappings.json, produced by v1 step1).
CHORIS_TO_PREPH = {
    1: 4, 2: 1, 3: 2, 4: 3, 5: 5, 6: 6, 7: 12, 8: 11, 9: 9, 10: 13,
    11: 7, 12: 10, 13: 14, 14: 15, 15: 16, 16: 8, 17: 22, 18: 21,
    19: 19, 20: 23, 21: 20, 22: 24, 23: 18, 24: 17, 25: 25, 26: 26,
}

N_ATOMS = 26  # both molecules: C10H10O6

# Covalent radii (Angstrom, Cordero et al.) for the elements present —
# the v1 radii table, reproduced exactly.
COVALENT_RADII = {"H": 0.31, "C": 0.76, "O": 0.66}
# v1 connectivity rule, reproduced exactly: a pair is bonded if
# d < r_i + r_j + tol (strict), with tol = 0.45 by default; TS-like
# geometries pass tol = 1.0 ("TS-aware") so the partial [3,3] bonds
# (~2.1-2.5 A) count as connections.
BOND_TOLERANCE_A = 0.45
TS_BOND_TOLERANCE_A = 1.0

# Classification thresholds for IRC / endpoint geometries (chorismate
# atom order, 1-based pairs). The 2.0 A cut splits intact/broken and
# formed/open far from every observed value; 3.2 A is the reactive-
# chair criterion recorded in the locked irc_summary.json.
INTACT_FORMED_A = 2.0
REACTIVE_CHAIR_MAX_A = 3.2

# The six key distances, chorismate 1-based indices, as recorded in the
# locked summaries' key_distances blocks.
KEY_PAIRS = (
    ("O1-C7 (breaking)", 1, 7),
    ("C11-C16 (forming)", 11, 16),
    ("O1-C14 (->C=O)", 1, 14),
    ("C14-C16", 14, 16),
    ("C7-C9", 7, 9),
    ("C9-C11", 9, 11),
)


def parse_sdf(path):
    """Parse an SDF/V2000 file -> (elements, coords_A, bonds).

    bonds: list of (atom_i, atom_j, order), 1-based as in the file.
    Fails hard on malformed counts or block sizes.
    """
    lines = open(path).read().splitlines()
    if len(lines) < 4:
        raise ValueError(f"{path}: truncated SDF (no counts line)")
    try:
        n_atoms, n_bonds = int(lines[3][0:3]), int(lines[3][3:6])
    except ValueError:
        raise ValueError(f"{path}: cannot parse counts line {lines[3]!r}")
    if n_atoms != N_ATOMS:
        raise ValueError(f"{path}: expected {N_ATOMS} atoms, got {n_atoms}")
    if len(lines) < 4 + n_atoms + n_bonds:
        raise ValueError(f"{path}: file too short for {n_atoms}+{n_bonds} blocks")

    elements, coords = [], []
    for line in lines[4:4 + n_atoms]:
        p = line.split()
        if len(p) < 4:
            raise ValueError(f"{path}: bad atom line {line!r}")
        elements.append(p[3])
        coords.append([float(p[0]), float(p[1]), float(p[2])])

    bonds = []
    for line in lines[4 + n_atoms:4 + n_atoms + n_bonds]:
        p = line.split()
        if len(p) < 3:
            raise ValueError(f"{path}: bad bond line {line!r}")
        bonds.append((int(p[0]), int(p[1]), int(p[2])))
    return elements, coords, bonds


def read_xyz_frame(path):
    """Read a single-frame XYZ file -> (n, comment, elements, coords_A).

    ORCA-written trajectory frames carry 'E <energy Eh>' in the comment;
    the energy is extracted by energy_from_comment().
    """
    lines = open(path).read().splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path}: truncated XYZ (no header)")
    try:
        n = int(lines[0].strip())
    except ValueError:
        raise ValueError(f"{path}: bad atom count {lines[0]!r}")
    if len(lines) < 2 + n:
        raise ValueError(f"{path}: file too short for {n} atoms")
    comment = lines[1]
    elements, coords = [], []
    for line in lines[2:2 + n]:
        p = line.split()
        if len(p) < 4:
            raise ValueError(f"{path}: bad atom line {line!r}")
        elements.append(p[0])
        coords.append([float(p[1]), float(p[2]), float(p[3])])
    return n, comment, elements, coords


def energy_from_comment(comment, path=""):
    """Extract the E <eh> energy from an ORCA XYZ comment line."""
    m = re.search(r"E\s+(-?\d+\.\d+)", comment)
    if not m:
        raise ValueError(f"{path}: no energy in comment {comment!r}")
    return float(m.group(1))


def distance(coords, i, j):
    """Euclidean distance (A) between 1-based atoms i and j."""
    return math.dist(coords[i - 1], coords[j - 1])


def key_distances(coords):
    """The six labelled key distances (A) for a chorismate-ordered
    geometry -> dict in the locked summaries' key order."""
    return {label: distance(coords, i, j) for label, i, j in KEY_PAIRS}


def bond_table(bonds):
    """Normalized bond table: {sorted (i, j): order}."""
    return {(min(i, j), max(i, j)): order for i, j, order in bonds}


def bond_diff(mapped_table, target_table):
    """Diff two normalized bond tables -> (broken, formed, order_changes).

    broken  : pairs present in mapped_table but not target_table
    formed  : pairs present in target_table but not mapped_table
    order_changes: [pair, old_order, new_order] for pairs present in
              both with differing order. Lists sorted by pair; pairs
              are in the target numbering (the atom_mappings.json
              verification-block convention: prephenate indices).
    """
    broken = sorted(p for p in mapped_table if p not in target_table)
    formed = sorted(p for p in target_table if p not in mapped_table)
    order_changes = sorted(
        [[list(p), mapped_table[p], target_table[p]]
         for p in mapped_table
         if p in target_table and mapped_table[p] != target_table[p]]
    )
    return broken, formed, order_changes


def map_bond_table(table, bijection):
    """Map a chorismate bond table through the 1-based bijection."""
    return {
        (min(bijection[i], bijection[j]), max(bijection[i], bijection[j])): order
        for (i, j), order in table.items()
    }


def heavy_fragments(elements, coords, tol=BOND_TOLERANCE_A):
    """Connected components by the v1 rule, reproduced exactly.

    The union-find runs over ALL atoms (hydrogens are part of the
    graph, exactly as in v1 — a bridging H can merge groups); a pair is
    bonded if d < r_i + r_j + tol (strict). Returns (n_fragments,
    heavy_fragment_sizes): non-H atom counts per group, sorted
    largest-first, exactly as recorded in the locked summaries.
    TS-like geometries pass tol=TS_BOND_TOLERANCE_A so the partial
    [3,3] bonds count as connections (v1's "TS-aware" call).
    """
    n = len(elements)
    for el in set(elements):
        if el not in COVALENT_RADII:
            raise ValueError(f"no covalent radius for element {el!r}")
    parent = list(range(n))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for a in range(n):
        for b in range(a + 1, n):
            ra, rb = COVALENT_RADII[elements[a]], COVALENT_RADII[elements[b]]
            if math.dist(coords[a], coords[b]) < ra + rb + tol:
                parent[find(a)] = find(b)

    groups = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(k)
    sizes = sorted(
        (sum(1 for k in g if elements[k] != "H") for g in groups.values()),
        reverse=True,
    )
    return len(groups), sizes


def classify_geometry(distances):
    """Endpoint classification from the six key distances.

    Returns (classification, chorismate_type) with the exact strings of
    the locked summaries:
      reactant-like (O1-C7 intact, C11-C16 open)     / reactive chair ...
      product-like  (O1-C7 broken, C11-C16 formed)   / null
    Fails hard on anything else (a geometry that is neither cleanly
    reactant-like nor product-like would contradict the locked record).
    """
    o1_c7 = distances["O1-C7 (breaking)"]
    c11_c16 = distances["C11-C16 (forming)"]
    o1_c7_intact = o1_c7 < INTACT_FORMED_A
    c11_c16_formed = c11_c16 < INTACT_FORMED_A
    if o1_c7_intact and not c11_c16_formed:
        classification = "reactant-like (O1-C7 intact, C11-C16 open)"
        chorismate_type = (
            "reactive chair (C11-C16 <=3.2 A)"
            if c11_c16 <= REACTIVE_CHAIR_MAX_A else None
        )
    elif not o1_c7_intact and c11_c16_formed:
        classification = "product-like (O1-C7 broken, C11-C16 formed)"
        chorismate_type = None
    else:
        raise ValueError(
            f"geometry is neither reactant-like nor product-like "
            f"(O1-C7 {o1_c7:.3f} A, C11-C16 {c11_c16:.3f} A)"
        )
    return classification, chorismate_type


def verify_mapping(chor, preph, mappings_artifact):
    """Two-way verification of the bijection against the locked
    data/derived/atom_mappings.json. Fails hard on any difference.

    (a) the embedded table equals the artifact's choris_to_preph and
        preph_to_choris in both directions, and the artifact's
        verification.ok is true;
    (b) the bond-diff — chorismate's SDF bond table mapped through the
        bijection, diffed against prephenate's — equals exactly the
        artifact's verification block (one bond broken, one formed,
        four order changes, nothing else).
    """
    artifact_c2p = {int(k): v for k, v in mappings_artifact["choris_to_preph"].items()}
    artifact_p2c = {int(k): v for k, v in mappings_artifact["preph_to_choris"].items()}
    p2c = {v: k for k, v in CHORIS_TO_PREPH.items()}

    if CHORIS_TO_PREPH != artifact_c2p:
        raise ValueError(
            "choris_to_preph differs from atom_mappings.json:\n"
            f"  embedded: {CHORIS_TO_PREPH}\n  artifact: {artifact_c2p}"
        )
    if p2c != artifact_p2c:
        raise ValueError(
            "preph_to_choris differs from atom_mappings.json:\n"
            f"  derived:  {p2c}\n  artifact: {artifact_p2c}"
        )
    if sorted(CHORIS_TO_PREPH) != list(range(1, N_ATOMS + 1)):
        raise ValueError("bijection is not a permutation of 1..26")
    if mappings_artifact["verification"].get("ok") is not True:
        raise ValueError("atom_mappings.json verification.ok is not true")

    mapped_table = map_bond_table(bond_table(chor[2]), CHORIS_TO_PREPH)
    target_table = bond_table(preph[2])
    broken, formed, order_changes = bond_diff(mapped_table, target_table)

    v = mappings_artifact["verification"]
    want_broken = [tuple(x) for x in v["broken"]]
    want_formed = [tuple(x) for x in v["formed"]]
    if broken != want_broken:
        raise ValueError(f"bond-diff broken {broken} != artifact {want_broken}")
    if formed != want_formed:
        raise ValueError(f"bond-diff formed {formed} != artifact {want_formed}")
    if order_changes != v["order_changes"]:
        raise ValueError(
            f"bond-diff order_changes {order_changes} != artifact {v['order_changes']}"
        )
    if not (len(broken) == 1 and len(formed) == 1 and len(order_changes) == 4):
        raise ValueError(
            "bond-diff is not exactly 1 broken / 1 formed / 4 order changes"
        )
    return broken, formed, order_changes