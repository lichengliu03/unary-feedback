from typing import Tuple


# Compact codes for CardinalBinsAllo labels
_DIR_LABEL_TO_CODE = {
    "north": "N",
    "north east": "NE",
    "east": "E",
    "south east": "SE",
    "south": "S",
    "south west": "SW",
    "west": "W",
    "north west": "NW",
}
_DIR_CODE_TO_LABEL = {v: k for k, v in _DIR_LABEL_TO_CODE.items()}

# Compact codes for StandardDistanceBins labels
_DIST_LABEL_TO_CODE = {
    "same distance": "same",
    "near": "near",
    "mid distance": "mid",
    "slightly far": "sfar",
    "far": "far",
    "very far": "vfar",
    "extremely far": "xfar",
}
_DIST_CODE_TO_LABEL = {v: k for k, v in _DIST_LABEL_TO_CODE.items()}


def _normalize_key(s: str) -> str:
    return str(s).strip().lower()


def direction_label_to_code(label: str) -> str:
    return _DIR_LABEL_TO_CODE.get(_normalize_key(label), _normalize_key(label))


def direction_code_to_label(code: str) -> str:
    return _DIR_CODE_TO_LABEL.get(_normalize_key(code), _normalize_key(code))


def distance_label_to_code(label: str) -> str:
    return _DIST_LABEL_TO_CODE.get(_normalize_key(label), _normalize_key(label))


def distance_code_to_label(code: str) -> str:
    return _DIST_CODE_TO_LABEL.get(_normalize_key(code), _normalize_key(code))


def to_code(value: str) -> str:
    """Unified: map direction/distance label-or-code to code.
    Examples: 'north'->'n'; 'n'->'n'; 'mid distance'->'mid'; 'mid'->'mid'.
    """
    v = _normalize_key(value)
    v_upper = v.upper()
    # direction codes/labels → uppercase codes
    if v in _DIR_LABEL_TO_CODE:
        return _DIR_LABEL_TO_CODE[v]
    # accept lower/upper direction codes
    if v_upper in _DIR_CODE_TO_LABEL:
        return v_upper
    if v in _DIST_CODE_TO_LABEL:
        return v
    if v in _DIST_LABEL_TO_CODE:
        return _DIST_LABEL_TO_CODE[v]
    return v


def to_label(value: str) -> str:
    """Unified: map direction/distance code-or-label to label.
    Examples: 'n'->'north'; 'north'->'north'; 'mid'->'mid distance'.
    """
    v = _normalize_key(value)
    if v.upper() in _DIR_CODE_TO_LABEL:
        return _DIR_CODE_TO_LABEL[v.upper()]
    if v in _DIST_CODE_TO_LABEL:
        return _DIST_CODE_TO_LABEL[v]
    if v in _DIR_LABEL_TO_CODE:
        return v
    if v in _DIST_LABEL_TO_CODE:
        return v
    return v


def encode_relation_codes(dir_value: str, dist_value: str) -> str:
    """Encode direction/distance values (code or label) to short tuple string '(dir, dist)'."""
    d = to_code(dir_value)
    r = to_code(dist_value)
    return f"({d}, {r})"


def decode_relation_codes(text: str) -> Tuple[str, str]:
    """Decode '(nw, mid)' or 'nw, mid' to (dir_code, dist_code)."""
    s = str(text).strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        return "", ""
    return to_code(parts[0]), to_code(parts[1])


def make_pair_key(a: str, b: str) -> str:
    """Canonical unordered pair key 'A|B' with lexicographic order (case-insensitive)."""
    a_s, b_s = str(a), str(b)
    return (f"{a_s}|{b_s}" if a_s.lower() <= b_s.lower() else f"{b_s}|{a_s}")



# ---- Ordered pair helpers (A relative to B is encoded as 'A|B') ----
def make_ordered_pair_key(a: str, b: str) -> str:
    """Ordered pair key 'A|B' (A relative to B)."""
    return f"{str(a)}|{str(b)}"


def parse_pair_key(key: str) -> Tuple[str, str]:
    """Parse 'A|B' into (A, B) preserving order; returns ("", "") if invalid."""
    s = str(key).split('|')
    return (s[0], s[1]) if len(s) == 2 else ("", "")


def invert_pair_key(key: str) -> str:
    a, b = parse_pair_key(key)
    return f"{b}|{a}" if a or b else key
# ---- Convenience helpers ----
def make_ordered_and_inverse_pair_keys(a: str, b: str) -> Tuple[str, str]:
    """Return (ordered_key 'A|B', inverse_key 'B|A')."""
    ordered = make_ordered_pair_key(a, b)
    return ordered, invert_pair_key(ordered)


# ---- Direction/distance utilities ----
def invert_dir_code(code: str) -> str:
    c = str(code).upper()
    inv = {
        'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E',
        'NE': 'SW', 'SW': 'NE', 'NW': 'SE', 'SE': 'NW'
    }
    return inv.get(c, c)


def invert_relation_codes_str(relation_str: str) -> str:
    """Invert a relation string like '(W, near)' -> '(E, near)'."""
    d, r = decode_relation_codes(relation_str)
    return f"({invert_dir_code(d)}, {r})"


# ---- Discrete relation construction from codes ----
def discrete_relation_from_codes(dir_code: str, dist_code: str):
    """Construct PairwiseRelationshipDiscrete from (dir_code, dist_code).

    Uses bin centers; no world positions are computed.
    """
    # Local import to avoid circulars
    from ..core.relationship import (
        PairwiseRelationshipDiscrete, DegreeRel, DegreeRelBinned,
        DistanceRelBinned, CardinalBinsAllo, StandardDistanceBins
    )

    bin_system = CardinalBinsAllo()
    dist_system = StandardDistanceBins()

    # Direction: map code->label->bin index -> bin center degree
    dir_label = to_label(dir_code)
    try:
        dir_idx = bin_system.LABELS.index(dir_label)
    except ValueError:
        # Fallback: default to 'north'
        dir_idx = 0
    lo_deg, hi_deg = bin_system.BINS[dir_idx]
    deg = (float(lo_deg) + float(hi_deg)) / 2.0
    drel = DegreeRel(degree=deg)
    d_binned = DegreeRelBinned.from_relation(drel, bin_system)

    # Distance: code->label->bin index -> bin center distance
    dist_label = to_label(dist_code)
    try:
        dist_idx = dist_system.LABELS.index(dist_label)
    except ValueError:
        # default to 'near'
        dist_idx = 1
    lo_r, hi_r = dist_system.BINS[dist_idx]
    dist_val = (float(lo_r) + float(hi_r)) / 2.0
    s_binned = DistanceRelBinned.from_value(dist_val, dist_system)

    return PairwiseRelationshipDiscrete(direction=d_binned, dist=s_binned)


