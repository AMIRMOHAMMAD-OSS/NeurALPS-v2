#!/usr/bin/env python
"""
connection_rules.py -- shared vocabulary + window rules for NeurALPS connection
cutting, IDENTICAL across natural, Bode-1, Bode-2. Import this from each driver;
never fork these rules per dataset.

Contains ONLY pure functions + tables. No dataset I/O, no GenBank parsing.
"""
from __future__ import annotations

# ----------------------------------------------------------------------------
# 1. DOMAIN VOCABULARY  (settled 2026-09; identical across all three sets)
# ----------------------------------------------------------------------------
# Carrier is one type: antiSMASH emits it as PCP (aSDomain) and PP-binding
# (PFAM). Merge to PCP everywhere.
CARRIER_ALIASES = {"PP-binding": "PCP", "PCP": "PCP"}

# Thioesterase family -> Thioesterase (TE recovered from PFAM where aSDomain
# lacks it; TD / Thioesterase_2 fold in).
TE_ALIASES = {"TD": "Thioesterase", "Thioesterase_2": "Thioesterase",
              "Thioesterase": "Thioesterase"}

# PKS machinery: kept in the ROUTE as neighbours (never deleted) but typed to a
# single out-of-core row, since the first model is NRPS-core.
PKS_TYPES = {
    "PKS_KS", "PKS_AT", "PKS_KR", "PKS_DH", "PKS_DH2", "PKS_DHt", "PKS_ER",
    "PKS_PP", "ACP", "ACP_beta", "SAT", "PT", "cAT", "FkbH", "Trans-AT_docking",
    "PKS_Docking_Nterm", "PKS_Docking_Cterm",
}
OUT_OF_CORE = "OUT_OF_CORE"

# TIGRFAM / region calls that OVERLAP an aSDomain are duplicates and collapse
# into the underlying aSDomain (drop the TIGRFAM node). When they STAND ALONE
# they are real domains and keep their own fine row. Decided per-instance by
# coordinate overlap in the driver, NOT by name. Names that are known TIGRFAM
# adenylation/region overlays:
TIGRFAM_OVERLAY_CANDIDATES = {"TIGR01733", "AMP-binding_C"}
# TIGR01720 / TIGR02353 / Interface were shown to stand ALONE in natural, so
# they are fine rows there; in raw GenBank they must still pass the overlap
# test per instance.

# Fine NRPS catalytic + tailoring rows (everything here keeps its own type).
# Anything not resolved by the maps above and not PKS and >= floor stays as-is;
# below floor -> OTHER. The driver supplies the natural-derived count table.
OTHER = "OTHER"
X_PLACEHOLDER = "X"          # antiSMASH unclassified -> OTHER


def canonical_type(raw: str) -> str:
    """Map a raw antiSMASH domain label to its canonical vocabulary row.
    Overlap-collapse of TIGRFAM overlays is handled in the driver (needs
    coordinates); this handles the context-free renames only."""
    if raw in CARRIER_ALIASES:
        return CARRIER_ALIASES[raw]
    if raw in TE_ALIASES:
        return TE_ALIASES[raw]
    if raw in PKS_TYPES:
        return OUT_OF_CORE
    if raw == X_PLACEHOLDER:
        return OTHER
    return raw  # fine row; driver may still fold to OTHER if below the floor


# ----------------------------------------------------------------------------
# 2. OVERLAP RESOLUTION  (route-level; same rule for every dataset)
# ----------------------------------------------------------------------------
def overlaps(a_start, a_end, b_start, b_end) -> int:
    """Residue overlap of two half-open aa intervals; <=0 means disjoint."""
    return min(a_end, b_end) - max(a_start, b_start)


def resolve_domain_nodes(domains, overlay_names=TIGRFAM_OVERLAY_CANDIDATES,
                         min_overlap_frac=0.5):
    """domains: list of dicts {start,end,raw,is_tigrfam} in aa coords, one
    protein. Returns the route node list, sorted by start, with TIGRFAM/region
    overlay calls COLLAPSED into an overlapping aSDomain (dropped as nodes) and
    standalone ones KEPT. Never deletes sequence: a dropped overlay's residues
    remain owned by the aSDomain it sat on. Non-overlay domains always kept.
    Returns (nodes, collapsed) for the audit."""
    doms = sorted(domains, key=lambda d: (d["start"], d["end"]))
    keep, collapsed = [], []
    for i, d in enumerate(doms):
        is_overlay = d["raw"] in overlay_names or d.get("is_tigrfam", False)
        if not is_overlay:
            keep.append(d); continue
        # overlay: keep only if it does NOT substantially overlap a non-overlay
        buried = False
        dlen = max(1, d["end"] - d["start"])
        for j, o in enumerate(doms):
            if j == i:
                continue
            o_overlay = o["raw"] in overlay_names or o.get("is_tigrfam", False)
            if o_overlay:
                continue
            ov = overlaps(d["start"], d["end"], o["start"], o["end"])
            if ov > 0 and ov / dlen >= min_overlap_frac:
                buried = True
                collapsed.append((d, o)); break
        if not buried:
            keep.append(d)
    keep.sort(key=lambda d: (d["start"], d["end"]))
    return keep, collapsed


# ----------------------------------------------------------------------------
# 3. WINDOW RULES  (two objects, each identical across datasets)
# ----------------------------------------------------------------------------
K_COVALENT = 20      # flank each side into the two flanking domains
BREAK_CAP = 64       # true-break terminal tail cap (literature-bounded)
MIN_OBS = 3          # below this a window/terminus -> state 'missing'


def covalent_window(protein, left_end, right_start, k=K_COVALENT):
    """Covalent boundary (B or J): one window across the seam,
    [left_end-k : right_start+k], clamped to the protein. Handles the
    overlapping-domain case (right_start <= left_end) by spanning min..max so
    the shared seam is kept, never skipped. Returns (a2,b2,seq)."""
    lo = min(left_end, right_start)
    hi = max(left_end, right_start)
    a2 = max(0, lo - k)
    b2 = min(len(protein), hi + k)
    return a2, b2, protein[a2:b2]


def break_terminus(protein, domain_edge, side, cap=BREAK_CAP):
    """True non-covalent break terminus. side='c': C-terminal tail = last `cap`
    residues from the last domain edge out to the CDS end. side='n':
    N-terminal tail = first `cap` residues from CDS start in to the first
    domain edge. Kept as-is when shorter than cap; never padded.
    Returns (a2,b2,seq)."""
    L = len(protein)
    if side == "c":
        a2 = max(0, domain_edge)          # domain end
        b2 = min(L, domain_edge + cap)
    elif side == "n":
        b2 = min(L, domain_edge)          # domain start
        a2 = max(0, domain_edge - cap)
    else:
        raise ValueError("side must be 'c' or 'n'")
    return a2, b2, protein[a2:b2]


def slot_state(seq, min_obs=MIN_OBS):
    """observed if long enough, else missing (never embed a degenerate window)."""
    return "observed" if len(seq) >= min_obs else "missing"


# ----------------------------------------------------------------------------
# 4b. SHARED CUTTER  (identical for Bode-2 / Bode-1 / natural)
# ----------------------------------------------------------------------------
# A front-end builds, per assembly, a mature protein string plus an ordered
# list of domain nodes:
#   {"start": aa, "end": aa, "canon": type, "break_after": bool}
# where break_after marks a real NON-COVALENT junction (docking/COM/multi-
# protein) between this domain and the next. Everything else is covalent.
# cut_assembly() then emits the connection windows with ONE rule set.

def cut_assembly(protein, nodes, aid, k=K_COVALENT, cap=BREAK_CAP, min_obs=MIN_OBS,
                 long_gap=None):
    """Return (records, stats). records = list of dicts:
      {header, seq, kind ('covalent'|'break'), slot, state, left, right, win}
    Covalent boundary i->i+1: window [left.end-k : right.start+k] on `protein`,
    linker+scar inline; overlap (right.start<=left.end) spans min..max.
    Real break requires explicit physical bounds: a['chain_end'] for the left
    chain and b['chain_start'] for the right chain, in this coordinate system.
    A stitched string alone cannot establish either physical terminus.
    Sub-min_obs -> missing. Long windows are preserved in full, never cropped.
    """
    L = len(protein)
    recs = []
    st = dict(B=0, J=0, break_term=0, missing=0, long=0)
    for i in range(len(nodes) - 1):
        a, b = nodes[i], nodes[i + 1]
        if a.get("break_after"):
            if 'chain_end' not in a or 'chain_start' not in b:
                raise ValueError('Physical chain bounds required for break termini')
            if not (0 <= a['end'] <= a['chain_end'] <= L and
                    0 <= b['chain_start'] <= b['start'] <= L):
                raise ValueError('Invalid physical chain bounds')
            # C-terminus of left
            cs, ce = a["end"], min(a["end"] + cap, a['chain_end'])
            cseq = protein[cs:ce]
            cstate = slot_state(cseq, min_obs)
            recs.append(dict(header=f"{aid}|CB_c|{a['canon']}__{b['canon']}|k={k}|win={cs}:{ce}",
                             seq=cseq if cstate == "observed" else "", kind="break", slot=1,
                             state=cstate, left=a["canon"], right=b["canon"], win=(cs, ce)))
            # N-terminus of right
            ns, ne = max(b["start"] - cap, b['chain_start']), b["start"]
            nseq = protein[ns:ne]
            nstate = slot_state(nseq, min_obs)
            recs.append(dict(header=f"{aid}|CB_n|{a['canon']}__{b['canon']}|k={k}|win={ns}:{ne}",
                             seq=nseq if nstate == "observed" else "", kind="break", slot=2,
                             state=nstate, left=a["canon"], right=b["canon"], win=(ns, ne)))
            st["break_term"] += 2
            if cstate == "missing": st["missing"] += 1
            if nstate == "missing": st["missing"] += 1
            continue
        a2, b2, seq = covalent_window(protein, a["end"], b["start"], k)
        gap = max(0, b["start"] - a["end"])
        flag = ""
        if long_gap and gap > long_gap:
            st["long"] += 1; flag = "|longgap"
        if len(seq) > 512:
            flag += "|over512_preserved"
        # B vs J is annotation only (module boundary); tag from caller if present
        typ = a.get("btype", "B")
        st[typ] = st.get(typ, 0) + 1
        state = slot_state(seq, min_obs)
        if state == 'missing': st['missing'] += 1
        recs.append(dict(header=f"{aid}|{typ}|{a['canon']}__{b['canon']}|gap={gap}|k={k}|win={a2}:{b2}{flag}",
                         seq=seq if state == 'observed' else '', kind="covalent", slot=0, state=state,
                         left=a["canon"], right=b["canon"], win=(a2, b2)))
    return recs, st
