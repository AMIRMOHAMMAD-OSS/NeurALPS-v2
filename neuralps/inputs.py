"""Ordered physical proteins plus approved, half-open domain annotations."""
import json
from pathlib import Path
from .contracts import require, seqsha, WINDOW_POLICY
from .connection_rules import canonical_type, covalent_window, break_terminus

AA = frozenset("ACDEFGHIKLMNPQRSTVWY")

def vocabulary():
    return json.loads((Path(__file__).parent / "data/domain_vocabulary.json").read_text())["vocab"]

def build_record(spec):
    """Protein list order is biosynthetic order; never infer it from names.

    Domain annotations are caller-approved and already overlap-resolved.
    start/end are 0-based half-open on each complete mature physical chain.
    """
    aid = spec.get("assembly_id")
    require(isinstance(aid, str) and bool(aid.strip()), "assembly_id is required")
    proteins = spec.get("proteins")
    require(isinstance(proteins, list) and proteins, "Ordered proteins are required")
    vocab = vocabulary()
    parents, parent_hashes, parent_lengths, seen, domains = {}, {}, {}, set(), []
    for chain, p in enumerate(proteins, 1):
        uid, sequence = p.get("id"), p.get("sequence")
        require(isinstance(uid, str) and uid and uid not in seen, "Protein IDs must be unique")
        require(isinstance(sequence, str) and sequence and set(sequence) <= AA,
                "Sequences must contain uppercase canonical amino acids without spaces or stops")
        seen.add(uid)
        ph = seqsha(sequence)
        parents[ph], parent_hashes[uid], parent_lengths[uid] = sequence, ph, len(sequence)
        calls, intervals = p.get("domains"), set()
        require(isinstance(calls, list) and calls, "Every supplied chain needs approved domains")
        local = []
        for call in calls:
            start, end, raw = call.get("start"), call.get("end"), call.get("type")
            require(type(start) is int and type(end) is int and 0 <= start < end <= len(sequence),
                    "Invalid domain coordinates")
            require(isinstance(raw, str), "Each domain needs a resolved type")
            canon = canonical_type(raw)
            require(canon in vocab and canon not in {"UNKNOWN", "OTHER"},
                    "Unsupported or unresolved domain type: "+raw)
            require((start, end) not in intervals, "Duplicate/conflicting domain interval")
            intervals.add((start, end))
            local.append(dict(kind=0, chain=chain, protein_uid=uid, domain_type=vocab[canon],
                              raw_domain_type=raw, canonical_domain_type=canon,
                              start_aa_0based=start, end_aa_0based_exclusive=end,
                              source_domain_uid=f"{uid}|{canon}|{start}:{end}",
                              seq_hash=seqsha(sequence[start:end])))
        domains.extend(sorted(local, key=lambda d: (d["start_aa_0based"], d["end_aa_0based_exclusive"])))
    route, slots, sequences = [], [], {}
    def slot(j, role, kind, domain, start, end, group):
        uid = domain["protein_uid"]
        ph = parent_hashes[uid]
        sequence = parents[ph][start:end]
        observed = bool(sequence) if kind == 0 else len(sequence) >= 3
        h = seqsha(sequence) if observed else None
        if observed:
            sequences[h] = sequence
        slots.append(dict(object_index=j, slot=role, kind=kind, state="OBSERVED" if observed else "MISSING",
                          sequence_hash=h, length=len(sequence), protein_uid=uid,
                          parent_sequence_hash=ph, start=start, end=end, group=group))
        return h, "OBSERVED" if observed else "MISSING"
    for n, domain in enumerate(domains):
        if n:
            left = domains[n-1]
            is_break = left["chain"] != domain["chain"]
            kind = 2 if is_break else 1
            boundary = dict(kind=kind, chain=-1 if is_break else domain["chain"], domain_type=0,
                            boundary_id=f"{aid}|boundary={n}", left_domain_uid=left["source_domain_uid"],
                            right_domain_uid=domain["source_domain_uid"],
                            left_protein_uid=left["protein_uid"], right_protein_uid=domain["protein_uid"])
            j = len(route)
            if is_break:
                for role, flank, side, edge, name in [(1, left, "c", left["end_aa_0based_exclusive"], "cterm"),
                                                    (2, domain, "n", domain["start_aa_0based"], "nterm")]:
                    a, b, _ = break_terminus(parents[parent_hashes[flank["protein_uid"]]], edge, side)
                    h, state = slot(j, role, 2, flank, a, b, "B:"+name)
                    boundary[name+"_hash"], boundary[name+"_state"] = h, state
            else:
                a, b, _ = covalent_window(parents[parent_hashes[left["protein_uid"]]],
                                          left["end_aa_0based_exclusive"], domain["start_aa_0based"])
                h, state = slot(j, 0, 1, left, a, b, f"C:{left['domain_type']}:{domain['domain_type']}")
                boundary.update(seq_hash=h, state0=state)
            route.append(boundary)
        domain["occurrence_id"] = f"{aid}|domain_occurrence={n+1}"
        slot(len(route), 0, 0, domain, domain["start_aa_0based"], domain["end_aa_0based_exclusive"],
             "D:"+str(domain["domain_type"]))
        route.append(domain)
    record = dict(assembly_id=aid, route=route, model_slots=slots, n_domains=len(domains),
                  n_chains=len(proteins), parent_sequence_hashes=parent_hashes, parent_lengths=parent_lengths,
                  boundary_window_policy=WINDOW_POLICY, cache_dependency_policy="independent_sequence")
    return record, parents, sequences

def region_targets(record, domain_indices):
    ds = list(domain_indices)
    require(ds and len(ds) == len(set(ds)) and all(type(j) is int for j in ds),
            "Provide unique domain object indices")
    require(all(0 <= j < len(record["route"]) and record["route"][j]["kind"] == 0 for j in ds),
            "Region indices must point to domain objects (0, 2, 4, ...)")
    ds = sorted(ds)
    require(ds == list(range(ds[0], ds[-1]+1, 2)), "A joint region must contain contiguous domains")
    bs = [j for j, obj in enumerate(record["route"]) if obj["kind"] != 0 and (j-1 in ds or j+1 in ds)]
    require(bs, "Joint scoring needs at least one touching boundary")
    return ds, bs
