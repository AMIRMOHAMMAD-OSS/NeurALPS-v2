"""Physical masking closure, independent of module partitions and torch.

Caller must establish the cache extraction mode before using this for training.
Whole exact copies of target sequences in other chains are masked as aliases.
Primary targets stay separate from collateral hidden feature slots.
"""


def intersects(a,b):
    return max(a[0],b[0]) < min(a[1],b[1])


def masking_closure(record, primary, parent_lookup, extraction_mode, known_alias_spans=None):
    if extraction_mode not in {'independent_sequence','full_parent_context'}:
        raise ValueError('Verified extraction provenance required before constructing SSL views')
    slots={(s['object_index'],s['slot']):s for s in record['model_slots']}
    primary=set(primary)
    if not primary or any(key not in slots or slots[key]['state']!='OBSERVED' for key in primary):
        raise ValueError('Primary targets must be observed physical slots')
    parents={h:parent_lookup(h) for h in set(record['parent_sequence_hashes'].values())}
    forbidden={h:[] for h in parents}
    for h,spans in (known_alias_spans or {}).items():
        if h not in parents or any(not 0<=a<b<=len(parents[h]) for a,b in spans):
            raise ValueError('Invalid known physical-copy alias span')
        forbidden[h].extend(spans)
    for key in primary:
        s=slots[key]
        target=parents[s['parent_sequence_hash']][s['start']:s['end']]
        if not target:
            raise ValueError('Empty primary target')
        for h,protein in parents.items():
            start=protein.find(target)
            while start!=-1:
                forbidden[h].append((start,start+len(target)))
                start=protein.find(target,start+1)
    hidden=set()
    for key,s in slots.items():
        if s['state']!='OBSERVED':
            continue
        h=s['parent_sequence_hash']
        dependency=(s['start'],s['end']) if extraction_mode=='independent_sequence' else (0,len(parents[h]))
        if any(intersects(dependency,region) for region in forbidden[h]):
            hidden.add(key)
    # Do not expand forbidden residues to collateral whole-domain dependencies.
    if not primary<=hidden:
        raise ValueError('Primary targets not hidden by closure')
    return dict(primary=sorted(primary),hidden=sorted(hidden),forbidden_spans=forbidden)


def domain_view(record, intervals):
    """Map user residue intervals to domain occurrences without changing the graph.

Returns coverage, not an invented residue embedding sliced out of a pooled vector.
Intervals are a list of (protein_uid, start, end) in the mature-chain coordinates.
"""
    result=[]
    for uid,start,end in intervals:
        if not 0<=start<end or uid not in record['parent_sequence_hashes'] or end>record['parent_lengths'][uid]:
            raise ValueError('Invalid requested physical interval')
        for i,d in enumerate(record['route']):
            if d['kind']!=0 or d['protein_uid']!=uid:
                continue
            a,b=max(start,d['start_aa_0based']),min(end,d['end_aa_0based_exclusive'])
            if a<b:
                partial=(a,b)!=(d['start_aa_0based'],d['end_aa_0based_exclusive'])
                result.append(dict(object_index=i,protein_uid=uid,start=a,end=b,
                    requires_residue_features=partial,whole_domain=not partial))
    return result
