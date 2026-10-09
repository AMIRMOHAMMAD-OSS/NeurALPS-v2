"""Empirical natural-reference ranks; no activity outcomes or model fitting."""
import bisect
from collections import Counter, defaultdict
import hashlib
import json
import math


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(*values):
    return hashlib.sha256(canonical(values).encode()).hexdigest()


def query_id(row):
    return row['assembly_id'] + '|object=' + str(row['object_index'])


def keys(row, protocol):
    """No phenotype, raw score, rank or reference support enters the strata."""
    if not isinstance(row['hidden_fraction'], (int,float)) or not 0 <= row['hidden_fraction'] <= 1:
        raise ValueError('Invalid masking fraction: '+query_id(row))
    if row['kind'] not in (0,1,2) or len(row['domain_types']) != (1 if row['kind']==0 else 2):
        raise ValueError('Invalid domain-type stratum: '+query_id(row))
    band = sum(row['hidden_fraction'] >= edge for edge in protocol['closure_fraction_edges'])
    common = [row['kind'], row['domain_types'], row['mode'], band]
    lengths = [s['length_bin'] for s in row['slots']]
    return [('type_mask_length', canonical(common + [lengths])),
            ('type_mask', canonical(common))]


def eligible(row):
    if row['status'] != 'SCORED':
        return False
    if not math.isfinite(row['native_score']) or not -1.00001 <= row['native_score'] <= 1.00001:
        raise ValueError('Invalid score: ' + query_id(row))
    return True


def build_reference(rows, metadata, protocol):
    candidates = defaultdict(dict)
    counts = Counter()
    seen = set()
    for row in rows:
        qid = query_id(row)
        if qid in seen:
            raise ValueError('Duplicate physical query: ' + qid)
        seen.add(qid)
        m = metadata[row['assembly_id']]
        if m['role'] != 'calibration' or not eligible(row):
            continue
        component = m['component_id']
        for level, key in keys(row, protocol):
            group = canonical([level, key])
            counts[group] += 1
            choice = digest(protocol['seed'], 'reference_representative', component, group, qid)
            old = candidates[group].get(component)
            if old is None or choice < old['selection_hash']:
                candidates[group][component] = dict(component_id=component, query_id=qid,
                    selection_hash=choice, score=row['native_score'],
                    native_hashes=row['native_hashes'], parent_hashes=row['parent_hashes'])
    groups = {}
    for group, picked in sorted(candidates.items()):
        values = [picked[k] for k in sorted(picked)]
        groups[group] = dict(n=len(values), candidate_objects=counts[group],
            scores=sorted(r['score'] for r in values), representatives=values)
    return dict(version=protocol['version'], protocol_sha256=digest(protocol),
                interpretation='Empirical native-context reference, not activity calibration', groups=groups)


def apply_one(row, reference, protocol):
    result = dict(row)
    result.update(natural_percentile_0_to_100=None, lower_tail_rank=None,
        reference_components=0, reference_candidate_objects=0, reference_level=None,
        reference_key=None, tail_resolution=None, reference_feature_overlap_components=None,
        reference_parent_overlap_components=None, warning=None,
        scale_status='UNSCORED' if row['status'] != 'SCORED' else 'INSUFFICIENT_REFERENCE')
    if not eligible(row):
        return result
    best_support = 0
    for level, key in keys(row, protocol):
        group = canonical([level, key])
        ref = reference['groups'].get(group)
        if ref is None:
            continue
        best_support = max(best_support, ref['n'])
        if ref['n'] < protocol['minimum_reference_components']:
            continue
        ss, score, n = ref['scores'], row['native_score'], ref['n']
        left, right = bisect.bisect_left(ss, score), bisect.bisect_right(ss, score)
        rank = (1 + right) / (n + 1)
        warning = 'LOW_NATURAL_SUPPORT' if rank <= protocol['low_tail_threshold'] else 'NOT_LOW_BY_REFERENCE'
        if n >= protocol['very_low_minimum_components'] and rank <= protocol['very_low_tail_threshold']:
            warning = 'VERY_LOW_NATURAL_SUPPORT'
        hs, ps = set(row['native_hashes']), set(row['parent_hashes'])
        result.update(natural_percentile_0_to_100=100 * (left + .5 * (right-left)) / n,
            lower_tail_rank=rank, reference_components=n,
            reference_candidate_objects=ref['candidate_objects'], reference_level=level,
            reference_key=group, tail_resolution=1/(n+1), warning=warning,
            scale_status='SCALED' if level=='type_mask_length' else 'SCALED_LENGTH_UNMATCHED',
            reference_feature_overlap_components=sum(bool(hs & set(x['native_hashes'])) for x in ref['representatives']),
            reference_parent_overlap_components=sum(bool(ps & set(x['parent_hashes'])) for x in ref['representatives']))
        return result
    result['reference_components'] = best_support
    return result


def wilson(k, n):
    if n == 0:
        return [None, None]
    z = 1.959963984540054
    den = 1 + z*z/n
    centre = (k/n + z*z/(2*n))/den
    width = z*math.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/den
    return [max(0., centre-width), min(1., centre+width)]


def audit_scale(scaled_rows, metadata, protocol):
    """One score-blind audit representative per component and resolved stratum."""
    picked = defaultdict(dict)
    counts = Counter()
    for row in scaled_rows:
        m = metadata[row['assembly_id']]
        if m['role'] != 'audit':
            continue
        counts[row['scale_status']] += 1
        if row['reference_key'] is None:
            continue
        group, component = row['reference_key'], m['component_id']
        choice = digest(protocol['seed'], 'audit_representative', component, group, query_id(row))
        old = picked[group].get(component)
        if old is None or choice < old[0]:
            picked[group][component] = (choice, row)
    groups = []
    for key, representatives in sorted(picked.items()):
        rr = [r for _,r in representatives.values()]
        n = len(rr)
        low = sum(r['lower_tail_rank'] <= protocol['low_tail_threshold'] for r in rr)
        ci = wilson(low,n)
        if n < protocol['minimum_audit_components_for_rate_assessment']:
            status = 'TOO_FEW_AUDIT_COMPONENTS'
        elif ci[1] <= protocol['warning_rate_diagnostic_ceiling']:
            status = 'WITHIN_DECLARED_RATE_CEILING'
        elif ci[0] > protocol['warning_rate_diagnostic_ceiling']:
            status = 'EXCESS_LOW_TAIL_WARNINGS'
        else:
            status = 'INDETERMINATE_RATE'
        groups.append(dict(reference_key=key,audit_components=n,low_tail_warnings=low,
            warning_rate=low/n,warning_rate_wilson95=ci,rate_assessment=status,
            target_tail_rank=protocol['low_tail_threshold'],diagnostic_ceiling=protocol['warning_rate_diagnostic_ceiling'],
            representative_query_ids=sorted(query_id(r) for r in rr)))
    return dict(object_coverage=dict(counts),groups=groups,
        rate_assessments=dict(Counter(g['rate_assessment'] for g in groups)),
        interpretation='Descriptive natural audit; no engineered false-alarm or activity guarantee',
        formal_conformal_validity_claim=False,multiple_testing_adjusted=False)


def assembly_summaries(rows, metadata):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['assembly_id']].append(row)
    output=[]
    for aid, rr in sorted(grouped.items()):
        scored=[r for r in rr if r['lower_tail_rank'] is not None]
        output.append(dict(assembly_id=aid,dataset=metadata[aid]['dataset'],role=metadata[aid]['role'],
            objects=len(rr),scaled_objects=len(scored),unscaled_objects=len(rr)-len(scored),
            low_support_objects=sum(r['warning'] in ('LOW_NATURAL_SUPPORT','VERY_LOW_NATURAL_SUPPORT') for r in scored),
            minimum_natural_percentile=min((r['natural_percentile_0_to_100'] for r in scored),default=None),
            whole_assembly_activity_prediction=None,whole_assembly_alarm_calibrated=False))
    return output
