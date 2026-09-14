"""Dependency-light metrics and donor purging; no fitting on external Bode-2.

Metrics are computed per specified cohort. Continuous TE AUROC requires actual
per-construct TE scores; it cannot be inferred from the two manuscript bins.
"""
import numpy as np


def _binary(y):
    y = np.asarray(y)
    if y.ndim != 1 or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError('Expected nonempty binary label vector.')
    return y.astype(int)


def average_ranks(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('Expected finite vector.')
    order = np.argsort(x, kind='stable'); result = np.empty(len(x), float)
    lo = 0
    while lo < len(x):
        hi = lo + 1
        while hi < len(x) and x[order[hi]] == x[order[lo]]:
            hi += 1
        result[order[lo:hi]] = (lo + hi + 1) / 2
        lo = hi
    return result


def auroc(y, score):
    y = _binary(y); score = np.asarray(score, float)
    if score.shape != y.shape or not np.isfinite(score).all():
        raise ValueError('Score/label mismatch or nonfinite scores.')
    pos = y.sum(); neg = len(y) - pos
    if not pos or not neg:
        return None
    ranks = average_ranks(score)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def spearman(y, score):
    y, score = np.asarray(y, float), np.asarray(score, float)
    if y.shape != score.shape or y.ndim != 1 or len(y) < 2:
        raise ValueError('Expected matching vectors with at least two observations.')
    a, b = average_ranks(y), average_ranks(score)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def confusion_metrics(tp, fp, fn, tn):
    counts = [tp, fp, fn, tn]
    if any(type(x) is not int or x < 0 for x in counts) or sum(counts) == 0:
        raise ValueError('Expected nonnegative integer counts with nonzero total.')
    ratio = lambda a, b: a / b if b else None
    sensitivity, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    return dict(n=sum(counts), accuracy=(tp + tn) / sum(counts), precision=ratio(tp, tp + fp),
                sensitivity=sensitivity, specificity=specificity,
                balanced_accuracy=None if sensitivity is None or specificity is None
                else (sensitivity + specificity) / 2)


def donor_fold(records, heldout, eligible=None):
    """Requires engineered-donor memberships; invariant recipient scaffold is separate.

    A multiswap construct enters test if any engineered donor is held out.
    Control inclusion must be specified by the caller's eligible cohort.
    """
    heldout = set(heldout)
    if not heldout:
        raise ValueError('Need held-out donors.')
    eligible = list(range(len(records))) if eligible is None else list(eligible)
    if len(set(eligible)) != len(eligible):
        raise ValueError('Duplicate eligible rows.')
    parts = []
    for record in records:
        if 'engineered_donors' not in record:
            raise ValueError('Missing donor membership; do not infer from construct names.')
        donors = record['engineered_donors']
        if not isinstance(donors, list) or any(not isinstance(x, str) or not x for x in donors):
            raise ValueError('engineered_donors must be a list of nonempty strings.')
        parts.append(set(donors))
    train = [i for i in eligible if not parts[i] & heldout]
    test = [i for i in eligible if parts[i] & heldout]
    if not train or not test:
        raise ValueError('Empty donor split.')
    return train, test


def igem_reported_threshold():
    return dict(cohort='105 single exchanges in supplied iGEM manuscript v14',
                threshold='donor-source TE similarity >50%', tp=43, fp=8, fn=20, tn=34,
                metrics=confusion_metrics(43, 8, 20, 34),
                evaluation='descriptive same-cohort calculation; not external validation',
                continuous_te_auroc=None)


def main():
    import json
    print(json.dumps(igem_reported_threshold(), indent=2))


if __name__ == '__main__':
    main()
