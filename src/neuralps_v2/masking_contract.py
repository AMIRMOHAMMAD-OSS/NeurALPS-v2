"""Provenance-only masking: target ownership differs from PLM dependencies.

No torch dependency. The caller applies returned masks to every neural input and
recomputes raw features. Coordinates refer to translated source proteins.
"""
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, order=True)
class Span:
    source: str
    start: int
    stop: int

    def __post_init__(self):
        if not self.source or self.start < 0 or self.stop <= self.start:
            raise ValueError('A physical span needs a source and nonempty half-open interval.')

    def intersects(self, other):
        return self.source == other.source and max(self.start,other.start) < min(self.stop,other.stop)


@dataclass(frozen=True)
class CopyAlias:
    """An exact, orientation-preserving translated AA copy, in both directions."""
    a: Span
    b: Span

    def __post_init__(self):
        if self.a.stop-self.a.start != self.b.stop-self.b.start:
            raise ValueError('Exact-copy coordinate intervals must have equal length.')


@dataclass(frozen=True)
class SlotProvenance:
    owned: tuple[Span, ...]
    dependencies: tuple[Span, ...]
    sequence_hash: str


def expand_aliases(target_spans: Sequence[Span], aliases: Sequence[CopyAlias]):
    """Propagate only forbidden residue slices, never all of a collateral channel."""
    result = set(target_spans)
    pending = list(result)
    while pending:
        span = pending.pop()
        for alias in aliases:
            for src,dst in ((alias.a,alias.b),(alias.b,alias.a)):
                if span.intersects(src):
                    lo,hi = max(span.start,src.start), min(span.stop,src.stop)
                    mapped = Span(dst.source, dst.start+lo-src.start, dst.start+hi-src.start)
                    if mapped not in result:
                        result.add(mapped)
                        pending.append(mapped)
    return tuple(sorted(result))


def mask_slots(slots: Sequence[SlotProvenance], primary: Sequence[int],
               aliases: Sequence[CopyAlias]=()):
    """Return (hidden_slot_booleans, forbidden_residue_spans).

    Slots here are observed sequence slots only. Empty/NA slots bypass this list.
    A sequence hash denotes the entire owned object sequence, not just a core.
    All aliases for subfragments/raw views must be supplied by the manifest.
    """
    if not primary or any(i < 0 or i >= len(slots) for i in primary):
        raise ValueError('At least one in-range primary target is required.')
    if any(not s.owned or not s.dependencies or not s.sequence_hash for s in slots):
        raise ValueError('Observed slots require complete ownership/dependency provenance.')
    keys = {slots[i].sequence_hash for i in primary}
    # Exact whole-object copies are targets too, even under a different parent.
    owned = [r for slot in slots if slot.sequence_hash in keys for r in slot.owned]
    forbidden = expand_aliases(owned, aliases)
    mask = [i in primary or slot.sequence_hash in keys or
            any(d.intersects(r) for d in slot.dependencies for r in forbidden)
            for i,slot in enumerate(slots)]
    return mask, forbidden


def purged_split(part_sets: Sequence[set[str]], heldout: set[str], eligible=None):
    indices = list(range(len(part_sets))) if eligible is None else list(eligible)
    train = [i for i in indices if not part_sets[i].intersection(heldout)]
    test = [i for i in indices if part_sets[i].intersection(heldout)]
    return train,test
