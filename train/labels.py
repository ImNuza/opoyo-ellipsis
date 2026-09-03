"""Positive class = body-weight impact on the floor.

`heeldrop` (dropping onto the heels from standing) and `jump` (landing from a
jump) are the same physical event: a person's mass arriving at the slab
through the feet. Trained separately they are indistinguishable -- both score
~0.72 mean out-of-fold when only heeldrop is positive -- because they are one
class, so we label them as one.

Dropped objects (bag, bottle, chair, key), walking and ambient are negatives.
NOTE: none of these takes is a person going down flat. Heeldrop and jump are
safe proxies for body-weight impact, not recordings of a fall.
"""

POS = {"heeldrop", "jump"}


def y_of(label: str) -> int | None:
    return 1 if label.lower() in POS else 0
