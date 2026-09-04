"""Answer parsing and paired diagnostics. No model or GPU dependencies."""
from __future__ import annotations
import re
import string
from collections import Counter
from typing import Mapping, Sequence
import numpy as np

INVALID = '<invalid>'

def parse_answer(text: str, kind: str, options: Mapping[str, str] | None = None, error: str = '') -> str:
    """Parse an explicit answer without turning the first letter of 'car' into C.

    Explicit leading labels take priority over optional explanatory text, consistent
    with the benchmark's letter-only contract. Free prose is not heuristically scored.
    An exact, full option text is accepted only when it identifies one option uniquely.
    """
    if error:
        return INVALID
    raw = str(text or '').strip()
    if kind == 'binary':
        m = re.match(r'^(yes|no)(?=$|[\s.!?,:;])', raw, re.I)
        return m[1].capitalize() + '.' if m else INVALID
    if kind != 'mcq':
        raise ValueError(f'Unknown question type: {kind!r}')
    m = re.match(r'^\s*[\(\[]?\s*([abcd])(?=$|[\s\)\].:\-])', raw, re.I)
    if not m:
        m = re.match(r'^(?:the\s+)?(?:option|answer|choice)\s*(?:is|:|-)\s*([abcd])\b', raw, re.I)
    if m:
        return m[1].upper()
    norm = lambda s: re.sub(r'\s+', ' ', str(s).lower().strip().strip(string.punctuation + ' '))
    matches = [label for label, value in (options or {}).items() if norm(value) == norm(raw)]
    return matches[0] if len(matches) == 1 else INVALID

def summarize(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> dict:
    if len(gold) != len(pred) or not len(gold):
        raise ValueError('Gold/prediction lengths must agree and be nonempty.')
    y, p = np.asarray(gold), np.asarray(pred)
    recall = {label: float(np.mean(p[y == label] == label)) for label in labels if np.any(y == label)}
    counts = Counter(pred)
    result = dict(n=len(y), accuracy=float(np.mean(y == p)), valid_rate=float(np.mean(p != INVALID)),
                  balanced_accuracy=float(np.mean(list(recall.values()))), recall=recall,
                  gold_counts=dict(Counter(gold)), prediction_counts=dict(counts))
    if labels == ['A', 'B', 'C', 'D']:
        rescue = int(np.sum((y != 'A') & (p == y)))
        damage = int(np.sum((y == 'A') & (p != 'A')))
        result.update(always_a=float(np.mean(y == 'A')), a_rate=float(np.mean(p == 'A')),
                      rescue=rescue, damage=damage, excess=(rescue-damage)/len(y))
        assert np.isclose(result['accuracy']-result['always_a'], result['excess'])
    else:
        result.update(tpr=recall.get('Yes.', float('nan')),
                      fpr=float(np.mean(p[y == 'No.'] == 'Yes.')),
                      fnr=float(np.mean(p[y == 'Yes.'] == 'No.')))
    return result

def paired_diagnostics(gold: Sequence[str], before: Sequence[str], after: Sequence[str]) -> dict:
    if not (len(gold) == len(before) == len(after)) or not len(gold):
        raise ValueError('Paired arrays must have the same nonzero length.')
    y, a, b = map(np.asarray, (gold, before, after))
    ca, cb = a == y, b == y
    return dict(n=len(y), delta=float(np.mean(cb.astype(float)-ca)),
                flip_rate=float(np.mean(a != b)), repaired=int(np.sum(~ca & cb)),
                regressed=int(np.sum(ca & ~cb)),
                stable_wrong=int(np.sum((a == b) & ~ca)),
                stable_wrong_rate=float(np.mean((a == b) & ~ca)))

def clustered_interval(values: Sequence[float], scenes: Sequence[str], *, seed: int=20260904, draws: int=10000) -> tuple[float,float]:
    """Percentile CI, resampling complete scenes; ratio preserves item-weighted mean.

    This captures scene sampling variation, NOT variation over decoding seeds.
    """
    if len(values) != len(scenes) or not len(values) or draws < 1:
        raise ValueError('Nonempty equal-length values/scenes and positive draws required.')
    _, inverse = np.unique(np.asarray(scenes), return_inverse=True)
    nscene = int(inverse.max())+1
    sums = np.bincount(inverse, weights=np.asarray(values, dtype=float))
    sizes = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, nscene, size=(draws, nscene))
    estimates = sums[indices].sum(axis=1)/sizes[indices].sum(axis=1)
    return tuple(float(x) for x in np.quantile(estimates, [0.025,0.975]))
