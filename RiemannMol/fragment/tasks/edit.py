"""Local editing: fix a prefix of fragments (e.g. everything up to and
including the ring you want to keep), freely continue the rest -- the same
forced_prefix_then_sample mechanism used for the atom backend's scaffold
extension (core/decode_utils.py), applied to fragment-encoded prefixes.

Key precondition (validated on Imatinib piperazine->morpholine/
homopiperazine edits): z must encode *only* the retained prefix, not the
whole original molecule -- otherwise z leaks the discarded part's
information and "free" continuation just reconstructs the original with
noise, not real novelty. This function always encodes prefix_smiles alone.

Directional limitation: can only precisely preserve content *before* the
edit point; cannot simultaneously force specific content *after* it. The
closer the edit point is to the start of the sequence (more must be freely
regenerated), the closer this degrades to ordinary free generation's
reliability."""
from collections import Counter
from typing import List

import torch

from ...core import canon
from ...core.decode_utils import forced_prefix_then_sample


def edit_locally(prefix_smiles: str, n_samples: int = 30, temperature: float = 1.5,
                  max_len: int = 128, model=None) -> List[dict]:
    if model is None:
        from ..inference import load_model
        model = load_model()

    prefix_ids = model.tokenizer.encode(prefix_smiles)
    ids = torch.zeros(1, len(prefix_ids), dtype=torch.long)
    ids[0] = torch.tensor(prefix_ids)
    with torch.no_grad():
        z_prefix = model.model.encode(ids)[0, 0]

    token_lists = forced_prefix_then_sample(model.model, model.tokenizer, z_prefix, prefix_ids,
                                             n_samples, temperature, max_len)
    counts = Counter()
    for tids in token_lists:
        c = canon(model.tokenizer.decode(tids))
        if c is not None:
            counts[c] += 1

    return [{"smiles": smi, "count": cnt, "frac": cnt / n_samples} for smi, cnt in counts.most_common()]
