"""Scaffold extension: force an incomplete SMILES prefix (an open branch,
unsatisfied valence) as the decode history, then freely sample a
completion. Empirical finding: only works with a genuinely *incomplete*
prefix -- forcing a complete, valence-satisfied molecule (a "finished"
scaffold) gets a 100% no-op every time, since the model correctly reads
that as "already done," not a capability failure."""
from collections import Counter
from typing import List

import torch

from ...core import canon
from ...core.decode_utils import forced_prefix_then_sample


def extend(prefix_smiles: str, n_samples: int = 30, temperature: float = 1.2,
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
