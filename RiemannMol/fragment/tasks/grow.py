"""Scaffold growth: force one fixed anchor fragment (which may carry
dangling ring-closure digits marking attachment points, from
safe_codec.smiles_to_safe or hand-authored) as the decode prefix, then
freely sample everything after it. Wraps FragmentModel.decode_scaffold_growth."""
from collections import Counter
from typing import List, Optional

import torch

from ...core import canon


def grow_scaffold(anchor_smiles: str, n_samples: int = 20, temperature: float = 1.0,
                   top_k: Optional[int] = None, min_grow_len: int = 0, max_len: int = 128,
                   model=None) -> List[dict]:
    if model is None:
        from ..inference import load_model
        model = load_model()

    anchor_ids = model.tokenizer.encode(anchor_smiles)
    dot_id = model.tokenizer.encode(".")[0]
    ids = torch.zeros(1, len(anchor_ids), dtype=torch.long)
    ids[0] = torch.tensor(anchor_ids)
    with torch.no_grad():
        z = model.model.encode(ids)  # (1, 1, H) -- encodes ONLY the anchor fragment

    counts = Counter()
    for _ in range(n_samples):
        token_ids = model.model.decode_scaffold_growth(
            z, anchor_ids, dot_id, max_len=max_len, temperature=temperature,
            top_k=top_k, min_grow_len=min_grow_len,
        )
        c = canon(model.tokenizer.decode(token_ids))
        if c is not None:
            counts[c] += 1

    return [{"smiles": smi, "count": cnt, "frac": cnt / n_samples} for smi, cnt in counts.most_common()]
