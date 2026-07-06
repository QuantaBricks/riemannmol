"""Linear interpolation between two molecules' z, decoded at each step.
Empirical finding: this is a near-step-function, not a smooth morph --
each endpoint's basin decodes deterministically, with one narrow ambiguous
("entangled") crossing near the midpoint. mode="sample" reveals that
crossing's full mixture; mode="greedy" only shows one (possibly
unrepresentative) point in it."""
from collections import Counter
from typing import List

from ...core import canon, sample_decode


def interpolate(smiles_a: str, smiles_b: str, n_steps: int = 9, mode: str = "sample",
                 n_samples: int = 100, temperature: float = 1.0, model=None) -> List[dict]:
    if model is None:
        from ..inference import load_model
        model = load_model()

    ca, cb = canon(smiles_a), canon(smiles_b)
    z = model.encode([ca, cb])
    z_a, z_b = z[0, 0], z[1, 0]

    results = []
    for i in range(n_steps):
        t = i / (n_steps - 1) if n_steps > 1 else 0.0
        z_t = (1 - t) * z_a + t * z_b
        if mode == "greedy":
            smi = model.tokenizer.decode(model.model.decode_greedy(z_t.unsqueeze(0).unsqueeze(0))[0])
            results.append({"t": t, "top_smiles": canon(smi), "top_frac": 1.0, "n_distinct": 1})
        else:
            token_lists = sample_decode(model.model, model.tokenizer, z_t, n_samples, temperature)
            smis = [canon(model.tokenizer.decode(ids)) for ids in token_lists]
            counts = Counter(smis)
            top_smi, top_count = counts.most_common(1)[0]
            results.append({
                "t": t,
                "top_smiles": top_smi,
                "top_frac": top_count / len(smis),
                "n_distinct": len(counts),
                "distribution": counts.most_common(5),
            })
    return results
