"""z_a + z_b / z_a - z_b, decoded via sampling. Empirical finding: no
word2vec-style analogy structure -- addition/subtraction decode to
plausible but unrelated mixtures, not a consistent semantic offset."""
from collections import Counter

from ...core import canon, sample_decode


def arithmetic(smiles_a: str, smiles_b: str, op: str = "+", n_samples: int = 100,
               temperature: float = 1.0, model=None) -> dict:
    if model is None:
        from ..inference import load_model
        model = load_model()

    ca, cb = canon(smiles_a), canon(smiles_b)
    z = model.encode([ca, cb])
    z_a, z_b = z[0, 0], z[1, 0]
    z_result = z_a + z_b if op == "+" else z_a - z_b

    token_lists = sample_decode(model.model, model.tokenizer, z_result, n_samples, temperature)
    smis = [canon(model.tokenizer.decode(ids)) for ids in token_lists]
    counts = Counter(smis)
    top_smi, top_count = counts.most_common(1)[0]
    return {
        "op": op,
        "z_norm": float(z_result.norm()),
        "top_smiles": top_smi,
        "top_frac": top_count / len(smis),
        "n_distinct": len(counts),
        "distribution": counts.most_common(5),
    }
