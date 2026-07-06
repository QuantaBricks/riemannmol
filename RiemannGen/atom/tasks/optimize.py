"""Property-guided molecule generation: CMA-ES search in the atom
backend's raw z space, decoding every candidate to a real molecule and
scoring it with a real property function every generation (not a metric-head
shortcut -- see project notes: z''-distance does not belong in a scoring
loop, decode -> real property is strictly more accurate and no more
expensive, since decoding is unavoidable anyway).

Default objective is QED (drug-likeness), which has no known
reward-hacking degenerate solution. Penalized logP is available but is
documented (Renz et al. 2019, and this project's own CMA-ES replication)
to be exploitable via long saturated carbon chains -- pass your own
`scoring_fn` for anything property-sensitive.
"""
from typing import Callable, List, Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED, Crippen

from ...core import canon


def _qed_score(smi: str) -> float:
    m = Chem.MolFromSmiles(smi)
    return QED.qed(m) if m is not None else -1.0


def _logp_score(smi: str) -> float:
    m = Chem.MolFromSmiles(smi)
    return Crippen.MolLogP(m) if m is not None else -100.0


BUILTIN_SCORERS = {
    "qed": _qed_score,
    "logp": _logp_score,
}


def optimize(
    seed_smiles: str,
    property: str = "qed",
    scoring_fn: Optional[Callable[[str], float]] = None,
    n_steps: int = 30,
    popsize: int = 20,
    sigma0_fraction: float = 0.2,
    invalid_penalty: float = -1.0,
    model=None,
) -> List[dict]:
    """CMA-ES-guided optimization of `property` (or a custom `scoring_fn`,
    which takes a canonical SMILES and returns a float to maximize) starting
    from `seed_smiles`. Returns the full history of (generation, smiles,
    score) for every strictly-improving candidate found, best last.

    Higher score is better. Invalid decodes get `invalid_penalty`, not
    excluded -- CMA-ES needs a real cost value for every candidate it
    proposes.
    """
    import cma

    if model is None:
        from ..inference import load_model
        model = load_model()

    score_fn = scoring_fn or BUILTIN_SCORERS[property]

    seed = canon(seed_smiles)
    z0 = model.encode([seed])[0, 0].numpy().astype(np.float64)
    sigma0 = sigma0_fraction * float(np.linalg.norm(z0))

    es = cma.CMAEvolutionStrategy(z0, sigma0, {"popsize": popsize, "verbose": -9})

    seed_score = score_fn(seed)
    best_score = seed_score
    history = [{"generation": 0, "smiles": seed, "score": seed_score}]

    for gen in range(1, n_steps + 1):
        candidates = es.ask()
        z_batch = torch.tensor(np.stack(candidates), dtype=torch.float32).unsqueeze(1)
        smis = model.decode(z_batch, mode="greedy")

        costs = []
        for smi in smis:
            c = canon(smi)
            score = score_fn(c) if c is not None else invalid_penalty
            costs.append(-score)  # cma minimizes
            if c is not None and score > best_score:
                best_score = score
                history.append({"generation": gen, "smiles": c, "score": score})
        es.tell(candidates, costs)

    return history
