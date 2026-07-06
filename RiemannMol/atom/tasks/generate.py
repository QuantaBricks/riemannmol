"""Generate analogs of a seed molecule: perturb its latent z with Gaussian
noise, decode, canonicalize, deduplicate, and rank by similarity to the
seed -- the one-call version of the encode/sample/decode/canon boilerplate.

std < ~0.6 barely changes anything; std >= ~1.0 jumps to mostly-unrelated
molecules -- there is no smooth "slightly different" middle ground (see
project notes). 1.0 is a reasonable default starting point to sweep from.
"""
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors

from ...core import canon


def generate(
    seed_smiles: str,
    n_samples: int = 20,
    std: float = 1.0,
    min_similarity: Optional[float] = None,
    max_mw: Optional[float] = None,
    include_seed: bool = True,
    model=None,
) -> List[dict]:
    """Returns deduplicated, canonicalized candidates as a list of dicts
    (smiles, similarity, mw), sorted by similarity to the seed descending.
    similarity is Tanimoto over 2048-bit radius-2 Morgan fingerprints.
    """
    if model is None:
        from ..inference import load_model
        model = load_model()

    seed = canon(seed_smiles)
    seed_mol = Chem.MolFromSmiles(seed)
    seed_fp = AllChem.GetMorganFingerprintAsBitVect(seed_mol, 2, 2048)

    z0 = model.encode([seed])
    zs = model.model.sample(z0, n_samples, std)
    smis = model.decode(zs, mode="greedy")

    seen = set()
    results = []
    if include_seed:
        seen.add(seed)
        results.append({"smiles": seed, "similarity": 1.0, "mw": Descriptors.MolWt(seed_mol)})

    for smi in smis:
        c = canon(smi)
        if c is None or c in seen:
            continue
        seen.add(c)
        mol = Chem.MolFromSmiles(c)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        sim = DataStructs.TanimotoSimilarity(seed_fp, fp)
        if min_similarity is not None and sim < min_similarity:
            continue
        mw = Descriptors.MolWt(mol)
        if max_mw is not None and mw > max_mw:
            continue
        results.append({"smiles": c, "similarity": sim, "mw": mw})

    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results
