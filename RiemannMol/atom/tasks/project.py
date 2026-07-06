"""Compare molecules through an attached metric head (e.g. "tanimoto" for
structural similarity, "potency_egfr" for EGFR-potency-likeness) -- thin
wrappers over AtomInference.project/.distance."""


def project(smiles: str, head: str, model=None):
    if model is None:
        from ..inference import load_model
        model = load_model()
    return model.project(smiles, head)


def distance(smiles_a: str, smiles_b: str, head: str, model=None) -> float:
    if model is None:
        from ..inference import load_model
        model = load_model()
    return model.distance(smiles_a, smiles_b, head)
