"""Confidence-state census for the fragment backend's z space -- thin
wrapper over core.census.run_census using the fragment research repo's
own training corpus (SAFE-format) as the default calibration sample."""
from typing import Optional

from ...core.census import run_census
from ..model import DEFAULT_TRAIN_CORPUS


def _default_calibration_smiles(n=500):
    with open(DEFAULT_TRAIN_CORPUS) as f:
        return [next(f).strip() for _ in range(n)]


def census(n_points: int = 1000, n_samples_per_point: int = 100,
           calibration_smiles: Optional[list] = None, model=None, **kwargs) -> dict:
    if model is None:
        from ..inference import load_model
        model = load_model()
    if calibration_smiles is None:
        calibration_smiles = _default_calibration_smiles()
    return run_census(model.model, model.tokenizer, calibration_smiles,
                       n_points=n_points, n_samples_per_point=n_samples_per_point, **kwargs)
