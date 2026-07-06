"""Confidence-state census: classify random points in a model's latent
space by how "decided" the decoder is about them, using sample_decode
(sampled, not greedy -- see decode_utils.py's module docstring for why
greedy decoding can misrepresent a genuinely ambiguous point).

Four states, based on the single largest outcome bucket among N sampled
decodes at a point (canonicalized SMILES, or an "invalid" bucket for
unparseable decodes):

- pure: largest bucket is one valid molecule, >=pure_threshold of samples
- entangled: largest bucket is one valid molecule, <pure_threshold
  (probability mass genuinely split across multiple distinct molecules)
- dead: largest bucket is "invalid SMILES", >=pure_threshold
- idle: largest bucket is "invalid SMILES", <pure_threshold
"""
import time
from collections import Counter
from typing import Iterable, Optional, Tuple

import torch

from .canon import canon
from .decode_utils import sample_decode

INVALID = "<INVALID>"


def classify_state(canon_smis, pure_threshold: float = 0.99) -> Tuple[str, float]:
    counts = Counter(canon_smis)
    top_label, top_count = counts.most_common(1)[0]
    frac = top_count / len(canon_smis)
    if top_label == INVALID:
        return ("dead", frac) if frac >= pure_threshold else ("idle", frac)
    else:
        return ("pure", frac) if frac >= pure_threshold else ("entangled", frac)


def run_census(
    model,
    tok,
    calibration_smiles: Iterable[str],
    n_points: int = 1000,
    n_samples_per_point: int = 100,
    temperature: float = 1.0,
    pure_threshold: float = 0.99,
    seed: int = 0,
    log_path: Optional[str] = None,
    progress_every: int = 50,
):
    """calibration_smiles: real molecules (already canonical) used only to
    calibrate the per-dimension std of "typical" z, so random points are
    sampled at the same scale as real molecule encodings rather than an
    arbitrary/out-of-scale radius. Returns a dict with state_counts and a
    per-point log."""
    calib_smis = list(calibration_smiles)
    ids = [tok.encode(s) for s in calib_smis]
    maxlen = max(len(e) for e in ids)
    batch = torch.zeros(len(ids), maxlen, dtype=torch.long)
    for i, e in enumerate(ids):
        batch[i, : len(e)] = torch.tensor(e)
    with torch.no_grad():
        z_calib = model.encode(batch).squeeze(1)
    per_dim_std = z_calib.std(dim=0).mean().item()
    z_dim = z_calib.shape[1]

    state_counts = Counter()
    per_point_log = []
    torch.manual_seed(seed)
    t_start = time.time()
    logf = open(log_path, "w") if log_path else None
    try:
        for i in range(n_points):
            z = (torch.randn(1, z_dim) * per_dim_std)[0]
            with torch.no_grad():
                token_lists = sample_decode(model, tok, z, n_samples_per_point, temperature)
            smis = [tok.decode(ids_) for ids_ in token_lists]
            canon_smis = [canon(s) or INVALID for s in smis]
            state, frac = classify_state(canon_smis, pure_threshold)
            state_counts[state] += 1
            z_norm = z.norm().item()
            per_point_log.append((i, state, frac, z_norm))
            if logf:
                logf.write(f"{i}\t{state}\t{frac:.3f}\t||z||={z_norm:.3f}\n")
                logf.flush()
            if progress_every and (i + 1) % progress_every == 0:
                elapsed = time.time() - t_start
                print(f"[{i+1}/{n_points}] elapsed={elapsed:.0f}s  running_counts={dict(state_counts)}", flush=True)
    finally:
        if logf:
            logf.close()

    return {
        "state_counts": dict(state_counts),
        "n_points": n_points,
        "n_samples_per_point": n_samples_per_point,
        "per_dim_std": per_dim_std,
        "z_dim": z_dim,
        "elapsed_s": time.time() - t_start,
        "per_point_log": per_point_log,
    }
