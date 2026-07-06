"""Atom backend: loads the plain-SMILES latent model trained in the
research repo at atom/checkpoints/scratch_run/final-atom.pt. Just a config
point (checkpoint/vocab paths) -- the architecture itself is
core.base_model.LatentSeqModel, unmodified."""
from pathlib import Path

from ..core import load_trained
from ..core.hf_download import resolve_path

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = _REPO_ROOT / "atom/checkpoints/scratch_run/final-atom.pt"
DEFAULT_VOCAB = _REPO_ROOT / "atom/vocab_new.txt"
DEFAULT_TRAIN_CORPUS = _REPO_ROOT / "atom/data/combined/train.txt"


def load_atom_model(checkpoint=DEFAULT_CHECKPOINT, device: str = "cpu"):
    checkpoint = resolve_path(checkpoint, "atom/final-atom.pt")
    return load_trained(checkpoint, device=device, default_use_key_padding_mask=True)
