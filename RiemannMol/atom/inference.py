"""Application-layer entry point for the atom backend. AtomInference.encode
/.decode are the base primitive every task in atom.tasks is built from,
and are meant to be directly usable on their own."""
from pathlib import Path
from typing import List, Optional

import torch

from ..core import MetricHead, SmilesTokenizer, load_metric_head, sample_decode
from ..core.hf_download import resolve_path
from ..core.metric_heads import KNOWN_HEADS
from .model import DEFAULT_CHECKPOINT, DEFAULT_VOCAB, load_atom_model

_REPO_ROOT = Path(__file__).resolve().parents[3]


class AtomInference:
    def __init__(self, model, tokenizer: SmilesTokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.heads: dict[str, MetricHead] = {}

    def _tokenize_batch(self, smiles_list: List[str]) -> torch.Tensor:
        encoded = [self.tokenizer.encode(s)[:126] for s in smiles_list]
        maxlen = max(len(e) for e in encoded)
        ids = torch.zeros(len(encoded), maxlen, dtype=torch.long, device=self.device)
        for i, e in enumerate(encoded):
            ids[i, : len(e)] = torch.tensor(e, device=self.device)
        return ids

    def encode(self, smiles_list: List[str]) -> torch.Tensor:
        """Returns z, shape (n, 1, hidden_size)."""
        ids = self._tokenize_batch(smiles_list)
        with torch.no_grad():
            return self.model.encode(ids)

    def decode(self, z: torch.Tensor, mode: str = "greedy", beam_size: int = 3,
               n_samples: int = 30, temperature: float = 1.0) -> List[str]:
        """mode="greedy": one decode per z (argmax). mode="beam": beam
        search, top-1 per z. mode="sample": n_samples independent sampled
        decodes per z (one z at a time)."""
        with torch.no_grad():
            if mode == "greedy":
                token_lists = self.model.decode_greedy(z)
            elif mode == "beam":
                token_lists = self.model.decode_beam(z, beam_size=beam_size)
            elif mode == "sample":
                if z.shape[0] != 1:
                    raise ValueError("mode='sample' decodes one z at a time; pass a (1,1,H) tensor")
                token_lists = sample_decode(self.model, self.tokenizer, z[0, 0], n_samples, temperature)
            else:
                raise ValueError(f"unknown decode mode {mode!r}")
        return [self.tokenizer.decode(tl) for tl in token_lists]

    def attach_head(self, name: str, checkpoint_path: Optional[str] = None) -> None:
        """Registers a MetricHead under `name`. If checkpoint_path is
        omitted, `name` is resolved through KNOWN_HEADS (repo-relative
        path); otherwise checkpoint_path is used directly."""
        path = checkpoint_path
        if path is None:
            if name not in KNOWN_HEADS:
                raise ValueError(f"unknown head {name!r}; pass checkpoint_path explicitly, "
                                  f"or use one of {list(KNOWN_HEADS)}")
            path = str(_REPO_ROOT / KNOWN_HEADS[name])
        self.heads[name] = load_metric_head(path)

    def _get_head(self, head: str) -> MetricHead:
        if head not in self.heads:
            self.attach_head(head)
        return self.heads[head]

    def project(self, smiles: str, head: str) -> torch.Tensor:
        z = self.encode([smiles])[0, 0]
        return self._get_head(head).project(z)

    def distance(self, smiles_a: str, smiles_b: str, head: str) -> float:
        za = self.encode([smiles_a])[0, 0]
        zb = self.encode([smiles_b])[0, 0]
        return self._get_head(head).distance(za, zb)


def load_model(checkpoint=DEFAULT_CHECKPOINT, vocab=DEFAULT_VOCAB, device: str = "cpu") -> AtomInference:
    model = load_atom_model(checkpoint, device=device)
    vocab = resolve_path(vocab, "atom/vocab_new.txt")
    tok = SmilesTokenizer(str(vocab))
    return AtomInference(model, tok, device=device)
