"""Application-layer entry point for the fragment (SAFE) backend."""
from typing import List

import torch

from ..core import SmilesTokenizer, sample_decode
from ..core.hf_download import resolve_path
from .model import DEFAULT_CHECKPOINT, DEFAULT_VOCAB, load_fragment_model


class FragmentInference:
    def __init__(self, model, tokenizer: SmilesTokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _tokenize_batch(self, smiles_list: List[str]) -> torch.Tensor:
        encoded = [self.tokenizer.encode(s)[:126] for s in smiles_list]
        maxlen = max(len(e) for e in encoded)
        ids = torch.zeros(len(encoded), maxlen, dtype=torch.long, device=self.device)
        for i, e in enumerate(encoded):
            ids[i, : len(e)] = torch.tensor(e, device=self.device)
        return ids

    def encode(self, smiles_list: List[str]) -> torch.Tensor:
        ids = self._tokenize_batch(smiles_list)
        with torch.no_grad():
            return self.model.encode(ids)

    def decode(self, z: torch.Tensor, mode: str = "greedy", beam_size: int = 3,
               n_samples: int = 30, temperature: float = 1.0) -> List[str]:
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


def load_model(checkpoint=DEFAULT_CHECKPOINT, vocab=DEFAULT_VOCAB, device: str = "cpu") -> FragmentInference:
    model = load_fragment_model(checkpoint, device=device)
    vocab = resolve_path(vocab, "fragment/vocab_new.txt")
    tok = SmilesTokenizer(str(vocab))
    return FragmentInference(model, tok, device=device)
