"""Fragment backend: SAFE-encoded model (loads the checkpoint trained in
the research repo at fragment/checkpoints/fragment_run/final.pt).

FragmentModel adds two decode-time-only capabilities on top of
core.base_model.LatentSeqModel (no architecture change needed for either --
decode_greedy/decode_beam already run a causal, KV-cached autoregressive
loop, so forcing a known prefix instead of feeding back the argmax token is
a decode-time change only):

- decode_scaffold_growth: force one fixed anchor fragment as a prefix, then
  freely sample everything after it. Covers single/multi-site scaffold
  decoration (the anchor can carry any number of dangling ring-closure
  digits) and superstructure generation.
- decode_fragment_linking: force two fixed anchor fragments, freely sample
  only the linker span between them. **Internal-only, not exposed via
  FragmentInference or the CLI** -- has a real decode-loop bug (the linker
  sampling loop terminates on the first '.' it samples, so any linker that
  itself needs an internal '.' can never complete; confirmed via 0/8, 0/10
  all-invalid on Aspirin/Imatinib using each molecule's own true z). Fixing
  it is out of scope for this refactor -- kept here, unwrapped, so the code
  and the bug report stay next to each other.

Local editing (fix a prefix of fragments, freely continue the rest) needs
no new method here at all -- it's the exact same forced_prefix_then_sample
mechanism in core.decode_utils already used for the atom backend's
scaffold extension. See fragment/tasks/edit.py.
"""
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

from ..core import SmilesTokenizer, load_trained
from ..core.base_model import LatentSeqModel
from ..core.hf_download import resolve_path

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = _REPO_ROOT / "fragment/checkpoints/fragment_run/final.pt"
DEFAULT_VOCAB = _REPO_ROOT / "fragment/vocab_new.txt"
DEFAULT_TRAIN_CORPUS = _REPO_ROOT / "fragment/data/combined/train_fragment.txt"


def _sample_token(logits: torch.Tensor, temperature: float = 0.0, top_k: Optional[int] = None) -> int:
    """logits: (1, V). temperature=0 -> greedy argmax (deterministic)."""
    if temperature <= 0:
        return logits.argmax(dim=-1).item()
    scaled = logits / temperature
    if top_k is not None:
        v, _ = scaled.topk(min(top_k, scaled.size(-1)))
        scaled = scaled.masked_fill(scaled < v[..., [-1]], float("-inf"))
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, 1).item()


class FragmentModel(LatentSeqModel):
    """Identical to LatentSeqModel except for the two extra decode methods
    described in this module's docstring."""

    def _kv_step_fn(self, z: torch.Tensor):
        """Returns a closure `_step(tok_id)` that feeds one token through the
        causal decoder at the next position, advancing a private KV cache,
        and returns logits for the following token. Shared by both custom
        decode methods so the feed/predict invariant only needs to be
        gotten right once."""
        device = z.device
        kv_caches = [None] * self.num_layers
        state = {"step": 0}

        def _step(tok_id: int):
            cur_t = torch.tensor([[tok_id]], dtype=torch.long, device=device)
            pos_t = torch.tensor([[state["step"]]], device=device)
            x = self.word_embed(cur_t) + self.dec_pos_embed(pos_t)
            new_caches = []
            for i, layer in enumerate(self.dec_layers):
                x, nkv = layer(x, z, kv_caches[i])
                new_caches.append(nkv)
            kv_caches[:] = new_caches
            state["step"] += 1
            x = self.dec_final_norm(x)
            return self.lm_head(x[:, 0, :])

        return _step, state

    def decode_scaffold_growth(
        self,
        z: torch.Tensor,
        anchor_ids: List[int],
        dot_id: int,
        max_len: Optional[int] = None,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        min_grow_len: int = 0,
    ) -> List[int]:
        """z: (1, 1, H) latent for a single molecule.

        anchor_ids: token ids for the fixed fragment to grow from (encode
        with the same tokenizer used for training; no BOS/EOS/'.'). It can
        carry any number of dangling ring-closure digits (from
        `safe_codec.smiles_to_safe`, or hand-authored) -- one for
        single-site scaffold decoration, several for multi-site /
        superstructure generation. Free generation can close any of them
        (in any order) and/or introduce entirely new rings/fragments, since
        nothing after the forced anchor is constrained.

        Returns anchor_ids + '.' + <freely generated> + EOS (no leading BOS).

        Caveat: this can only grow off the *end* of the forced anchor -- it
        cannot insert branches at an arbitrary interior atom of anchor_ids
        that wasn't given its own ring-closure digit up front (would need a
        fill-in-the-middle-style training objective, which this model
        doesn't have).
        """
        max_len = min(max_len or self.max_len, self.max_len)
        BOS, EOS = SmilesTokenizer.BOS_ID, SmilesTokenizer.EOS_ID
        _step, state = self._kv_step_fn(z)

        forced_prefix = list(anchor_ids) + [dot_id]
        generated: List[int] = []
        cur = BOS
        for tok_id in forced_prefix:
            if state["step"] >= max_len:
                return generated
            _step(cur)
            cur = tok_id
            generated.append(tok_id)

        grown = 0
        while state["step"] < max_len:
            logits = _step(cur)
            if grown < min_grow_len:
                logits = logits.clone()
                logits[:, dot_id] = float("-inf")
                logits[:, EOS] = float("-inf")
            next_tok = _sample_token(logits, temperature, top_k)
            cur = next_tok
            if next_tok == EOS:
                break
            generated.append(next_tok)
            grown += 1

        return generated

    def decode_fragment_linking(
        self,
        z: torch.Tensor,
        frag1_ids: List[int],
        frag2_ids: List[int],
        dot_id: int,
        max_linker_len: int = 40,
        max_total_len: Optional[int] = None,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        min_linker_len: int = 0,
    ) -> List[int]:
        """**Internal/experimental -- known-broken, not part of the public
        application surface.** See this module's docstring. Kept for
        reference/future fix, not wrapped by FragmentInference or the CLI.

        z: (1, 1, H) latent for a single molecule. frag1_ids/frag2_ids:
        token ids for the two anchor fragments (no BOS/EOS/'.'). Returns
        frag1 + '.' + linker + '.' + frag2 + EOS (no leading BOS).
        """
        max_total_len = min(max_total_len or self.max_len, self.max_len)
        BOS, EOS = SmilesTokenizer.BOS_ID, SmilesTokenizer.EOS_ID
        _step, state = self._kv_step_fn(z)

        forced_prefix = list(frag1_ids) + [dot_id]
        forced_suffix = [dot_id] + list(frag2_ids) + [EOS]

        generated: List[int] = []
        cur = BOS
        for tok_id in forced_prefix:
            if state["step"] >= max_total_len:
                return generated
            _step(cur)
            cur = tok_id
            generated.append(tok_id)

        linker_len = 0
        for _ in range(max_linker_len):
            if state["step"] >= max_total_len - len(forced_suffix):
                break
            logits = _step(cur)
            if linker_len < min_linker_len:
                logits = logits.clone()
                logits[:, dot_id] = float("-inf")
                logits[:, EOS] = float("-inf")
            next_tok = _sample_token(logits, temperature, top_k)
            cur = next_tok
            if next_tok in (dot_id, EOS):
                break
            generated.append(next_tok)
            linker_len += 1

        for tok_id in forced_suffix:
            if state["step"] >= max_total_len:
                break
            _step(cur)
            cur = tok_id
            generated.append(tok_id)

        return generated


def load_fragment_model(checkpoint=DEFAULT_CHECKPOINT, device: str = "cpu"):
    # Fragment's checkpoint was trained without the Perceiver encoder's
    # key-padding mask (a divergence from the atom backend's model) --
    # default_use_key_padding_mask=False preserves that, since fragment's
    # already-trained weights were never exposed to the masked computation.
    checkpoint = resolve_path(checkpoint, "fragment/final.pt")
    return load_trained(checkpoint, device=device, model_cls=FragmentModel,
                         default_use_key_padding_mask=False)
