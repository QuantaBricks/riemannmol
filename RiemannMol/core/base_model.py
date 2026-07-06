"""LatentSeqModel -- shared latent sequence-to-sequence architecture
(Perceiver encoder + causal Transformer decoder + a sigma-conditioned
Gaussian latent head) used by both backends. Module/attribute names match
the already-trained checkpoints exactly (word_embed, perceiver_layers,
gaussian_head, dec_layers, ...) so existing weights load without any
remapping.

use_key_padding_mask: the atom backend's encoder masks out PAD positions in
the Perceiver cross-attention (correct -- otherwise a molecule's encoding
depends on how much padding happens to be in its batch); the fragment
backend's checkpoint was trained *without* that mask. Kept as an explicit
constructor flag rather than silently unifying the two, since fragment's
already-trained weights were never exposed to the masked computation --
changing it now would run them through an operation they never saw during
training.
"""
import math
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from .layers import DecoderLayer, LatentSelfAttnLayer, PerceiverCrossAttnLayer
from .tokenizer import SmilesTokenizer

LOG_2PI = math.log(2 * math.pi)


class ConditionalGaussianHead(nn.Module):
    """Sigma-conditioned Gaussian posterior (A-MIM training scheme).

    Training: logvar ~ Uniform(min_logvar, max_logvar) per sample.
    Inference: logvar fixed at min_logvar, z = z_mean (no noise).
    """

    def __init__(self, hidden_size: int, min_logvar: float = -6.0, max_logvar: float = 0.0,
                 map_var_to_hiddens: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.map_var_to_hiddens = map_var_to_hiddens

        if map_var_to_hiddens:
            self.logvar_hiddens_to_mean = nn.Sequential(
                nn.Linear(hidden_size + 1, hidden_size + 1),
                nn.ReLU(),
                nn.Linear(hidden_size + 1, hidden_size),
            )
        else:
            self.hiddens_to_mean = nn.Linear(hidden_size, hidden_size)

    def forward(self, hiddens: torch.Tensor):
        """hiddens: (B, 1, H). Returns z, z_mean, logvar, z_log_prob (all (B, 1, H) except
        logvar which is broadcastable)."""
        B = hiddens.shape[0]
        device, dtype = hiddens.device, hiddens.dtype

        if self.training:
            logvar = (torch.rand(B, 1, 1, device=device, dtype=dtype)
                      * (self.max_logvar - self.min_logvar) + self.min_logvar)
        else:
            logvar = torch.full((B, 1, 1), self.min_logvar, device=device, dtype=dtype)

        if self.map_var_to_hiddens:
            z_mean = self.logvar_hiddens_to_mean(torch.cat([hiddens, logvar.expand(-1, 1, 1)], dim=-1))
        else:
            z_mean = self.hiddens_to_mean(hiddens)

        logvar = logvar.expand_as(z_mean)

        if self.training:
            eps = torch.randn_like(z_mean)
            z = (0.5 * logvar).exp() * eps + z_mean
        else:
            z = z_mean

        z_log_prob = -0.5 * (LOG_2PI + logvar + (z - z_mean).pow(2) / logvar.exp())
        return z, z_mean, logvar, z_log_prob


class LatentSeqModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 384,
        num_heads: int = 6,
        num_layers: int = 4,
        max_len: int = 128,
        min_logvar: float = -6.0,
        max_logvar: float = 0.0,
        map_var_to_hiddens: bool = True,
        use_key_padding_mask: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_len = max_len
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.map_var_to_hiddens = map_var_to_hiddens
        self.use_key_padding_mask = use_key_padding_mask

        H, NH, NL = hidden_size, num_heads, num_layers

        self.word_embed = nn.Embedding(vocab_size, H)
        self.enc_pos_embed = nn.Embedding(max_len, H)
        self.dec_pos_embed = nn.Embedding(max_len, H)

        self.init_hidden = nn.Parameter(torch.zeros(1, H))
        self.perceiver_layers = nn.ModuleList([PerceiverCrossAttnLayer(H, NH) for _ in range(NL)])
        self.latent_self_layers = nn.ModuleList([LatentSelfAttnLayer(H, NH) for _ in range(NL)])
        self.enc_final_norm = nn.LayerNorm(H, eps=1e-5)

        self.gaussian_head = ConditionalGaussianHead(H, min_logvar, max_logvar, map_var_to_hiddens)

        self.dec_layers = nn.ModuleList([DecoderLayer(H, NH) for _ in range(NL)])
        self.dec_final_norm = nn.LayerNorm(H, eps=1e-5)
        self.lm_head = nn.Linear(H, vocab_size)

    def _encoder_hiddens(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.shape
        if T > self.max_len:
            token_ids = token_ids[:, : self.max_len]
            T = self.max_len
        pos = torch.arange(T, device=token_ids.device).unsqueeze(0)
        context = self.word_embed(token_ids) + self.enc_pos_embed(pos)
        key_padding_mask = (token_ids == SmilesTokenizer.PAD_ID)[:, None, None, :] if self.use_key_padding_mask else None
        latent = self.init_hidden.unsqueeze(0).expand(B, -1, -1)
        for pa, sa in zip(self.perceiver_layers, self.latent_self_layers):
            residual = latent
            latent = pa(latent, context, key_padding_mask=key_padding_mask)
            latent = sa(latent)
            latent = latent + residual
        return self.enc_final_norm(latent)

    def encode_with_stats(self, token_ids: torch.Tensor):
        """Training-time encode: returns z, z_mean, logvar, z_log_prob."""
        hiddens = self._encoder_hiddens(token_ids)
        return self.gaussian_head(hiddens)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Inference-time encode: returns z_mean (B, 1, H)."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            hiddens = self._encoder_hiddens(token_ids)
            _, z_mean, _, _ = self.gaussian_head(hiddens)
        if was_training:
            self.train()
        return z_mean

    def decode_teacher_forced(self, z: torch.Tensor, dec_input_ids: torch.Tensor) -> torch.Tensor:
        """Parallel teacher-forced decode for training. z: (B,1,H), dec_input_ids: (B,T).
        Returns logits (B, T, vocab_size)."""
        B, T = dec_input_ids.shape
        pos = torch.arange(T, device=dec_input_ids.device).unsqueeze(0)
        x = self.word_embed(dec_input_ids) + self.dec_pos_embed(pos)
        for layer in self.dec_layers:
            x, _ = layer(x, z, kv_cache=None)
        x = self.dec_final_norm(x)
        return self.lm_head(x)

    def decode_greedy(self, z: torch.Tensor, max_len: Optional[int] = None) -> List[List[int]]:
        max_len = min(max_len or self.max_len, self.max_len)
        B = z.shape[0]
        BOS, EOS = SmilesTokenizer.BOS_ID, SmilesTokenizer.EOS_ID
        cur = torch.full((B, 1), BOS, dtype=torch.long, device=z.device)
        kv_caches = [None] * self.num_layers
        results = [[] for _ in range(B)]
        done = [False] * B

        for step in range(max_len):
            pos = torch.tensor([[step]], device=z.device)
            x = self.word_embed(cur) + self.dec_pos_embed(pos)
            new_caches = []
            for i, layer in enumerate(self.dec_layers):
                x, nkv = layer(x, z, kv_caches[i])
                new_caches.append(nkv)
            kv_caches = new_caches
            x = self.dec_final_norm(x)
            tok = self.lm_head(x[:, 0, :]).argmax(dim=-1)
            cur = tok.unsqueeze(1)
            for b in range(B):
                if done[b]:
                    continue
                t = tok[b].item()
                if t == EOS:
                    done[b] = True
                else:
                    results[b].append(t)
            if all(done):
                break
        return results

    def decode_beam(self, z: torch.Tensor, beam_size: int = 3, max_len: Optional[int] = None) -> List[List[int]]:
        max_len = min(max_len or self.max_len, self.max_len)
        B = z.shape[0]
        BOS, EOS = SmilesTokenizer.BOS_ID, SmilesTokenizer.EOS_ID
        NL = self.num_layers
        results = []

        for b in range(B):
            zb = z[b:b + 1]
            beams = [{"score": 0.0, "tokens": [BOS], "kv": [None] * NL}]
            completed = []

            for step in range(max_len):
                if not beams:
                    break
                n = len(beams)
                cur = torch.tensor([[bm["tokens"][-1]] for bm in beams], dtype=torch.long, device=zb.device)
                pos = torch.full((n, 1), step, dtype=torch.long, device=zb.device)
                z_exp = zb.expand(n, -1, -1)

                x = self.word_embed(cur) + self.dec_pos_embed(pos)
                new_kv_per_beam = [[] for _ in range(n)]

                for li, layer in enumerate(self.dec_layers):
                    if beams[0]["kv"][li] is not None:
                        k_cat = torch.cat([bm["kv"][li][0] for bm in beams], dim=0)
                        v_cat = torch.cat([bm["kv"][li][1] for bm in beams], dim=0)
                        kv_in = (k_cat, v_cat)
                    else:
                        kv_in = None
                    x, nkv = layer(x, z_exp, kv_in)
                    for i, (ks, vs) in enumerate(zip(nkv[0].chunk(n, 0), nkv[1].chunk(n, 0))):
                        new_kv_per_beam[i].append((ks, vs))

                x = self.dec_final_norm(x)
                log_probs = torch.log_softmax(self.lm_head(x[:, 0, :]), dim=-1)

                all_cands = []
                for i, bm in enumerate(beams):
                    for lp, tid in zip(*log_probs[i].topk(beam_size)):
                        all_cands.append({
                            "score": bm["score"] + lp.item(),
                            "tokens": bm["tokens"] + [tid.item()],
                            "kv": new_kv_per_beam[i],
                            "eos": tid.item() == EOS,
                        })

                all_cands.sort(key=lambda c: c["score"] / max(len(c["tokens"]) - 1, 1), reverse=True)

                beams = []
                for cand in all_cands:
                    if cand["eos"]:
                        completed.append((cand["score"], cand["tokens"][1:-1]))
                    elif len(beams) < beam_size:
                        beams.append({"score": cand["score"], "tokens": cand["tokens"], "kv": cand["kv"]})

            for bm in beams:
                completed.append((bm["score"], bm["tokens"][1:]))

            if not completed:
                results.append([])
                continue
            _, best = max(completed, key=lambda c: c[0] / max(len(c[1]), 1))
            results.append(best)

        return results

    def sample(self, z: torch.Tensor, n: int, std: float) -> torch.Tensor:
        noise = torch.randn(n, 1, self.hidden_size, device=z.device) * std
        return z.expand(n, -1, -1) + noise

    def save_checkpoint(self, path: str | Path):
        torch.save({
            "config": {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "max_len": self.max_len,
                "min_logvar": self.min_logvar,
                "max_logvar": self.max_logvar,
                "map_var_to_hiddens": self.map_var_to_hiddens,
                "use_key_padding_mask": self.use_key_padding_mask,
            },
            "state_dict": self.state_dict(),
        }, path)


def load_trained(path: str | Path, device: str | torch.device = "cpu",
                  model_cls=LatentSeqModel, default_use_key_padding_mask: bool = True) -> LatentSeqModel:
    """Loads a checkpoint saved by save_checkpoint(). Old checkpoints
    (saved before the use_key_padding_mask flag existed) have no such key
    in their config -- defaults to `default_use_key_padding_mask`. Pass
    model_cls to load into a subclass (e.g. the fragment backend's model),
    and default_use_key_padding_mask=False for checkpoints known to have
    been trained unmasked."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = dict(ckpt["config"])
    config.setdefault("use_key_padding_mask", default_use_key_padding_mask)
    model = model_cls(**config)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model
