"""Low-level decode primitives shared by every "channel" (interpolate,
arithmetic, extend, census, grow, edit, ...) built on top of a
LatentSeqModel-family model. Generic over any model exposing the standard
word_embed/dec_pos_embed/dec_layers/dec_final_norm/lm_head/num_layers
attributes.

Three distinct decode strategies, each answering a different question about
a point z in latent space:

- sample_decode: unbiased Monte Carlo estimate of the full P(molecule|z)
  distribution via repeated independent ancestral sampling. Fast, noisy per
  candidate, good for scanning many points (a census) or estimating overall
  shape.
- beam_search_all_completed: exact per-sequence log-probability via
  beam search, keeping every completed candidate (not just the argmax
  best). Slower, but exact and complete within the beam width -- good for
  a precise answer at one already-identified point of interest, and for
  cross-validating sample_decode's estimates.
- forced_prefix_then_sample: teacher-forces a given prefix (not sampled --
  the caller decides these tokens unconditionally), then switches to free
  multinomial sampling for the continuation. This is the general mechanism
  behind scaffold extension (feed an incomplete SMILES, sample a valid
  completion) and fragment growth/local editing (feed fixed fragments,
  sample the rest).
"""
from typing import List

import torch
import torch.nn.functional as F


def sample_decode(model, tok, z, n_samples: int, temperature: float, max_len: int = 128) -> List[List[int]]:
    """z: (H,) single latent vector. Returns n_samples independently sampled
    token-id sequences (no BOS, stops at EOS)."""
    BOS, EOS = tok.BOS_ID, tok.EOS_ID
    B = n_samples
    z_rep = z.unsqueeze(0).expand(B, -1, -1).contiguous() if z.dim() == 2 else z.unsqueeze(0).unsqueeze(0).expand(B, -1, -1).contiguous()
    cur = torch.full((B, 1), BOS, dtype=torch.long, device=z.device)
    kv_caches = [None] * model.num_layers
    results = [[] for _ in range(B)]
    done = [False] * B

    for step in range(max_len):
        pos = torch.tensor([[step]], device=z.device)
        x = model.word_embed(cur) + model.dec_pos_embed(pos)
        new_caches = []
        for i, layer in enumerate(model.dec_layers):
            x, nkv = layer(x, z_rep, kv_caches[i])
            new_caches.append(nkv)
        kv_caches = new_caches
        x = model.dec_final_norm(x)
        logits = model.lm_head(x[:, 0, :]) / temperature
        probs = F.softmax(logits, dim=-1)
        tok_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
        cur = tok_ids.unsqueeze(1)
        for b in range(B):
            if done[b]:
                continue
            t = tok_ids[b].item()
            if t == EOS:
                done[b] = True
            else:
                results[b].append(t)
        if all(done):
            break
    return results


def beam_search_all_completed(model, tok, z, beam_size: int, max_len: int = 128):
    """z: (H,) single latent vector. Returns every completed (score,
    token_ids) candidate the beam search finds (not just the single best),
    so the caller can inspect the full set of high-probability modes --
    unlike LatentSeqModel.decode_beam, which discards everything except
    the top-1 candidate per input."""
    BOS, EOS = tok.BOS_ID, tok.EOS_ID
    NL = model.num_layers
    zb = z.unsqueeze(0).unsqueeze(0) if z.dim() == 1 else z
    beams = [{"score": 0.0, "tokens": [BOS], "kv": [None] * NL}]
    completed = []

    for step in range(max_len):
        if not beams:
            break
        n = len(beams)
        cur = torch.tensor([[bm["tokens"][-1]] for bm in beams], dtype=torch.long)
        pos = torch.full((n, 1), step, dtype=torch.long)
        z_exp = zb.expand(n, -1, -1)
        x = model.word_embed(cur) + model.dec_pos_embed(pos)
        new_kv_per_beam = [[] for _ in range(n)]
        for li, layer in enumerate(model.dec_layers):
            if beams[0]["kv"][li] is not None:
                k_cat = torch.cat([bm["kv"][li][0] for bm in beams], dim=0)
                v_cat = torch.cat([bm["kv"][li][1] for bm in beams], dim=0)
                kv_in = (k_cat, v_cat)
            else:
                kv_in = None
            x, nkv = layer(x, z_exp, kv_in)
            for i, (ks, vs) in enumerate(zip(nkv[0].chunk(n, 0), nkv[1].chunk(n, 0))):
                new_kv_per_beam[i].append((ks, vs))
        x = model.dec_final_norm(x)
        log_probs = F.log_softmax(model.lm_head(x[:, 0, :]), dim=-1)

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
    return completed


def forced_prefix_then_sample(model, tok, z, prefix_ids: List[int], n_samples: int,
                               temperature: float, max_len: int = 128) -> List[List[int]]:
    """z: (H,). prefix_ids: token ids to force as decode history (teacher
    forcing -- not sampled, the caller decides these unconditionally),
    then switches to free multinomial sampling for the continuation, per
    independent sample, until EOS or max_len."""
    BOS, EOS = tok.BOS_ID, tok.EOS_ID
    B = n_samples
    z_rep = z.unsqueeze(0).unsqueeze(0).expand(B, -1, -1).contiguous()
    kv_caches = [None] * model.num_layers
    results = [list(prefix_ids) for _ in range(B)]
    done = [False] * B

    forced = [BOS] + list(prefix_ids)
    cur = torch.tensor([forced], dtype=torch.long).expand(B, -1)
    x = None
    for step, tok_col in enumerate(cur.T):
        x = model.word_embed(tok_col.unsqueeze(1)) + model.dec_pos_embed(torch.tensor([[step]]))
        new_caches = []
        for i, layer in enumerate(model.dec_layers):
            x, nkv = layer(x, z_rep, kv_caches[i])
            new_caches.append(nkv)
        kv_caches = new_caches
    step0 = len(forced)

    x_final = model.dec_final_norm(x)
    logits = model.lm_head(x_final[:, 0, :]) / temperature
    probs = F.softmax(logits, dim=-1)
    cur_tok = torch.multinomial(probs, num_samples=1)
    for b in range(B):
        t = cur_tok[b, 0].item()
        if t == EOS:
            done[b] = True
        else:
            results[b].append(t)

    for step in range(step0, max_len):
        pos = torch.tensor([[step]])
        x = model.word_embed(cur_tok) + model.dec_pos_embed(pos)
        new_caches = []
        for i, layer in enumerate(model.dec_layers):
            x, nkv = layer(x, z_rep, kv_caches[i])
            new_caches.append(nkv)
        kv_caches = new_caches
        x = model.dec_final_norm(x)
        logits = model.lm_head(x[:, 0, :]) / temperature
        probs = F.softmax(logits, dim=-1)
        cur_tok = torch.multinomial(probs, num_samples=1)
        for b in range(B):
            if done[b]:
                continue
            t = cur_tok[b, 0].item()
            if t == EOS:
                done[b] = True
            else:
                results[b].append(t)
        if all(done):
            break
    return results
