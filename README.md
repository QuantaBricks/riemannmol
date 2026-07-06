# RiemannGen

Properties-guided molecule generation, with two latent-space backends —
and a built-in way to tell which generated molecules are actually
trustworthy, not just plausible-looking.

## Why

Most latent-space molecule generators will happily decode *any* point you
give them into *some* molecule. But not every point in latent space is
equally trustworthy: points near real training data decode confidently and
reproducibly; points far from it are often ambiguous — sampling the same
point repeatedly can yield several unrelated molecules, or nothing valid at
all. RiemannGen's `census` command surfaces this directly, so you know
which candidates to actually trust.

## Install

```bash
pip install riemanngen
```

Checkpoints are hosted on the Hugging Face Hub
([XeonChem/riemanngen](https://huggingface.co/XeonChem/riemanngen)) and
downloaded automatically on first use — no separate setup step.

## Features

**atom backend** (plain-SMILES latent model):
- `encode` / `decode` — greedy, beam-search, or sampled decoding.
- `optimize` — property-guided generation: CMA-ES search in latent space,
  decoding and scoring every candidate against a real property function
  (QED by default; bring your own scoring function for anything else).
- `interpolate` — walk the latent path between two molecules.
- `arithmetic` — add/subtract two molecules' latent vectors and decode the
  result.
- `extend` — complete an incomplete (open-valence) SMILES fragment.
- `project` / `distance` — compare molecules through a pluggable metric
  head (structural similarity, or a custom property-similarity head you
  train yourself).
- `census` — the confidence-state diagnostic described above: samples many
  points and classifies each as *pure* (confident, single-answer),
  *entangled* (ambiguous, multiple candidates), or *dead/idle* (decoding
  breaks down).

**fragment backend** (SAFE-fragment-encoded model):
- `grow` — grow new structure off a fixed anchor fragment.
- `edit` — fix a prefix of fragments, freely regenerate the rest.

## Usage

```python
from RiemannGen.atom import load_model, optimize

model = load_model()
z = model.encode(["CCO", "c1ccccc1"])
print(model.decode(z))

history = optimize("CC(C)Cc1ccc(cc1)C(C)C(=O)O", property="qed", n_steps=30)
for h in history:
    print(h)
```

Or from the command line:

```bash
riemanngen encode "CCO"
riemanngen optimize "CC(C)Cc1ccc(cc1)C(C)C(=O)O" --property qed --steps 30
riemanngen interpolate "CCO" "c1ccccc1" --steps 5
riemanngen census --n-points 200
riemanngen grow "c1ccccc12" --n-samples 10   # fragment backend
```

Run `riemanngen --help` for the full list of subcommands.

## License

Apache License 2.0.
