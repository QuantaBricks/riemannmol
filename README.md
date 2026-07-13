# RiemannMol

Properties-guided molecule generation, with two latent-space backends —
and a built-in way to tell which generated molecules are actually
trustworthy, not just plausible-looking.

## Why

Most latent-space molecule generators will happily decode *any* point you
give them into *some* molecule. But not every point in latent space is
equally trustworthy: points near real training data decode confidently and
reproducibly; points far from it are often ambiguous — sampling the same
point repeatedly can yield several unrelated molecules, or nothing valid at
all. RiemannMol's `census` command surfaces this directly, so you know
which candidates to actually trust.

## Install

```bash
pip install riemannmol
```

Checkpoints are hosted on the Hugging Face Hub
([QuantaBricks/riemannmol](https://huggingface.co/QuantaBricks/riemannmol)) and
downloaded automatically on first use — no separate setup step.

## Features

**atom backend** (plain-SMILES latent model):
- `encode` / `decode` — greedy, beam-search, or sampled decoding.
- `generate` — analogs of a seed molecule via noise-based latent sampling,
  deduplicated and ranked by similarity to the seed.
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

Load the model once and reuse it across calls (avoids reloading weights
every time) — every task function below takes an optional `model=`.

```python
from RiemannMol.atom import load_model

model = load_model()
```

### Generate analogs of a seed molecule (noise-based sampling)

```python
from RiemannMol.atom import generate

# std < ~0.6 barely changes anything; std >= ~1.0 jumps to mostly-unrelated
# molecules -- there is no smooth "slightly different" middle ground.
for r in generate("CC(C)Cc1ccc(cc1)C(C)C(=O)O", n_samples=12, std=1.0, model=model):
    print(f"{r['similarity']:.3f}  {r['smiles']}")
```

`generate` deduplicates, canonicalizes, and ranks by Tanimoto similarity to
the seed for you; pass `min_similarity=`/`max_mw=` to filter results.

### Property-guided generation (CMA-ES optimization)

```python
from RiemannMol.atom import optimize

history = optimize(seed, property="qed", n_steps=30, popsize=20, model=model)
for h in history:  # each entry is a strictly-improving candidate found
    print(h["generation"], h["score"], h["smiles"])
print("best:", history[-1])
```

Bring your own objective instead of the `qed`/`logp` builtins:

```python
from rdkit import Chem
from rdkit.Chem import Descriptors

def my_scorer(smiles: str) -> float:
    m = Chem.MolFromSmiles(smiles)
    return -abs(Descriptors.MolWt(m) - 300)  # e.g. target MW ~300

history = optimize(seed, scoring_fn=my_scorer, n_steps=30, model=model)
```

### Interpolate between two molecules

```python
from RiemannMol.atom import interpolate

results = interpolate("CCO", "c1ccccc1", n_steps=9, mode="sample", model=model)
for r in results:
    print(f"t={r['t']:.2f}  top={r['top_smiles']}  confidence={r['top_frac']:.2f}")
```

`mode="sample"` (not greedy) is what reveals the ambiguous crossing zone
between the two molecules — see `census` below for why that matters.

### Latent arithmetic

```python
from RiemannMol.atom import arithmetic

result = arithmetic("CCO", "c1ccccc1", op="+", model=model)
print(result["top_smiles"], result["top_frac"])
```

### Complete an incomplete SMILES fragment

```python
from RiemannMol.atom import extend

for r in extend("CC(C)Cc1ccc(", n_samples=20, temperature=1.2, model=model):
    print(r["smiles"], r["frac"])
```

### Compare molecules through a metric head

```python
from RiemannMol.atom import distance

print(distance("CCO", "CCN", head="tanimoto"))
```

### Check whether a point is actually trustworthy (confidence census)

```python
from RiemannMol.atom import census

result = census(n_points=200, n_samples_per_point=100, model=model)
print(result["state_counts"])  # e.g. {'pure': 190, 'entangled': 10}
```

### Fragment backend: scaffold growth and local editing

```python
from RiemannMol.fragment import load_model as load_fragment_model, grow_scaffold, edit_locally

fmodel = load_fragment_model()
print(grow_scaffold("c1ccccc12", n_samples=10, model=fmodel))       # grow off an anchor
print(edit_locally("CC(=O)Nc1ccc(", n_samples=10, model=fmodel))    # fix a prefix, regenerate the rest
```

### Command line

```bash
riemannmol encode "CCO"
riemannmol encode "CCO" --head tanimoto  # encode through a metric head's encoder instead of the base z
riemannmol decode --smiles "CCO"
riemannmol generate "CC(C)Cc1ccc(cc1)C(C)C(=O)O" --n-samples 20 --std 1.0
riemannmol optimize "CC(C)Cc1ccc(cc1)C(C)C(=O)O" --property qed --steps 30
riemannmol interpolate "CCO" "c1ccccc1" --steps 5
riemannmol arithmetic "CCO" "c1ccccc1" --op +
riemannmol extend "CC(C)Cc1ccc(" --n-samples 20
riemannmol distance "CCO" "CCN" --head tanimoto
riemannmol census --n-points 200
riemannmol grow "c1ccccc12" --n-samples 10    # fragment backend
riemannmol edit "CC(=O)Nc1ccc(" --n-samples 10  # fragment backend
```

Run `riemannmol --help` (or `riemannmol <command> --help`) for the full
list of subcommands and options.

## License

Apache License 2.0.
