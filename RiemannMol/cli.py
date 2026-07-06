"""RiemannMol command-line interface: one entry point for both backends.

Shared verbs (`encode`, `decode`, `census`) take `--backend atom|fragment`
(default atom). Backend-specific verbs are only registered for the backend
they apply to: `interpolate`/`arithmetic`/`extend`/`project`/`distance` are
atom-only (they operate on plain-SMILES z arithmetic); `grow`/`edit` are
fragment-only (they operate on SAFE-fragment-anchored generation).

UX target is an ordinary user, not a researcher: sensible defaults for
every parameter, a human-readable table as the default output, `--json`
as the opt-in machine-readable form.
"""
import argparse
import json
import sys


def _print_table(rows, fields):
    if not rows:
        print("(no results)")
        return
    widths = {f: max(len(f), max(len(str(r.get(f, ""))) for r in rows)) for f in fields}
    header = "  ".join(f.ljust(widths[f]) for f in fields)
    print(header)
    print("  ".join("-" * widths[f] for f in fields))
    for r in rows:
        print("  ".join(str(r.get(f, "")).ljust(widths[f]) for f in fields))


def _load_backend(name: str):
    if name == "atom":
        from .atom import load_model
        return load_model()
    elif name == "fragment":
        from .fragment import load_model
        return load_model()
    raise ValueError(f"unknown backend {name!r}, expected 'atom' or 'fragment'")


def cmd_encode(args):
    model = _load_backend(args.backend)
    z = model.encode(args.smiles)
    if args.json:
        print(json.dumps([z[i, 0].tolist() for i in range(z.shape[0])]))
    else:
        for smi, zi in zip(args.smiles, z):
            print(f"{smi}: dim={zi.shape[-1]} norm={float(zi.norm()):.3f}")


def cmd_decode(args):
    model = _load_backend(args.backend)
    if args.smiles:
        z = model.encode([args.smiles])
    elif args.z_file:
        with open(args.z_file) as f:
            vec = json.load(f)
        import torch
        z = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    else:
        print("must pass --smiles (round-trip demo) or --z-file", file=sys.stderr)
        sys.exit(1)
    smis = model.decode(z, mode=args.mode, n_samples=args.n_samples, temperature=args.temperature)
    if args.json:
        print(json.dumps(smis))
    else:
        for smi in smis:
            print(smi)


def cmd_interpolate(args):
    from .atom import interpolate
    results = interpolate(args.smiles_a, args.smiles_b, n_steps=args.steps, mode=args.mode,
                           n_samples=args.n_samples, temperature=args.temperature)
    if args.json:
        print(json.dumps(results))
    else:
        _print_table(results, ["t", "top_smiles", "top_frac", "n_distinct"])


def cmd_arithmetic(args):
    from .atom import arithmetic
    result = arithmetic(args.smiles_a, args.smiles_b, op=args.op,
                         n_samples=args.n_samples, temperature=args.temperature)
    print(json.dumps(result) if args.json else result)


def cmd_extend(args):
    from .atom import extend
    results = extend(args.prefix, n_samples=args.n_samples, temperature=args.temperature)
    if args.json:
        print(json.dumps(results))
    else:
        _print_table(results, ["smiles", "count", "frac"])


def cmd_project(args):
    from .atom import project
    vec = project(args.smiles, head=args.head)
    print(json.dumps(vec.tolist()) if args.json else vec.tolist())


def cmd_distance(args):
    from .atom import distance
    d = distance(args.smiles_a, args.smiles_b, head=args.head)
    print(json.dumps({"distance": d}) if args.json else f"distance ({args.head}): {d:.4f}")


def cmd_census(args):
    if args.backend == "atom":
        from .atom import census
    else:
        from .fragment import census
    result = census(n_points=args.n_points, n_samples_per_point=args.n_samples)
    result.pop("per_point_log", None)
    print(json.dumps(result) if args.json else result)


def cmd_generate(args):
    from .atom import generate
    results = generate(args.smiles, n_samples=args.n_samples, std=args.std,
                        min_similarity=args.min_similarity, max_mw=args.max_mw)
    if args.json:
        print(json.dumps(results))
    else:
        _print_table(results, ["smiles", "similarity", "mw"])


def cmd_optimize(args):
    from .atom import optimize
    history = optimize(args.smiles, property=args.property, n_steps=args.steps,
                        popsize=args.popsize, sigma0_fraction=args.sigma0_fraction)
    if args.json:
        print(json.dumps(history))
    else:
        _print_table(history, ["generation", "smiles", "score"])


def cmd_grow(args):
    from .fragment import grow_scaffold
    results = grow_scaffold(args.anchor, n_samples=args.n_samples, temperature=args.temperature)
    if args.json:
        print(json.dumps(results))
    else:
        _print_table(results, ["smiles", "count", "frac"])


def cmd_edit(args):
    from .fragment import edit_locally
    results = edit_locally(args.prefix, n_samples=args.n_samples, temperature=args.temperature)
    if args.json:
        print(json.dumps(results))
    else:
        _print_table(results, ["smiles", "count", "frac"])


def main():
    p = argparse.ArgumentParser(prog="riemannmol", description="RiemannMol: molecule latent-space generation and analysis")
    p.add_argument("--json", action="store_true", help="machine-readable JSON output instead of a table")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("encode", help="encode one or more SMILES into latent z")
    sp.add_argument("smiles", nargs="+")
    sp.add_argument("--backend", default="atom", choices=["atom", "fragment"])
    sp.set_defaults(func=cmd_encode)

    sp = sub.add_parser("decode", help="decode a latent z (or round-trip a SMILES) back to molecules")
    sp.add_argument("--smiles", help="round-trip demo: encode then decode this SMILES")
    sp.add_argument("--z-file", help="path to a JSON file containing a saved z vector")
    sp.add_argument("--backend", default="atom", choices=["atom", "fragment"])
    sp.add_argument("--mode", default="greedy", choices=["greedy", "beam", "sample"])
    sp.add_argument("--n-samples", type=int, default=30)
    sp.add_argument("--temperature", type=float, default=1.0)
    sp.set_defaults(func=cmd_decode)

    sp = sub.add_parser("interpolate", help="[atom] walk the latent path between two molecules")
    sp.add_argument("smiles_a")
    sp.add_argument("smiles_b")
    sp.add_argument("--steps", type=int, default=9)
    sp.add_argument("--mode", default="sample", choices=["greedy", "sample"])
    sp.add_argument("--n-samples", type=int, default=100)
    sp.add_argument("--temperature", type=float, default=1.0)
    sp.set_defaults(func=cmd_interpolate)

    sp = sub.add_parser("arithmetic", help="[atom] add or subtract two molecules' z and decode the result")
    sp.add_argument("smiles_a")
    sp.add_argument("smiles_b")
    sp.add_argument("--op", default="+", choices=["+", "-"])
    sp.add_argument("--n-samples", type=int, default=100)
    sp.add_argument("--temperature", type=float, default=1.0)
    sp.set_defaults(func=cmd_arithmetic)

    sp = sub.add_parser("extend", help="[atom] complete an incomplete (open-valence) SMILES fragment")
    sp.add_argument("prefix")
    sp.add_argument("--n-samples", type=int, default=30)
    sp.add_argument("--temperature", type=float, default=1.2)
    sp.set_defaults(func=cmd_extend)

    sp = sub.add_parser("project", help="[atom] compare how similar/potent-like a molecule looks, via a metric head")
    sp.add_argument("smiles")
    sp.add_argument("--head", default="tanimoto", help="tanimoto | potency_egfr | <path to a custom head checkpoint>")
    sp.set_defaults(func=cmd_project)

    sp = sub.add_parser("distance", help="[atom] compare two molecules through a metric head")
    sp.add_argument("smiles_a")
    sp.add_argument("smiles_b")
    sp.add_argument("--head", default="tanimoto")
    sp.set_defaults(func=cmd_distance)

    sp = sub.add_parser("census", help="survey how confidently the decoder answers across many random latent points")
    sp.add_argument("--backend", default="atom", choices=["atom", "fragment"])
    sp.add_argument("--n-points", type=int, default=1000)
    sp.add_argument("--n-samples", type=int, default=100)
    sp.set_defaults(func=cmd_census)

    sp = sub.add_parser("generate", help="[atom] generate analogs of a seed molecule (noise-based sampling)")
    sp.add_argument("smiles", help="seed molecule")
    sp.add_argument("--n-samples", type=int, default=20)
    sp.add_argument("--std", type=float, default=1.0,
                     help="noise scale; <0.6 barely changes anything, >=1.0 jumps to mostly-unrelated molecules")
    sp.add_argument("--min-similarity", type=float, default=None)
    sp.add_argument("--max-mw", type=float, default=None)
    sp.set_defaults(func=cmd_generate)

    sp = sub.add_parser("optimize", help="[atom] property-guided molecule generation (CMA-ES in latent space)")
    sp.add_argument("smiles", help="seed molecule to start the search from")
    sp.add_argument("--property", default="qed", choices=["qed", "logp"],
                     help="qed = drug-likeness (safe default); logp is documented to be exploitable "
                          "via long carbon chains -- use a custom scoring_fn via the Python API instead "
                          "of the CLI if you need a property-safe objective")
    sp.add_argument("--steps", type=int, default=30, help="number of CMA-ES generations")
    sp.add_argument("--popsize", type=int, default=20)
    sp.add_argument("--sigma0-fraction", type=float, default=0.2,
                     help="initial CMA-ES step size, as a fraction of the seed's z-norm")
    sp.set_defaults(func=cmd_optimize)

    sp = sub.add_parser("grow", help="[fragment] grow new structure off a fixed anchor fragment")
    sp.add_argument("anchor", help="a SAFE fragment string, may carry dangling ring-closure digits as attachment points")
    sp.add_argument("--n-samples", type=int, default=20)
    sp.add_argument("--temperature", type=float, default=1.0)
    sp.set_defaults(func=cmd_grow)

    sp = sub.add_parser("edit", help="[fragment] fix a prefix of fragments, freely regenerate the rest")
    sp.add_argument("prefix", help="a SAFE fragment prefix to keep fixed")
    sp.add_argument("--n-samples", type=int, default=30)
    sp.add_argument("--temperature", type=float, default=1.5)
    sp.set_defaults(func=cmd_edit)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
