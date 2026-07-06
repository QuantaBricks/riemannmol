"""SAFE-style molecule encoding, implemented directly on top of RDKit's
BRICS decomposition -- no dependency on the `safe-mol` package (which
pulls in an unrelated ~50-package HF/transformers stack).

The trick SAFE representations rely on: RDKit's SMILES parser treats a ring
closure digit that appears in two different dot-separated fragments as a
bond between those two fragments (e.g. "C1.C1" parses to ethane, "CC"). So a
molecule can be "fragmented" into a dot-joined string of BRICS pieces, with
each cut bond represented as a shared ring-closure digit instead of a pair
of `[*]` dummy atoms -- the result is still a single, valid, RDKit-parseable
SMILES string, and needs no new tokenizer/vocab machinery beyond what the
existing SMILES tokenizer regex (digits, '%NN', '.') already covers.
"""

import re
from typing import List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import BRICS

# Reuses the same alphabet as core.tokenizer's SMILES_REGEX -- bracket atoms
# are captured whole, so digits inside e.g. "[13C]" are never mistaken for
# ring-closure numbers.
_TOKEN_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)
_RING_DIGIT_RE = re.compile(r"^(?:\d|%\d\d)$")

# Isotope label offset for temporary dummy atoms marking a cut bond. Chosen
# far above any isotope that occurs in real training data so we can
# unambiguously find-and-replace them.
_DUMMY_BASE = 10_000


def _ring_token(n: int) -> str:
    return str(n) if n < 10 else f"%{n:02d}"


def _used_ring_numbers(smi: str) -> List[int]:
    used = []
    for tok in _TOKEN_REGEX.findall(smi):
        if _RING_DIGIT_RE.match(tok):
            used.append(int(tok[1:]) if tok.startswith("%") else int(tok))
    return used


def smiles_to_safe(smiles: str, max_fragments: int = 8) -> Optional[str]:
    """Convert a SMILES string into a SAFE-style dot-joined fragment string.

    Returns None if the input doesn't parse. Molecules with no BRICS-breakable
    bonds (or where breaking would exceed `max_fragments`) are returned
    unchanged (still valid "SAFE", just a single fragment).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    brics_bonds = list(BRICS.FindBRICSBonds(mol))
    if not brics_bonds:
        return Chem.MolToSmiles(mol)

    if len(brics_bonds) + 1 > max_fragments:
        brics_bonds = brics_bonds[: max_fragments - 1]

    bond_idxs = []
    dummy_labels = []
    for i, ((a1, a2), _) in enumerate(brics_bonds):
        bond = mol.GetBondBetweenAtoms(a1, a2)
        if bond is None:
            continue
        # BRICS occasionally proposes cutting a stereo-relevant double bond
        # (e.g. an enol/vinyl rule-7 cut). Splitting the bond that a C=C
        # cis/trans descriptor is anchored to loses the E/Z relationship
        # across the two resulting fragments, so restrict cuts to plain
        # single bonds.
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        bond_idxs.append(bond.GetIdx())
        label = _DUMMY_BASE + len(dummy_labels)
        dummy_labels.append((label, label))

    if not bond_idxs:
        return Chem.MolToSmiles(mol)

    frag_mol = Chem.FragmentOnBonds(mol, bond_idxs, addDummies=True, dummyLabels=dummy_labels)
    frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)

    frag_smis = []
    for frag in frags:
        try:
            Chem.SanitizeMol(frag)
        except Exception:
            return None
        # Root the SMILES at a real (non-dummy) atom so a dummy is never the
        # first token. A leading dummy would need its ring-closure digit
        # spliced in after the fact, which silently corrupts stereo parity
        # for a leading chiral neighbor -- rooting elsewhere avoids the
        # rewrite entirely.
        root = next((a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() != 0), 0)
        frag_smis.append(Chem.MolToSmiles(frag, rootedAtAtom=root))

    # Assign each cut bond's dummy-label pair a small ring-closure number that
    # doesn't collide with any ring-closure digit already used by any
    # fragment's own (real) rings.
    used = set()
    for fs in frag_smis:
        used.update(_used_ring_numbers(fs))

    label_to_num = {}
    next_num = 1
    for i in range(len(bond_idxs)):
        label = _DUMMY_BASE + i
        while next_num in used:
            next_num += 1
        label_to_num[label] = next_num
        used.add(next_num)
        next_num += 1

    out_frags = []
    for fs in frag_smis:
        # A dummy atom at the very start of the fragment has no preceding
        # atom to attach its ring-closure digit to -- move the digit to
        # just after the following atom token instead.
        lead = re.match(r"^([=#\-/\\:~]?)\[(\d+)\*\]([=#\-/\\:~]?)(.*)$", fs, re.DOTALL)
        if lead:
            bond_sym = lead.group(1) or lead.group(3)
            label = int(lead.group(2))
            rest = lead.group(4)
            atom_match = _TOKEN_REGEX.match(rest)
            atom_tok = atom_match.group(0) if atom_match else ""
            fs = atom_tok + bond_sym + _ring_token(label_to_num[label]) + rest[len(atom_tok):]

        def _sub(m: "re.Match") -> str:
            bond_sym = m.group(1) or ""
            label = int(m.group(2))
            return bond_sym + _ring_token(label_to_num[label])

        # Parenthesised branch containing only the dummy atom, e.g. "(=[10000*])".
        fs = re.sub(r"\(([=#\-/\\:~]?)\[(\d+)\*\]\)", _sub, fs)
        # Bare dummy atom (chain-terminal or otherwise unwrapped).
        fs = re.sub(r"([=#\-/\\:~]?)\[(\d+)\*\]", _sub, fs)
        out_frags.append(fs)

    return ".".join(out_frags)


def safe_to_smiles(safe: str) -> Optional[str]:
    """SAFE strings are valid SMILES by construction -- this just canonicalizes."""
    mol = Chem.MolFromSmiles(safe)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def roundtrip_ok(smiles: str) -> Tuple[bool, Optional[str]]:
    """Returns (ok, safe_string). ok=True iff safe_to_smiles(safe) canonically
    matches the original input."""
    ref = Chem.MolFromSmiles(smiles)
    if ref is None:
        return False, None
    ref_canon = Chem.MolToSmiles(ref)

    safe = smiles_to_safe(smiles)
    if safe is None:
        return False, None

    out = safe_to_smiles(safe)
    return (out == ref_canon), safe
