"""Shared SMILES/SAFE tokenizer, used by both the atom (plain SMILES) and
fragment (SAFE-encoded SMILES) backends -- SAFE strings parse with the same
regex since they're just ordinary SMILES with shared ring-closure digits
standing in for fragment attachment points."""
import re
from pathlib import Path
from typing import List

SMILES_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)


class SmilesTokenizer:
    PAD_ID = 0
    BOS_ID = 2
    EOS_ID = 3

    def __init__(self, vocab_path: str | Path):
        self.vocab: List[str] = []
        with open(vocab_path) as f:
            for line in f:
                self.vocab.append(line.rstrip("\n"))
        self.tok2id = {t: i for i, t in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, smiles: str) -> List[int]:
        tokens = SMILES_REGEX.findall(smiles)
        return [self.tok2id.get(t, 1) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        out = []
        for i in ids:
            if i in (self.BOS_ID, self.PAD_ID):
                continue
            if i == self.EOS_ID:
                break
            if 0 <= i < len(self.vocab):
                out.append(self.vocab[i])
        return "".join(out)
