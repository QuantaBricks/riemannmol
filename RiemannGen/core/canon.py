"""RDKit canonicalization and shared molecule-grid rendering."""
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def canon(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None


def render_molecule_grid(smis_or_mols, legends, out_path, n_cols=4, sub_size=420, padding=0.03):
    """Standard rendering style: CoordGen layout (proportional rings,
    avoids free-standing substituents looking oversized next to fused
    ring systems), black-and-white atoms (print-friendly), large legend
    font, tight padding.

    padding: fraction of each panel reserved as whitespace margin around the
    molecule drawing (RDKit's MolDrawOptions.padding). Higher = more visual
    separation between adjacent molecules (each one is drawn smaller,
    inset further from its panel's edges); 0.03 is tight/default, try
    ~0.1-0.15 for a visibly roomier grid."""
    from rdkit.Chem import rdCoordGen
    from rdkit.Chem.Draw import rdMolDraw2D

    mols = []
    for item in smis_or_mols:
        m = item if isinstance(item, Chem.Mol) else Chem.MolFromSmiles(item)
        rdCoordGen.AddCoords(m)
        mols.append(m)

    n_rows = (len(mols) + n_cols - 1) // n_cols
    d2d = rdMolDraw2D.MolDraw2DCairo(sub_size * n_cols, sub_size * n_rows, sub_size, sub_size)
    opts = d2d.drawOptions()
    opts.legendFontSize = 24
    opts.legendFraction = 0.14
    opts.padding = padding
    opts.bondLineWidth = 3.0
    opts.useBWAtomPalette()
    opts.clearBackground = True
    d2d.DrawMolecules(mols, legends=legends)
    d2d.FinishDrawing()
    with open(out_path, "wb") as f:
        f.write(d2d.GetDrawingText())
