"""Fallback checkpoint/vocab download from the HuggingFace Hub mirror
(XeonChem/riemanngen), used when a backend's default local path (which
assumes a full checkout of the research repo, with atom/ and fragment/
sibling directories present) doesn't exist -- e.g. when RiemannGen is
pip-installed standalone, without the rest of the repo."""
from pathlib import Path

HF_REPO = "XeonChem/riemanngen"


def resolve_path(local_path, hf_filename: str) -> str:
    """local_path: a Path that exists only in a full repo checkout.
    hf_filename: the corresponding path in the XeonChem/riemanngen HF repo
    (e.g. "atom/final-atom.pt"). Returns a local path either way, downloading
    (and caching, via huggingface_hub's own cache dir) if needed."""
    local_path = Path(local_path)
    if local_path.exists():
        return str(local_path)
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import LocalEntryNotFoundError
    except ImportError:
        raise ImportError(
            f"{local_path} not found, and huggingface_hub is not installed to fetch it "
            f"from {HF_REPO}. Run `pip install huggingface_hub` (or install RiemannGen "
            f"with the [hf] extra) to enable automatic checkpoint download."
        )
    # Check the HF cache first (no network call) so repeated load_model() calls
    # in the same session/machine don't print a misleading "downloading" every
    # time -- only the very first call actually hits the network.
    try:
        return hf_hub_download(repo_id=HF_REPO, filename=hf_filename, local_files_only=True)
    except LocalEntryNotFoundError:
        pass
    print(f"{hf_filename} not cached -- downloading from {HF_REPO} (one-time) ...")
    return hf_hub_download(repo_id=HF_REPO, filename=hf_filename)
