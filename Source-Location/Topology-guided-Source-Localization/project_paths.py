from pathlib import Path


CT_ISL_DIR = Path(__file__).resolve().parent
SOURCE_LOCATION_DIR = CT_ISL_DIR.parent
VFSL_ROOT = SOURCE_LOCATION_DIR.parent

DATASET_DIFFUSION_ROOT = VFSL_ROOT / "Dataset-diffusion"
CLASSICAL_DIFFUSION_ROOT = (
    DATASET_DIFFUSION_ROOT / "Classical-diffusion" / "simulated-diffusion"
)
LLM_DIFFUSION_ROOT = DATASET_DIFFUSION_ROOT / "LLM-diffusion"
PROFILE_EMBEDDING_PATH = LLM_DIFFUSION_ROOT / "profile_fields_embedding_Qwen_8B.npy"
POKEC_LLM_TOPOLOGY_EDGE_PATH = (
    VFSL_ROOT
    / "LLM-assisted-Dataset-Construction"
    / "Pokec"
    / "data"
    / "processed"
    / "subgraph_multi_community_N20000"
    / "edges.csv"
)

INFERRED_GRAPH_ROOT = SOURCE_LOCATION_DIR / "input_data(inferred_topology,  context)"
CLASSICAL_INFERRED_GRAPH_ROOT = INFERRED_GRAPH_ROOT / "classical"
LLM_INFERRED_GRAPH_ROOT = INFERRED_GRAPH_ROOT / "context-llm"

OUTPUT_ROOT = CT_ISL_DIR / "outputs"
MODEL_SAVE_ROOT = CT_ISL_DIR / "model_save"


def ensure_within_vfsl(path):
    resolved = Path(path).resolve()
    root = VFSL_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Path must stay inside VF-SL: {path}")
    return resolved


def resolve_vfsl_path(path, base=None):
    p = Path(path)
    if p.is_absolute():
        return ensure_within_vfsl(p)

    roots = []
    if base is not None:
        roots.append(Path(base))
    roots.extend([CT_ISL_DIR, VFSL_ROOT])

    for root in roots:
        candidate = root / p
        if candidate.exists():
            return ensure_within_vfsl(candidate)

    return ensure_within_vfsl((base or VFSL_ROOT) / p)


def rel_to_vfsl(path):
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(VFSL_ROOT.resolve()))
    except ValueError:
        return str(path)


def make_output_dir(*parts):
    path = ensure_within_vfsl(OUTPUT_ROOT.joinpath(*parts))
    path.mkdir(parents=True, exist_ok=True)
    return path
