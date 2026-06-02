import argparse
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

base_dir = Path(__file__).resolve().parent

Pokec_path = "Pokec/data/processed/subgraph_multi_community_N20000"
news_rel_path = "Pokec/data/news/real_recent_news_bbc.jsonl"

LLM_type = "Qwen_8B"
result = "pokec_bbc_llm_diffusion_2w"
MODEL_PATH_ENV = "VFSL_QWEN_MODEL_PATH"


@dataclass
class Config:
    nodes_path: Path = base_dir / Pokec_path / "nodes.csv"
    edges_path: Path = base_dir / Pokec_path / "edges.csv"
    news_path: Path = base_dir / news_rel_path

    # Override with --topic to simulate a different BBC topic.
    target_topic: str = "business"
    news_start: int = 0
    news_end: int = 300
    R: int = 1

    output_root: Path = base_dir / Pokec_path / result
    output_path: Path = base_dir / Pokec_path / result / f"diffusion_results_bbc_{target_topic}_Qwen_8B.jsonl"
    cache_dir: Path = base_dir / Pokec_path / result / f"diffusion_chunk_cache_bbc_{target_topic}_Qwen_8B"
    source_candidate_cache_path: Path = (
        base_dir / Pokec_path / "source_candidate_cache" / f"source_candidates_bbc_{target_topic}_Qwen_8B_top10.json"
    )

    if LLM_type == "Qwen_8B":
        LOCAL_QWEN_PATH: Optional[str] = None
        API_URL: str = "http://127.0.0.1:8002/v1/chat/completions"
        MODEL_NAME: str = "qwen3-8b"

    if LLM_type == "Qwen_4B":
        LOCAL_QWEN_PATH: Optional[str] = None
        API_URL: str = "http://127.0.0.1:8000/v1/chat/completions"
        MODEL_NAME: str = "qwen3-4b"

    # Draw seed_fraction * node_count seeds from the topic source-candidate pool.
    source_top_fraction: float = 0.10
    seed_fraction: float = 0.0025
    min_final_infected: int = 120
    max_final_infected: int = 300
    max_steps: int = 50

    TOP_M: int = 100
    LLM_CHUNK_SIZE: int = 100
    infection_threshold: float = 0.15
    random_seed: int = 42

    w_match: float = 0.33
    w_simI: float = 0.33
    w_soc: float = 0.34

    embed_device: str = "cpu"
    embed_batch_size: int = 8
    embed_max_length_persona: int = 256
    embed_max_length_news: int = 384
    reuse_base_profile_embeddings: bool = True
    base_profile_embedding_dir: Path = base_dir / "Pokec/data/processed/subgraph_multi_community_N20000"
    force_recompute_profile_embeddings: bool = False

    llm_temperature: float = 0.1
    llm_top_p: float = 0.8
    llm_top_k: int = 20
    llm_presence_penalty: float = 0.0
    llm_max_tokens: int = 6000

    max_persona_chars_in_prompt: int = 180
    max_news_chars: int = 1600

    save_every_news: int = 1
    force_recompute_source_candidates: bool = False


CFG = Config()
CHUNK_CACHE_VERSION = "threshold_no_budget_v1"


def safe_tag(text: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip().lower())
    return tag.strip("_") or "topic"


def configure_runtime_paths() -> None:
    topic_tag = safe_tag(CFG.target_topic)
    range_tag = f"{CFG.news_start:03d}_{CFG.news_end:03d}"
    CFG.output_root = base_dir / Pokec_path / result
    CFG.output_path = CFG.output_root / f"diffusion_results_bbc_{topic_tag}_{range_tag}_{LLM_type}.jsonl"
    CFG.cache_dir = CFG.output_root / f"diffusion_chunk_cache_bbc_{topic_tag}_{range_tag}_{LLM_type}"
    CFG.source_candidate_cache_path = (
        base_dir / Pokec_path / "source_candidate_cache" /
        f"source_candidates_bbc_{topic_tag}_{LLM_type}_top{int(CFG.source_top_fraction * 100):02d}.json"
    )


def resolve_model_path(cli_model_path: Optional[str]) -> str:
    model_path = cli_model_path or os.environ.get(MODEL_PATH_ENV)
    if not model_path:
        raise ValueError(
            f"Qwen model path is required. Set --model-path or the {MODEL_PATH_ENV} environment variable."
        )
    return model_path

def ensure_parent_dir(path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_text(x, max_chars: Optional[int] = None) -> str:
    if x is None:
        return ""
    if pd.isna(x):
        return ""
    s = " ".join(str(x).strip().split())
    if s.lower() == "null":
        return ""
    if max_chars is not None:
        s = s[:max_chars]
    return s


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norm


def chunk_list(xs: List, chunk_size: int) -> List[List]:
    return [xs[i:i + chunk_size] for i in range(0, len(xs), chunk_size)]

def get_chunk_cache_path(news_idx: int, repeat_id: int, step_t: int, chunk_idx: int) -> Path:
    ensure_dir(CFG.cache_dir)
    fname = f"news{news_idx:03d}_rep{repeat_id:03d}_step{step_t:03d}_chunk{chunk_idx:03d}.json"
    return Path(CFG.cache_dir) / fname


def load_chunk_cache(news_idx: int, repeat_id: int, step_t: int, chunk_idx: int) -> Optional[dict]:
    path = get_chunk_cache_path(news_idx, repeat_id, step_t, chunk_idx)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_chunk_cache(
    news_idx: int,
    repeat_id: int,
    step_t: int,
    chunk_idx: int,
    payload: dict,
) -> None:
    path = get_chunk_cache_path(news_idx, repeat_id, step_t, chunk_idx)
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

class QwenEncoder:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        dtype = torch.float16 if "cuda" in device else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=dtype,
        )
        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 8,
        max_length: int = 256,
        progress_desc: Optional[str] = None,
    ) -> np.ndarray:
        all_embs = []

        total_batches = (len(texts) + batch_size - 1) // batch_size
        batch_starts = range(0, len(texts), batch_size)
        if progress_desc and tqdm is not None:
            batch_iter = tqdm(
                batch_starts,
                total=total_batches,
                desc=progress_desc,
                unit="batch",
            )
        else:
            batch_iter = batch_starts

        for batch_idx, start in enumerate(batch_iter, start=1):
            batch = texts[start:start + batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )

            hidden = outputs.hidden_states[-1]
            mask = inputs["attention_mask"].unsqueeze(-1)

            masked_hidden = hidden * mask
            summed = masked_hidden.sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            mean_pooled = summed / counts

            embs = mean_pooled.detach().cpu().float().numpy()
            all_embs.append(embs)

            if progress_desc and tqdm is None:
                print(
                    f"{progress_desc}: batch {batch_idx}/{total_batches}",
                    end="\r",
                    flush=True,
                )

        if progress_desc and tqdm is None and total_batches > 0:
            print()

        all_embs = np.concatenate(all_embs, axis=0)
        all_embs = l2_normalize(all_embs)
        return all_embs

POKEC_PERSONA_FIELDS = [
    ("gender", "gender"),
    ("age_bucket", "age group"),
    ("region", "region"),
    ("spoken_languages", "spoken languages"),
    ("hobbies", "hobbies"),
    ("I_like_music", "music interests"),
    ("fun", "fun interests"),
    ("science_technologies", "science and technology interests"),
    ("computers_internet", "computer and internet interests"),
    ("education", "education interests"),
    ("sport", "sport interests"),
    ("movies", "movie interests"),
    ("travelling", "travel interests"),
    ("activity_proxy", "activity level"),
    ("completion_percentage", "profile completion"),
]


def build_pokec_profile_text(row) -> str:
    parts = []
    for col, label in POKEC_PERSONA_FIELDS:
        if not hasattr(row, col):
            continue
        value = getattr(row, col)
        value = normalize_text(value)
        if value:
            parts.append(f"{label}: {value}")
    if not parts:
        return "This user has limited available profile information."
    return "; ".join(parts)


def load_nodes(nodes_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(nodes_path)
    if "u_id" not in df.columns or "orig_id" not in df.columns:
        raise ValueError("Pokec nodes.csv must contain at least: u_id, orig_id")
    df["u_id"] = df["u_id"].astype(int)
    df["orig_id"] = df["orig_id"].astype(str)
    df = df.sort_values("u_id").reset_index(drop=True)

    expected = list(range(len(df)))
    actual = df["u_id"].tolist()
    if actual != expected:
        raise ValueError("This script expects contiguous u_id values from 0 to N-1. Please check nodes.csv.")

    # Build persona text from raw Pokec profile fields for embeddings and prompts.
    df["profile_text_for_llm"] = [
        build_pokec_profile_text(row)
        for row in df.itertuples(index=False)
    ]

    return df


def load_edges(edges_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(edges_path)
    required = {"src_u_id", "dst_u_id"}
    if not required.issubset(set(df.columns)):
        raise ValueError("edges.csv must contain at least: src_u_id, dst_u_id")
    df["src_u_id"] = df["src_u_id"].astype(int)
    df["dst_u_id"] = df["dst_u_id"].astype(int)
    return df


def load_topic_news(news_path: str | Path, target_topic: str) -> List[dict]:
    all_news = []
    topics = set()
    with open(news_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            topic = str(item.get("topic", "")).strip()
            topics.add(topic)
            if topic == target_topic:
                item = dict(item)
                item["_source_line_idx"] = line_idx
                item["_topic_local_idx"] = len(all_news)
                all_news.append(item)

    if not all_news:
        raise ValueError(
            f"No news found for topic={target_topic!r}; available topics={sorted(topics)}"
        )

    return all_news


def select_news_range(topic_news: List[dict]) -> List[dict]:
    start = max(0, int(CFG.news_start))
    end = int(CFG.news_end)
    if end <= 0 or end > len(topic_news):
        end = len(topic_news)
    if start >= end:
        raise ValueError(
            f"Invalid news range: news_start={CFG.news_start}, news_end={CFG.news_end}, "
            f"topic_news_total={len(topic_news)}"
        )
    return topic_news[start:end]


def build_news_text(item: dict, max_chars: int = 1600) -> str:
    topic = normalize_text(item.get("topic", ""), 120)
    content = normalize_text(item.get("content", ""), max_chars)
    parts = []
    if topic:
        parts.append(f"topic: {topic}")
    if content:
        parts.append(f"content: {content}")
    return "\n".join(parts)

def build_graph_helpers(nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
    """Treat all edges as undirected for social proof and parent suggestions."""
    uids = nodes_df["u_id"].tolist()

    undirected_neighbors = {u: set() for u in uids}
    edge_set = set()

    for s, d in edges_df[["src_u_id", "dst_u_id"]].itertuples(index=False):
        s = int(s)
        d = int(d)
        if s not in undirected_neighbors or d not in undirected_neighbors:
            continue

        undirected_neighbors[s].add(d)
        undirected_neighbors[d].add(s)
        edge_set.add((s, d))
        edge_set.add((d, s))

    out_neighbors = {u: sorted(vs) for u, vs in undirected_neighbors.items()}
    return out_neighbors, edge_set

def get_profile_embedding_path() -> Path:
    return Path(CFG.nodes_path).resolve().parent / f"profile_fields_embedding_{LLM_type}.npy"


def load_reusable_base_profile_embeddings(
    nodes_df: pd.DataFrame,
    target_emb_path: Path,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not CFG.reuse_base_profile_embeddings:
        return None, None

    base_dir_path = Path(CFG.base_profile_embedding_dir).resolve()
    base_emb_path = base_dir_path / f"profile_fields_embedding_{LLM_type}.npy"
    if base_emb_path.resolve() == target_emb_path.resolve():
        return None, None
    if not base_emb_path.exists():
        return None, None

    base_nodes_path = base_dir_path / "nodes_clean.csv"
    if not base_nodes_path.exists():
        base_nodes_path = base_dir_path / "nodes.csv"
    if not base_nodes_path.exists():
        print(f"[Embedding] Base nodes file not found, skip reuse: {base_dir_path}")
        return None, None

    base_nodes = pd.read_csv(base_nodes_path, usecols=["u_id", "orig_id"])
    base_nodes["u_id"] = base_nodes["u_id"].astype(int)
    base_nodes["orig_id"] = base_nodes["orig_id"].astype(int)
    base_nodes = base_nodes.sort_values("u_id").reset_index(drop=True)

    base_embs = np.asarray(np.load(base_emb_path), dtype=np.float32)
    if base_embs.shape[0] != len(base_nodes):
        print(
            f"[Embedding] Base embedding shape {base_embs.shape} does not match "
            f"base nodes {len(base_nodes)}, skip reuse."
        )
        return None, None

    emb_dim = int(base_embs.shape[1])
    reused_embs = np.zeros((len(nodes_df), emb_dim), dtype=np.float32)
    reused_mask = np.zeros(len(nodes_df), dtype=bool)
    base_pos_by_orig = {
        int(row.orig_id): idx
        for idx, row in enumerate(base_nodes.itertuples(index=False))
    }

    for row in nodes_df.itertuples(index=False):
        cur_idx = int(row.u_id)
        base_idx = base_pos_by_orig.get(int(row.orig_id))
        if base_idx is None:
            continue
        reused_embs[cur_idx] = base_embs[base_idx]
        reused_mask[cur_idx] = True

    reused_count = int(reused_mask.sum())
    if reused_count == 0:
        return None, None

    print(
        f"[Embedding] Reused {reused_count}/{len(nodes_df)} profile embeddings "
        f"from {base_emb_path}"
    )
    return reused_embs, reused_mask


def load_or_encode_personas(nodes_df: pd.DataFrame, encoder: QwenEncoder) -> np.ndarray:
    profile_emb_path = get_profile_embedding_path()
    profile_emb_path.parent.mkdir(parents=True, exist_ok=True)

    if profile_emb_path.exists() and not CFG.force_recompute_profile_embeddings:
        print(f"[Embedding] Loading cached profile-field embeddings: {profile_emb_path}")
        persona_embs = np.load(profile_emb_path)
        if persona_embs.shape[0] == len(nodes_df):
            return np.asarray(persona_embs, dtype=np.float32)
        print(
            f"[Embedding] Cached shape {persona_embs.shape} does not match nodes {len(nodes_df)}, recomputing."
        )
    elif profile_emb_path.exists() and CFG.force_recompute_profile_embeddings:
        print(f"[Embedding] Force recomputing profile-field embeddings; ignoring cache: {profile_emb_path}")

    persona_embs, reused_mask = load_reusable_base_profile_embeddings(
        nodes_df=nodes_df,
        target_emb_path=profile_emb_path,
    )
    if persona_embs is None or reused_mask is None:
        persona_embs = None
        reused_mask = np.zeros(len(nodes_df), dtype=bool)

    missing_indices = np.where(~reused_mask)[0].tolist()
    if not missing_indices:
        np.save(profile_emb_path, np.asarray(persona_embs, dtype=np.float32))
        print(f"[Embedding] Saved merged profile-field embeddings: {profile_emb_path}")
        return np.asarray(persona_embs, dtype=np.float32)

    missing_rows = nodes_df.iloc[missing_indices]
    persona_texts = [
        normalize_text(row.profile_text_for_llm, 800) or "This user has limited available profile information."
        for row in missing_rows.itertuples(index=False)
    ]
    new_embs = encoder.encode_texts(
        persona_texts,
        batch_size=CFG.embed_batch_size,
        max_length=CFG.embed_max_length_persona,
        progress_desc=f"[Embedding] Encoding missing Pokec raw profile fields ({len(missing_indices)} nodes)",
    )
    new_embs = np.asarray(new_embs, dtype=np.float32)

    if persona_embs is None:
        persona_embs = np.zeros((len(nodes_df), new_embs.shape[1]), dtype=np.float32)
    persona_embs[missing_indices] = new_embs
    persona_embs = np.asarray(persona_embs, dtype=np.float32)
    np.save(profile_emb_path, persona_embs)
    print(f"[Embedding] Saved profile-field embeddings: {profile_emb_path}")
    return persona_embs


def load_cached_source_candidates(num_nodes: int) -> Optional[Tuple[List[int], Optional[np.ndarray]]]:
    if CFG.force_recompute_source_candidates:
        return None
    path = Path(CFG.source_candidate_cache_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if payload.get("topic") != CFG.target_topic:
        return None
    if payload.get("similarity_basis") != "profile_fields_vs_topic_label":
        return None
    if int(payload.get("num_nodes", -1)) != num_nodes:
        return None
    if abs(float(payload.get("source_top_fraction", -1.0)) - CFG.source_top_fraction) > 1e-9:
        return None

    uids = [int(x) for x in payload.get("candidate_u_ids", [])]
    if not uids:
        return None

    print(f"[Source Candidates] Loaded cached source candidates: {path}")
    all_scores = payload.get("all_scores", None)
    if isinstance(all_scores, list) and len(all_scores) == num_nodes:
        return uids, np.asarray(all_scores, dtype=np.float32)
    return uids, None


def compute_or_load_source_candidates(
    nodes_df: pd.DataFrame,
    persona_embs: np.ndarray,
    encoder: QwenEncoder,
) -> Tuple[List[int], np.ndarray]:
    num_nodes = len(nodes_df)
    cached = load_cached_source_candidates(num_nodes)
    if cached is not None:
        source_uids, topic_scores = cached
        if topic_scores is None:
            topic_scores = np.full(num_nodes, np.nan, dtype=np.float32)
        return source_uids, topic_scores

    topic_texts = [f"topic: {CFG.target_topic}"]
    topic_embs = encoder.encode_texts(
        topic_texts,
        batch_size=CFG.embed_batch_size,
        max_length=CFG.embed_max_length_news,
    )
    topic_emb = topic_embs[:1]
    topic_scores = (persona_embs @ topic_emb.T).reshape(-1)

    top_k = max(1, int(round(num_nodes * CFG.source_top_fraction)))
    ranked = sorted(
        [(int(u), float(topic_scores[int(u)])) for u in nodes_df["u_id"].tolist()],
        key=lambda x: x[1],
        reverse=True,
    )
    source_uids = [u for u, _ in ranked[:top_k]]

    payload = {
        "topic": CFG.target_topic,
        "num_nodes": int(num_nodes),
        "source_top_fraction": float(CFG.source_top_fraction),
        "candidate_count": int(len(source_uids)),
        "similarity_basis": "profile_fields_vs_topic_label",
        "candidate_u_ids": [int(x) for x in source_uids],
        "ranked_scores": [
            {"u_id": int(u), "score": round(float(score), 6)}
            for u, score in ranked[:top_k]
        ],
        "all_scores": [round(float(x), 6) for x in topic_scores.tolist()],
        "params": str(asdict(CFG)),
    }
    ensure_parent_dir(CFG.source_candidate_cache_path)
    with Path(CFG.source_candidate_cache_path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[Source Candidates] Saved source candidates: {CFG.source_candidate_cache_path}")

    return source_uids, topic_scores


def sample_source_seeds(
    source_uids: List[int],
    num_nodes: int,
    news_idx: int,
    repeat_id: int,
) -> List[int]:
    seed_count = max(1, int(round(num_nodes * CFG.seed_fraction)))
    if seed_count > len(source_uids):
        raise ValueError(
            f"seed_count={seed_count} exceeds the number of source candidates={len(source_uids)}. "
            f"Increase source_top_fraction or decrease seed_fraction."
        )
    rng = random.Random(CFG.random_seed + news_idx * 100003 + repeat_id * 9176)
    return sorted(rng.sample(source_uids, seed_count))

def suggest_parents_for_candidate(
    cand_u: int,
    infected_list: List[int],
    out_neighbors: Dict[int, List[int]],
    persona_embs: np.ndarray,
    top_k: int = 3,
) -> List[int]:
    social_candidates = [v for v in out_neighbors.get(cand_u, []) if v in infected_list]

    if len(social_candidates) > 0:
        u_emb = persona_embs[cand_u:cand_u + 1]
        sc_embs = persona_embs[social_candidates]
        score = (u_emb @ sc_embs.T)[0]
        pairs = sorted(zip(social_candidates, score.tolist()), key=lambda x: x[1], reverse=True)
        return [int(x[0]) for x in pairs[:top_k]]

    if len(infected_list) == 0:
        return []

    u_emb = persona_embs[cand_u:cand_u + 1]
    inf_embs = persona_embs[infected_list]
    score = (u_emb @ inf_embs.T)[0]
    pairs = sorted(zip(infected_list, score.tolist()), key=lambda x: x[1], reverse=True)
    return [int(x[0]) for x in pairs[:top_k]]

def build_decisions_json_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "u_id": {"type": "integer"},
                        "p": {"type": "number"},
                        "infected": {"type": "boolean"},
                        "parent_u_id": {
                            "anyOf": [
                                {"type": "integer"},
                                {"type": "null"}
                            ]
                        },
                    },
                    "required": ["u_id", "p", "infected", "parent_u_id"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }


def build_llm_prompt(
    news_item: dict,
    candidate_rows: List[dict],
    infected_count: int,
    step_t: int,
) -> str:
    topic = normalize_text(news_item.get("topic", ""), 120)
    content = normalize_text(news_item.get("content", ""), CFG.max_news_chars)
    cand_json = json.dumps(candidate_rows, ensure_ascii=False)

    prompt = f"""
/no_think

You are simulating information diffusion on a Pokec social network.

Task:
For the current BBC news item and the candidate users in this chunk, output exactly one decision object for each candidate user.

For each candidate, estimate:
1. "p": a calibrated probability from 0 to 1 that this user becomes infected by this news in the next diffusion step,
2. "infected": a boolean value,
3. "parent_u_id": one parent chosen only from that candidate's candidate_parents list, or null.

Available evidence:
- News topic and content
- profile_text
- content_match: text encoder similarity between this user persona and the news content
- infected_similarity: text encoder similarity between this user persona and already infected users
- social_proof and social_proof_norm: how many already infected neighbors are connected to this user
- candidate_parents
- Current step and current infected population size

Probability rule:
- Current infected population size is {infected_count}.
- The external simulator will mark a candidate infected in this step only when p > {CFG.infection_threshold:.4f}.
- Do not inflate probabilities to hit a quota; there is no per-chunk or per-step infection quota.
- If no candidate has p > {CFG.infection_threshold:.4f}, mark everyone as infected=false; the simulator will stop naturally.

Important rules:
- You must reason only about the listed candidates in this chunk.
- You must output every candidate exactly once.
- Do not output any user outside this chunk.
- Different candidates should usually receive different probabilities.
- Use profile_text and content_match to judge topic interest.
- Use social_proof as a useful signal, but do not let it dominate all other evidence.

Parent attribution rules:
- If infected = true, parent_u_id must be chosen from that candidate's candidate_parents list, or null if no parent is plausible.
- If infected = false, parent_u_id should be null.
- Never output a parent_u_id that is not in the provided candidate_parents list for that candidate.

Probability guidance:
- Very weak candidate: 0.02 to 0.15
- Weak-to-uncertain candidate: 0.15 to 0.35
- Plausible candidate: 0.35 to 0.60
- Strong candidate: 0.60 to 0.85
- Very strong candidate: 0.85 to 0.97

Current step: {step_t}
Current infected population size: {infected_count}

News:
topic: {topic}
content: {content}

Candidates (JSON list):
{cand_json}

Output requirements:
- Return valid JSON only.
- Return exactly one JSON object.
- The top-level object must contain a key "decisions".
- Do not use markdown code fences.
- Do not write any explanation before or after the JSON.
- Use double quotes for all keys and all string values.
- Use null, not None.
""".strip()

    return prompt


def call_qwen_llm(prompt: str) -> str:
    schema = build_decisions_json_schema()

    payload = {
        "model": CFG.MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "/no_think\nYou are a careful simulation assistant that outputs strict JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": CFG.llm_temperature,
        "top_p": CFG.llm_top_p,
        "top_k": CFG.llm_top_k,
        "presence_penalty": CFG.llm_presence_penalty,
        "max_tokens": CFG.llm_max_tokens,
        "chat_template_kwargs": {
            "enable_thinking": False
        },
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "diffusion_decisions",
                "schema": schema
            }
        }
    }

    resp = requests.post(CFG.API_URL, json=payload, timeout=600)
    if resp.status_code != 200:
        print("[LLM ERROR]", resp.status_code)
        print(resp.text[:3000])
        resp.raise_for_status()

    data = resp.json()
    content = data["choices"][0]["message"].get("content", "")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content).strip()


def safe_json_loads(text: str):
    if isinstance(text, (dict, list)):
        return text

    if text is None:
        raise ValueError("LLM returned empty content")

    text = str(text).strip()
    if not text:
        raise ValueError("LLM returned an empty string")

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    text = text.replace("None", "null")
    text = text.replace("True", "true")
    text = text.replace("False", "false")

    try:
        return json.loads(text)
    except Exception:
        pass

    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except Exception:
            pass

    arr_match = re.search(r"\[[\s\S]*\]", text)
    if arr_match:
        return json.loads(arr_match.group(0))

    return json.loads(text)


def parse_llm_decisions(text: str) -> List[dict]:
    try:
        obj = safe_json_loads(text)
    except Exception as e:
        print("\n===== LLM RAW OUTPUT repr =====")
        print(repr(text))
        print("\n===== LLM RAW OUTPUT START =====")
        print(str(text)[:2000])
        print("\n===== LLM RAW OUTPUT END TAIL =====")
        print(str(text)[-2000:])
        print("\n===== RAW OUTPUT LENGTH =====")
        print(len(str(text)))
        raise e

    if isinstance(obj, list):
        decisions = obj
    elif isinstance(obj, dict) and "decisions" in obj:
        decisions = obj["decisions"]
    else:
        raise ValueError(f"Invalid LLM output structure: {type(obj)}")

    if not isinstance(decisions, list):
        raise ValueError("decisions must be a list")

    normalized = []
    for d in decisions:
        if not isinstance(d, dict):
            continue

        u_id = d.get("u_id", None)
        if u_id is None:
            continue

        if "p" not in d and "infection_probability" in d:
            d["p"] = d["infection_probability"]

        try:
            p = float(d.get("p", 0.0))
        except Exception:
            p = 0.0
        p = max(0.0, min(1.0, p))

        infected_val = d.get("infected", False)
        if isinstance(infected_val, str):
            infected_val = infected_val.strip().lower() in {"1", "true", "yes"}
        elif isinstance(infected_val, (int, np.integer)):
            infected_val = bool(infected_val)
        else:
            infected_val = bool(infected_val)

        parent_u_id = d.get("parent_u_id", None)
        if parent_u_id is not None:
            try:
                parent_u_id = int(parent_u_id)
            except Exception:
                parent_u_id = None

        try:
            u_id = int(u_id)
        except Exception:
            continue

        normalized.append({
            "u_id": u_id,
            "p": p,
            "infected": infected_val,
            "parent_u_id": parent_u_id,
        })

    return normalized


def compute_social_proof(
    uids: List[int],
    infected_set: Set[int],
    out_neighbors: Dict[int, List[int]],
) -> Dict[int, int]:
    soc = {}
    for u in uids:
        cnt = 0
        for v in out_neighbors.get(u, []):
            if v in infected_set:
                cnt += 1
        soc[u] = cnt
    return soc


def rank_candidates(
    candidate_uids: List[int],
    content_match: np.ndarray,
    infected_similarity: np.ndarray,
    social_proof: Dict[int, int],
    top_m: int,
) -> List[Tuple[int, float]]:
    rows = []
    max_soc = max([social_proof[u] for u in candidate_uids], default=0)
    max_soc = max(max_soc, 1)

    for u in candidate_uids:
        soc_norm = social_proof[u] / max_soc
        score = (
            CFG.w_match * float(content_match[u]) +
            CFG.w_simI * float(infected_similarity[u]) +
            CFG.w_soc * float(soc_norm)
        )
        rows.append((u, score))

    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    return rows[:top_m]


def build_fallback_decisions(candidate_rows: List[dict]) -> List[dict]:
    rows = []
    for row in candidate_rows:
        cm = float(row.get("content_match", 0.0))
        sim = float(row.get("infected_similarity", 0.0))
        sp = float(row.get("social_proof_norm", 0.0))
        p = 0.10 + 0.30 * cm + 0.25 * sim + 0.35 * sp
        p = max(0.02, min(0.95, p))
        rows.append((row, p))

    decisions = []
    for row, p in rows:
        infected = p > CFG.infection_threshold
        parent_choices = row.get("candidate_parents", [])
        parent_u_id = int(parent_choices[0]) if (infected and len(parent_choices) > 0) else None
        decisions.append({
            "u_id": int(row["u_id"]),
            "p": round(float(p), 4),
            "infected": bool(infected),
            "parent_u_id": parent_u_id,
        })
    return decisions


def llm_decide_candidates_in_chunks(
    news_idx: int,
    repeat_id: int,
    news_item: dict,
    candidate_rows: List[dict],
    infected_count: int,
    step_t: int,
) -> List[dict]:
    all_decisions = []
    candidate_chunks = chunk_list(candidate_rows, CFG.LLM_CHUNK_SIZE)

    total_llm_seconds = 0.0
    total_llm_calls = 0

    for chunk_idx, chunk in enumerate(candidate_chunks, start=1):
        chunk_uids = [int(x["u_id"]) for x in chunk]
        cached = load_chunk_cache(news_idx, repeat_id, step_t, chunk_idx)
        if (
            cached is not None
            and cached.get("candidate_u_ids") == chunk_uids
            and cached.get("cache_version") == CHUNK_CACHE_VERSION
            and float(cached.get("infection_threshold", -1.0)) == float(CFG.infection_threshold)
        ):
            print(
                f"[LLM Cache] news={news_idx} rep={repeat_id} step={step_t} chunk={chunk_idx}"
            )
            all_decisions.extend(cached.get("decisions", []))
            continue

        print(f"[LLM Call] news={news_idx} rep={repeat_id} step={step_t} chunk={chunk_idx}")
        prompt = build_llm_prompt(
            news_item=news_item,
            candidate_rows=chunk,
            infected_count=infected_count,
            step_t=step_t,
        )

        llm_elapsed_seconds = None
        used_fallback = False
        llm_error = None

        try:
            t0 = time.perf_counter()
            llm_text = call_qwen_llm(prompt)
            llm_elapsed_seconds = time.perf_counter() - t0
            total_llm_seconds += llm_elapsed_seconds
            total_llm_calls += 1

            print(
                f"[LLM Time] news={news_idx} rep={repeat_id} step={step_t} "
                f"chunk={chunk_idx} seconds={llm_elapsed_seconds:.4f}"
            )

            decisions = parse_llm_decisions(llm_text)

        except Exception as e:
            if llm_elapsed_seconds is None:
                llm_elapsed_seconds = time.perf_counter() - t0
            total_llm_seconds += llm_elapsed_seconds
            total_llm_calls += 1
            used_fallback = True
            llm_error = str(e)

            print(
                f"[LLM Parse Fallback] news={news_idx} rep={repeat_id} step={step_t} "
                f"chunk={chunk_idx}: {e}"
            )
            decisions = build_fallback_decisions(chunk)

        cache_payload = {
            "cache_version": CHUNK_CACHE_VERSION,
            "news_idx": news_idx,
            "repeat_id": repeat_id,
            "step_t": step_t,
            "chunk_idx": chunk_idx,
            "candidate_u_ids": chunk_uids,
            "decisions": decisions,
            "infection_threshold": float(CFG.infection_threshold),
            "llm_elapsed_seconds": llm_elapsed_seconds,
            "used_fallback": used_fallback,
            "llm_error": llm_error,
        }
        save_chunk_cache(news_idx, repeat_id, step_t, chunk_idx, cache_payload)
        all_decisions.extend(decisions)

    if total_llm_calls > 0:
        avg_llm_seconds = total_llm_seconds / total_llm_calls
        print(
            f"[LLM Time Summary] news={news_idx} rep={repeat_id} step={step_t} "
            f"calls={total_llm_calls} total_seconds={total_llm_seconds:.4f} "
            f"avg_seconds={avg_llm_seconds:.4f}"
        )
    return all_decisions

def normalize_decisions(
    decisions: List[dict],
    cand_uids: List[int],
    suggested_parent_map: Dict[int, List[int]],
) -> List[dict]:
    valid_candidate_set = set(cand_uids)
    parsed_rows = []
    existing_uids = set()

    for d in decisions:
        if "u_id" not in d:
            continue
        u = int(d["u_id"])
        if u not in valid_candidate_set or u in existing_uids:
            continue
        existing_uids.add(u)

        p = max(0.0, min(1.0, float(d.get("p", 0.0))))
        llm_infected_val = d.get("infected", False)
        if isinstance(llm_infected_val, str):
            llm_infected_flag = llm_infected_val.strip().lower() in {"1", "true", "yes"}
        else:
            llm_infected_flag = bool(llm_infected_val)
        infected_flag = p > CFG.infection_threshold
        parent_u = d.get("parent_u_id", None)
        if parent_u is not None:
            try:
                parent_u = int(parent_u)
            except Exception:
                parent_u = None

        allowed_parents = set(suggested_parent_map.get(u, []))
        if parent_u is not None and parent_u not in allowed_parents:
            parent_u = None

        if infected_flag and parent_u is None:
            fallback = suggested_parent_map.get(u, [])
            parent_u = fallback[0] if len(fallback) > 0 else None
        if not infected_flag:
            parent_u = None

        parsed_rows.append({
            "u_id": u,
            "p": p,
            "infected": infected_flag,
            "llm_infected": llm_infected_flag,
            "parent_u_id": parent_u,
            "forced": False,
        })

    for u in cand_uids:
        if u not in existing_uids:
            parsed_rows.append({
                "u_id": int(u),
                "p": 0.0,
                "infected": False,
                "llm_infected": False,
                "parent_u_id": None,
                "forced": False,
            })

    return parsed_rows


def select_positives_with_scale_control(
    parsed_rows: List[dict],
    max_new_this_step: int,
) -> List[dict]:
    positives_by_uid = {}
    for item in parsed_rows:
        if item["infected"] is True:
            positives_by_uid[int(item["u_id"])] = item

    positives = sorted(positives_by_uid.values(), key=lambda x: x["p"], reverse=True)
    return positives[:max_new_this_step]


def simulate_one_news_one_repeat(
    news_idx: int,
    news_item: dict,
    repeat_id: int,
    nodes_df: pd.DataFrame,
    persona_embs: np.ndarray,
    news_emb: np.ndarray,
    source_uids: List[int],
    out_neighbors: Dict[int, List[int]],
    edge_set: set,
) -> dict:
    num_nodes = len(nodes_df)
    all_uids = nodes_df["u_id"].tolist()

    profile_text_map = {
        int(row.u_id): normalize_text(row.profile_text_for_llm, CFG.max_persona_chars_in_prompt)
        for row in nodes_df.itertuples(index=False)
    }
    orig_id_map = {
        int(row.u_id): str(row.orig_id)
        for row in nodes_df.itertuples(index=False)
    }

    seeds = sample_source_seeds(
        source_uids=source_uids,
        num_nodes=num_nodes,
        news_idx=news_idx,
        repeat_id=repeat_id,
    )

    infected_set = set(seeds)
    infection_time = {u: 0 for u in seeds}
    parent = {u: None for u in seeds}
    transmission_edges = []
    step_logs = []
    stop_reason_detail = None

    content_match_full = (persona_embs @ news_emb.T).reshape(-1)

    for t in range(1, CFG.max_steps + 1):
        if len(infected_set) >= CFG.max_final_infected:
            print(f"[Stop] news={news_idx} rep={repeat_id} reached max size {len(infected_set)}")
            stop_reason_detail = "reached_max_final_infected"
            break

        uninfected = [u for u in all_uids if u not in infected_set]
        if len(uninfected) == 0:
            stop_reason_detail = "all_nodes_infected"
            break

        infected_list = sorted(list(infected_set))
        infected_centroid = persona_embs[infected_list].mean(axis=0, keepdims=True)
        infected_centroid = l2_normalize(infected_centroid)
        infected_similarity = (persona_embs @ infected_centroid.T).reshape(-1)
        social_proof = compute_social_proof(uninfected, infected_set, out_neighbors)

        cand_rank = rank_candidates(
            candidate_uids=uninfected,
            content_match=content_match_full,
            infected_similarity=infected_similarity,
            social_proof=social_proof,
            top_m=CFG.TOP_M,
        )
        cand_uids = [u for u, _ in cand_rank]
        if len(cand_uids) == 0:
            stop_reason_detail = "no_candidates"
            break

        max_new_this_step = CFG.max_final_infected - len(infected_set)
        if max_new_this_step <= 0:
            stop_reason_detail = "reached_max_final_infected"
            break

        candidate_rows = []
        max_soc = max([social_proof[u] for u in cand_uids], default=0)
        max_soc = max(max_soc, 1)

        for u, pre_score in cand_rank:
            parents = suggest_parents_for_candidate(
                cand_u=u,
                infected_list=infected_list,
                out_neighbors=out_neighbors,
                persona_embs=persona_embs,
                top_k=3,
            )
            candidate_rows.append({
                "u_id": int(u),
                "orig_id": str(orig_id_map[u]),
                "content_match": round(float(content_match_full[u]), 4),
                "infected_similarity": round(float(infected_similarity[u]), 4),
                "social_proof": int(social_proof[u]),
                "social_proof_norm": round(float(social_proof[u] / max_soc), 4),
                "pre_score": round(float(pre_score), 4),
                "candidate_parents": [int(x) for x in parents],
                "profile_text": profile_text_map[u],
            })

        decisions = llm_decide_candidates_in_chunks(
            news_idx=news_idx,
            repeat_id=repeat_id,
            news_item=news_item,
            candidate_rows=candidate_rows,
            infected_count=len(infected_set),
            step_t=t,
        )

        suggested_parent_map = {
            row["u_id"]: row["candidate_parents"]
            for row in candidate_rows
        }
        parsed_rows = normalize_decisions(
            decisions=decisions,
            cand_uids=cand_uids,
            suggested_parent_map=suggested_parent_map,
        )
        positives = select_positives_with_scale_control(
            parsed_rows=parsed_rows,
            max_new_this_step=max_new_this_step,
        )
        threshold_positive_count = sum(
            1 for item in parsed_rows
            if float(item.get("p", 0.0)) > CFG.infection_threshold
        )

        if len(positives) == 0:
            step_logs.append({
                "t": t,
                "candidate_count": len(cand_uids),
                "chunk_count": len(chunk_list(candidate_rows, CFG.LLM_CHUNK_SIZE)),
                "infection_threshold": float(CFG.infection_threshold),
                "threshold_positive_count": int(threshold_positive_count),
                "max_new_this_step": int(max_new_this_step),
                "new_infected": [],
                "infected_count_after_step": len(infected_set),
                "stop_reason": "no_probability_above_threshold",
            })
            print(
                f"[Stop] news={news_idx} rep={repeat_id} step={t} "
                f"no p > {CFG.infection_threshold:.4f}; infected={len(infected_set)}"
            )
            stop_reason_detail = "no_probability_above_threshold"
            break

        new_nodes = []
        for item in positives:
            if len(infected_set) >= CFG.max_final_infected:
                stop_reason_detail = "reached_max_final_infected"
                break

            u = int(item["u_id"])
            if u in infected_set:
                continue

            parent_u = item.get("parent_u_id", None)
            if parent_u is None:
                fallback = suggested_parent_map.get(u, [])
                parent_u = fallback[0] if len(fallback) > 0 else None

            infected_set.add(u)
            infection_time[u] = t
            parent[u] = parent_u

            new_item = {
                "u_id": u,
                "p": round(float(item["p"]), 4),
                "infected": True,
                "parent_u_id": None if parent_u is None else int(parent_u),
                "forced": False,
            }
            new_nodes.append(new_item)

            if parent_u is not None:
                source_type = "social" if ((u, parent_u) in edge_set or (parent_u, u) in edge_set) else "recommendation"
                transmission_edges.append({
                    "parent_u_id": int(parent_u),
                    "child_u_id": int(u),
                    "t": int(t),
                    "source_type": source_type,
                    "p": round(float(item["p"]), 4),
                    "forced": False,
                })

        print(
            f"[Step] news={news_idx} rep={repeat_id} step={t} "
            f"new={len(new_nodes)} infected={len(infected_set)}"
        )

        step_logs.append({
            "t": t,
            "candidate_count": len(cand_uids),
            "chunk_count": len(chunk_list(candidate_rows, CFG.LLM_CHUNK_SIZE)),
            "infection_threshold": float(CFG.infection_threshold),
            "threshold_positive_count": int(threshold_positive_count),
            "max_new_this_step": int(max_new_this_step),
            "new_infected": new_nodes,
            "infected_count_after_step": int(len(infected_set)),
        })

        if len(new_nodes) == 0:
            stop_reason_detail = "no_new_nodes"
            break

    infected_mask = [0] * num_nodes
    for u in infected_set:
        infected_mask[u] = 1

    final_size = len(infected_set)
    if stop_reason_detail is not None:
        stop_reason = stop_reason_detail
    elif final_size < CFG.min_final_infected:
        stop_reason = "max_steps_but_below_min"
    elif final_size >= CFG.max_final_infected:
        stop_reason = "reached_max_final_infected"
    else:
        stop_reason = "stopped_between_min_and_max"

    observation = {
        "news_topic": news_item.get("topic", ""),
        "news_content": normalize_text(news_item.get("content", "")),
        "news_idx": int(news_idx),
        "source_line_idx": int(news_item.get("_source_line_idx", -1)),
        "topic_local_idx": int(news_item.get("_topic_local_idx", news_idx)),
        "repeat_id": int(repeat_id),
        "seeds": [int(x) for x in seeds],
        "seed_count": int(len(seeds)),
        "source_candidate_count": int(len(source_uids)),
        "infected_mask": infected_mask,
        "infected_u_ids": sorted([int(x) for x in infected_set]),
        "infected_count": int(final_size),
        "infection_time": {str(int(k)): int(v) for k, v in infection_time.items()},
        "parent": {
            str(int(k)): (None if v is None else int(v))
            for k, v in parent.items()
        },
        "transmission_edges": transmission_edges,
        "step_logs": step_logs,
        "stop_reason": stop_reason,
        "params": str(asdict(CFG)),
    }
    return observation

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, default=None, help="BBC topic, e.g. business/sport/tech/politics/entertainment")
    parser.add_argument("--news-start", type=int, default=None, help="0-based inclusive start index within the selected topic")
    parser.add_argument("--news-end", type=int, default=None, help="0-based exclusive end index within the selected topic")
    parser.add_argument("--repeat", type=int, default=None, help="Repeat count per news")
    parser.add_argument("--api-url", type=str, default=None, help="Override OpenAI-compatible local API URL")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=f"Local Qwen model path. Can also be set with {MODEL_PATH_ENV}.",
    )
    parser.add_argument("--min-final-infected", type=int, default=None)
    parser.add_argument("--max-final-infected", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--top-m", type=int, default=None)
    parser.add_argument("--infection-threshold", type=float, default=None, help="Infect candidates with p strictly greater than this threshold")
    parser.add_argument("--source-top-fraction", type=float, default=None)
    parser.add_argument("--seed-fraction", type=float, default=None)
    parser.add_argument("--embed-device", type=str, default=None, help="Embedding device, e.g. cpu/cuda/cuda:0")
    parser.add_argument("--embed-batch-size", type=int, default=None)
    parser.add_argument("--base-profile-embedding-dir", type=str, default=None)
    parser.add_argument("--no-reuse-base-profile-embeddings", action="store_true")
    parser.add_argument("--force-recompute-profile-embeddings", action="store_true")
    parser.add_argument("--force-recompute-source-candidates", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.topic is not None:
        CFG.target_topic = args.topic
    if args.news_start is not None:
        CFG.news_start = args.news_start
    if args.news_end is not None:
        CFG.news_end = args.news_end
    if args.repeat is not None:
        CFG.R = args.repeat
    if args.api_url is not None:
        CFG.API_URL = args.api_url
    CFG.LOCAL_QWEN_PATH = resolve_model_path(args.model_path)
    if args.min_final_infected is not None:
        CFG.min_final_infected = args.min_final_infected
    if args.max_final_infected is not None:
        CFG.max_final_infected = args.max_final_infected
    if args.max_steps is not None:
        CFG.max_steps = args.max_steps
    if args.top_m is not None:
        CFG.TOP_M = args.top_m
    if args.infection_threshold is not None:
        CFG.infection_threshold = args.infection_threshold
    if not (0.0 <= CFG.infection_threshold <= 1.0):
        raise ValueError(f"--infection-threshold must be within [0, 1], got {CFG.infection_threshold}")
    if args.source_top_fraction is not None:
        CFG.source_top_fraction = args.source_top_fraction
    if args.seed_fraction is not None:
        CFG.seed_fraction = args.seed_fraction
    if args.embed_device is not None:
        CFG.embed_device = args.embed_device
    if args.embed_batch_size is not None:
        CFG.embed_batch_size = args.embed_batch_size
    if args.base_profile_embedding_dir is not None:
        CFG.base_profile_embedding_dir = Path(args.base_profile_embedding_dir)
    if args.no_reuse_base_profile_embeddings:
        CFG.reuse_base_profile_embeddings = False
    if args.force_recompute_profile_embeddings:
        CFG.force_recompute_profile_embeddings = True
    if args.force_recompute_source_candidates:
        CFG.force_recompute_source_candidates = True

    configure_runtime_paths()

    random.seed(CFG.random_seed)
    np.random.seed(CFG.random_seed)
    torch.manual_seed(CFG.random_seed)

    ensure_parent_dir(CFG.output_path)
    ensure_dir(CFG.cache_dir)

    print("[1] Loading data...")
    nodes_df = load_nodes(CFG.nodes_path)
    edges_df = load_edges(CFG.edges_path)
    all_topic_news = load_topic_news(CFG.news_path, CFG.target_topic)
    topic_news = select_news_range(all_topic_news)

    print(f"nodes: {nodes_df.shape}")
    print(f"edges: {edges_df.shape}")
    print(f"topic: {CFG.target_topic}")
    print(f"topic news total: {len(all_topic_news)}")
    print(f"news range: [{CFG.news_start}, {CFG.news_end})")
    print(f"news used: {len(topic_news)}")
    print(f"final infected target range: [{CFG.min_final_infected}, {CFG.max_final_infected}]")
    print(f"infection threshold: p > {CFG.infection_threshold}")
    print(f"output: {CFG.output_path}")

    print("[2] Building graph helpers (treat all edges as undirected)...")
    out_neighbors, edge_set = build_graph_helpers(nodes_df, edges_df)

    print("[3] Loading local Qwen encoder...")
    encoder = QwenEncoder(
        model_path=CFG.LOCAL_QWEN_PATH,
        device=CFG.embed_device,
    )

    print("[4] Loading/encoding persona embeddings...")
    persona_embs = load_or_encode_personas(nodes_df, encoder)
    print("persona_embs:", persona_embs.shape)

    print("[5] Loading/computing topic source candidates...")
    source_uids, topic_scores = compute_or_load_source_candidates(
        nodes_df=nodes_df,
        persona_embs=persona_embs,
        encoder=encoder,
    )
    print(f"source candidates: {len(source_uids)}")
    if np.isfinite(topic_scores).any():
        print(
            f"topic score range: "
            f"min={float(np.nanmin(topic_scores)):.4f}, max={float(np.nanmax(topic_scores)):.4f}"
        )
    else:
        print("topic score range: loaded source candidates from cache; scores not stored in this cache")

    print("[6] Simulating diffusion...")
    results = []

    for news_idx, item in enumerate(topic_news):
        news_text = build_news_text(item, max_chars=CFG.max_news_chars)
        news_emb = encoder.encode_texts(
            [news_text],
            batch_size=1,
            max_length=CFG.embed_max_length_news,
        )

        for r in range(CFG.R):
            print(f"  -> topic={CFG.target_topic} news {news_idx + 1}/{len(topic_news)}, repeat {r + 1}/{CFG.R}")
            obs = simulate_one_news_one_repeat(
                news_idx=news_idx,
                news_item=item,
                repeat_id=r,
                nodes_df=nodes_df,
                persona_embs=persona_embs,
                news_emb=news_emb,
                source_uids=source_uids,
                out_neighbors=out_neighbors,
                edge_set=edge_set,
            )
            results.append(obs)

        if (news_idx + 1) % CFG.save_every_news == 0:
            with Path(CFG.output_path).open("w", encoding="utf-8") as f:
                for obj in results:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            print(f"[Saved] {len(results)} observations -> {CFG.output_path}")

    with Path(CFG.output_path).open("w", encoding="utf-8") as f:
        for obj in results:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[Done] saved to {CFG.output_path}")
    print(f"[Cache Dir] {CFG.cache_dir}")
    print(f"[Source Candidate Cache] {CFG.source_candidate_cache_path}")


if __name__ == "__main__":
    main()
