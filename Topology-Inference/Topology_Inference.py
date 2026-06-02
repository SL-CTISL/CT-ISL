"""Unified MI network inference script.

The workflow uses final-state diffusion records to build a shared topology,
then refines each topology with the same final-state records.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import hashlib
import json
import math
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


VFSL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIFFUSION_ROOT = (
    VFSL_ROOT
    / "Dataset-diffusion"
    / "Classical-diffusion"
    / "simulated-diffusion"
)
DEFAULT_OUTPUT_ROOT = VFSL_ROOT / "Topology-Inference" / "results"
DEFAULT_LLM_DIFFUSION_ROOT = VFSL_ROOT / "Dataset-diffusion" / "LLM-diffusion"
DEFAULT_LLM_OUTPUT_ROOT = VFSL_ROOT / "Topology-Inference" / "results-llm-context"
DEFAULT_LLM_TOPICS = ["business", "entertainment", "politics", "sport", "tech"]
DEFAULT_DIFFUSION_ROOTS = {
    ("cora_ml", "SI"): DEFAULT_DIFFUSION_ROOT,
    ("cora_ml", "IC"): DEFAULT_DIFFUSION_ROOT,
    ("karate", "SI"): DEFAULT_DIFFUSION_ROOT,
    ("karate", "IC"): DEFAULT_DIFFUSION_ROOT,
    ("power_grid", "SI"): DEFAULT_DIFFUSION_ROOT,
    ("power_grid", "IC"): DEFAULT_DIFFUSION_ROOT,
}


@dataclass
class PairScoreRanges:
    mi_min: float
    mi_max: float
    persona_min: float
    persona_max: float
    score_min: float
    score_max: float


@dataclass
class PairScoreCache:
    node_num: int
    pair_count: int
    mi_raw: np.memmap
    persona_raw: np.memmap
    edge: np.memmap
    score: np.memmap
    ranges: PairScoreRanges


class Tee:
    """Write stdout/stderr to both terminal and a log file."""

    def __init__(self, path: Path, stream):
        self.file = path.open("a", buffering=1)
        self.stream = stream

    def write(self, data: str) -> int:
        self.file.write(data)
        return self.stream.write(data)

    def flush(self) -> None:
        self.file.flush()
        self.stream.flush()


def setup_logging(out_dir: Path) -> Path:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    log_path = log_dir / f"mi_pind_v12_{stamp}.log"
    sys.stdout = Tee(log_path, sys.stdout)
    sys.stderr = Tee(log_path, sys.stderr)
    print(f"[LOG] Writing to: {log_path}", flush=True)
    return log_path


def per_record_f1(
    ground_truth: np.ndarray,
    inferred: np.ndarray,
    records: np.ndarray,
) -> np.ndarray:
    record_array = np.asarray(records)
    if record_array.ndim == 1:
        record_array = record_array.reshape(1, -1)

    gt_undirected = (ground_truth != 0) | (ground_truth.T != 0)
    pred_undirected = (inferred != 0) | (inferred.T != 0)
    values = []
    eps = 1e-20

    for record in record_array:
        active_nodes = np.flatnonzero(record != 0)
        if active_nodes.size < 2:
            values.append((0.0, 0.0, 0.0))
            continue

        row_idx, col_idx = np.triu_indices(active_nodes.size, k=1)
        src = active_nodes[row_idx]
        dst = active_nodes[col_idx]
        gt = gt_undirected[src, dst]
        pred = pred_undirected[src, dst]
        tp = float(np.sum(gt & pred))
        fp = float(np.sum((~gt) & pred))
        fn = float(np.sum(gt & (~pred)))
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        values.append((precision, recall, f1))

    return np.asarray(values, dtype=np.float64)


def summarize_record_metrics(metrics: np.ndarray) -> dict:
    if metrics.size == 0:
        return {
            "precision_mean": 0.0,
            "recall_mean": 0.0,
            "f1_mean": 0.0,
        }
    return {
        "precision_mean": float(np.mean(metrics[:, 0])),
        "recall_mean": float(np.mean(metrics[:, 1])),
        "f1_mean": float(np.mean(metrics[:, 2])),
    }


def records_fingerprint(record_data_list: list[np.ndarray]) -> str:
    hasher = hashlib.sha256()
    for records in record_data_list:
        arr = np.ascontiguousarray(records.astype(np.int8, copy=False))
        hasher.update(str(arr.shape).encode("utf-8"))
        hasher.update(arr.tobytes())
    return hasher.hexdigest()


def _mi_term(nxy: np.ndarray, nx: np.ndarray, ny: np.ndarray, beta: int) -> np.ndarray:
    term = np.zeros(nxy.shape, dtype=np.float64)
    valid = (nxy > 0) & (nx > 0) & (ny > 0)
    if not np.any(valid):
        return term

    ratio = np.zeros(nxy.shape, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(beta * nxy, nx * ny, out=ratio, where=valid)
        np.log(ratio, out=ratio, where=valid)
    term[valid] = (nxy[valid] / beta) * ratio[valid]
    return term


def mutual_information_matrix(
    records: np.ndarray,
    mode: int = 2,
    show_progress: bool = True,
    block_size: int = 512,
) -> np.ndarray:
    records = (records != 0).astype(np.float32, copy=False)
    beta, node_num = records.shape
    mi = np.zeros((node_num, node_num), dtype=np.float32)
    col_sum = records.sum(axis=0, dtype=np.float64)
    n1_y = col_sum.reshape(1, node_num)
    n0_y = beta - n1_y

    block_size = max(1, int(block_size))
    block_starts = range(0, node_num, block_size)
    total_blocks = (node_num + block_size - 1) // block_size
    if show_progress and tqdm is not None:
        block_starts = tqdm(block_starts, desc="mutual information", unit="block", total=total_blocks, mininterval=1.0)

    for block_idx, start in enumerate(block_starts, start=1):
        end = min(start + block_size, node_num)
        block_records = records[:, start:end]
        n11 = (block_records.T @ records).astype(np.float64, copy=False)
        n1_x = col_sum[start:end].reshape(end - start, 1)
        n0_x = beta - n1_x

        n10 = n1_x - n11
        n01 = n1_y - n11
        n00 = beta - n1_x - n1_y + n11

        same = _mi_term(n11, n1_x, n1_y, beta) + _mi_term(n00, n0_x, n0_y, beta)
        term10 = _mi_term(n10, n1_x, n0_y, beta)
        term01 = _mi_term(n01, n0_x, n1_y, beta)
        if mode == 1:
            block_mi = same - term10 - term01
        elif mode == 2:
            block_mi = same - np.abs(term10) - np.abs(term01)
        else:
            block_mi = same + term10 + term01

        mi[start:end, :] = block_mi.astype(np.float32, copy=False)
        if show_progress and tqdm is None:
            done = end
            width = 30
            filled = int(width * done / max(1, node_num))
            bar = "#" * filled + "." * (width - filled)
            pct = 100.0 * done / max(1, node_num)
            print(
                f"[MI] [{bar}] {pct:6.2f}% "
                f"({done}/{node_num} nodes, block {block_idx}/{total_blocks})",
                flush=True,
            )

    mi = 0.5 * (mi + mi.T)
    np.fill_diagonal(mi, 0)
    return mi


def build_shared_topology(
    record_data_list: list[np.ndarray],
    mode: int,
    threshold_scale: float,
    top_in_per_node: int | None,
    block_size: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    all_records = np.concatenate(record_data_list, axis=0)
    print(
        f"computing shared-topology MI: records={all_records.shape[0]}, nodes={all_records.shape[1]}",
        flush=True,
    )
    mi = mutual_information_matrix(all_records, mode=mode, show_progress=True, block_size=block_size)

    upper_mask = np.triu(np.ones_like(mi, dtype=bool), k=1)
    vals = mi[upper_mask & (mi > 0)].reshape(-1, 1)
    if vals.size == 0 or np.unique(vals).size < 2:
        threshold = 0.0
    else:
        labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(vals)
        cluster0 = vals[labels == 0]
        cluster1 = vals[labels == 1]
        if cluster0.size == 0 or cluster1.size == 0:
            threshold = 0.0
        else:
            max0 = cluster0.max()
            max1 = cluster1.max()
            threshold = float(min(max0, max1)) * threshold_scale
            # min0 = cluster0.min()
            # min1 = cluster1.min()
            # threshold = float(max(min0, min1)) * threshold_scale

    shared = (mi > threshold).astype(np.float32)
    np.fill_diagonal(shared, 0)

    if top_in_per_node is not None and top_in_per_node > 0:
        top_shared = np.zeros_like(shared)
        for child in range(shared.shape[0]):
            scores = mi[:, child].copy()
            scores[child] = -np.inf
            idx = np.argsort(-scores)[:top_in_per_node]
            idx = idx[scores[idx] > 0]
            top_shared[idx, child] = 1
        shared = np.maximum(shared, top_shared)

    shared = symmetrize_graph(shared)

    return mi, threshold, shared


def load_or_build_shared_topology(
    args: argparse.Namespace,
    record_data_list: list[np.ndarray],
    node_num: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    cache_path = args.out_dir / "shared_topology.npz"
    record_hash = records_fingerprint(record_data_list)
    expected_record_count = sum(records.shape[0] for records in record_data_list)

    if cache_path.exists() and not args.force_shared_recompute:
        cached = np.load(cache_path, allow_pickle=False)
        mi = cached["mi"]
        threshold = float(cached["threshold"])
        shared = cached["shared_topology"]
        cached_hash = str(cached["record_hash"]) if "record_hash" in cached.files else None
        cached_mi_mode = int(cached["mi_mode"]) if "mi_mode" in cached.files else None
        cached_threshold_scale = (
            float(cached["mi_threshold_scale"])
            if "mi_threshold_scale" in cached.files
            else None
        )
        cached_top_in = (
            int(cached["shared_top_in_per_node"])
            if "shared_top_in_per_node" in cached.files
            else -1
        )
        expected_top_in = args.shared_top_in_per_node if args.shared_top_in_per_node is not None else -1

        cache_matches = (
            mi.shape == (node_num, node_num)
            and shared.shape == (node_num, node_num)
            and cached_hash == record_hash
            and cached_mi_mode == args.mi_mode
            and cached_threshold_scale == args.mi_threshold_scale
            and cached_top_in == expected_top_in
        )
        if cache_matches:
            print(
                f"loaded shared topology/MI cache: {cache_path} "
                f"records={expected_record_count} hash={record_hash[:12]}",
                flush=True,
            )
            return mi, threshold, symmetrize_graph(shared)

        print(
            "shared topology cache mismatch; recomputing. "
            f"cache_mi_shape={mi.shape}, expected={(node_num, node_num)}, "
            f"cache_hash={cached_hash}, expected_hash={record_hash}",
            flush=True,
        )

    print("building shared topology from records...", flush=True)
    mi, threshold, shared = build_shared_topology(
        record_data_list,
        mode=args.mi_mode,
        threshold_scale=args.mi_threshold_scale,
        top_in_per_node=args.shared_top_in_per_node,
        block_size=args.mi_block_size,
    )
    np.savez_compressed(
        cache_path,
        mi=mi,
        threshold=np.array(threshold),
        shared_topology=shared,
        record_hash=np.array(record_hash),
        record_count=np.array(expected_record_count),
        mi_mode=np.array(args.mi_mode),
        mi_threshold_scale=np.array(args.mi_threshold_scale),
        shared_top_in_per_node=np.array(
            args.shared_top_in_per_node if args.shared_top_in_per_node is not None else -1
        ),
    )
    save_edge_list(args.out_dir / "shared_topology_edges.txt", shared)
    print(
        f"saved shared topology/MI cache: {cache_path} "
        f"records={expected_record_count} hash={record_hash[:12]}",
        flush=True,
    )
    return mi, threshold, shared


def save_edge_list(path: Path, graph: np.ndarray) -> None:
    edges = undirected_edge_array(graph)
    np.savetxt(path, edges, fmt="%d", delimiter="\t")


def mi_probability_scale(mi: np.ndarray) -> tuple[float, float]:
    upper_mask = np.triu(np.ones_like(mi, dtype=bool), k=1)
    positive = mi[upper_mask & (mi > 0)]
    if positive.size == 0:
        return 0.0, 0.0
    return float(np.min(positive)), float(np.max(positive))


def probability_from_mi_score(mi_score: float, mi_min: float, mi_max: float, p_min: float, p_max: float) -> float:
    if mi_score <= 0:
        return p_min
    if mi_max <= mi_min + 1e-12:
        return p_max
    normalized = (mi_score - mi_min) / (mi_max - mi_min)
    normalized = min(1.0, max(0.0, normalized))
    return float(p_min + normalized * (p_max - p_min))


def probability_matrix_from_mi(graph: np.ndarray, mi: np.ndarray, p_min: float, p_max: float) -> np.ndarray:
    graph = symmetrize_graph(graph)
    p_matrix = np.zeros_like(graph, dtype=np.float64)
    mi_min, mi_max = mi_probability_scale(mi)
    for u, v in undirected_edge_array(graph):
        p_value = probability_from_mi_score(float(mi[int(u), int(v)]), mi_min, mi_max, p_min, p_max)
        p_matrix[int(u), int(v)] = p_value
        p_matrix[int(v), int(u)] = p_value
    return p_matrix


def contribution(records: np.ndarray, parent: int, p_value: float) -> np.ndarray:
    """Return the log-survival contribution from one infected parent."""
    p_value = float(np.clip(p_value, 1e-12, 1.0 - 1e-12))
    return (records[:, parent] != 0).astype(np.float64) * math.log1p(-p_value)


def column_loss(
    records: np.ndarray,
    child: int,
    log_survival_sum: np.ndarray,
    alpha: float,
) -> float:
    """IC log-likelihood for one child column; larger is better."""
    _ = alpha
    eps = 1e-12
    y = (records[:, child] != 0).astype(np.float64)
    q = 1.0 - np.exp(log_survival_sum)
    q = np.clip(q, eps, 1.0 - eps)
    return float(np.sum(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))


def graph_log_sum(graph: np.ndarray, p_matrix: np.ndarray, records: np.ndarray) -> np.ndarray:
    beta, node_num = records.shape
    sums = np.zeros((beta, node_num), dtype=np.float64)  
    for child in range(node_num):
        parents = np.where((graph[:, child] != 0) | (graph[child, :] != 0))[0]
        parents = parents[parents != child]
        for parent in parents:
            sums[:, child] += contribution(records, int(parent), float(p_matrix[parent, child]))
    return sums


def likelihood_loss_from_sum(
    records: np.ndarray,
    log_survival_sum: np.ndarray,
    alpha: float,
) -> float:
    """IC log-likelihood over all records/nodes; larger is better."""
    _ = alpha
    eps = 1e-12
    y = (records != 0).astype(np.float64)
    q = 1.0 - np.exp(log_survival_sum)
    q = np.clip(q, eps, 1.0 - eps)
    return float(np.sum(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))


def top_add_candidates(mi: np.ndarray, graph: np.ndarray, per_node: int) -> list[tuple[int, int]]:
    if per_node <= 0:
        return []
    candidates: set[tuple[int, int]] = set()
    node_num = graph.shape[0]
    for child in range(node_num):
        scores = mi[:, child].copy()
        scores[child] = -np.inf
        connected = (graph[:, child] != 0) | (graph[child, :] != 0)
        scores[connected] = -np.inf
        idx = np.argsort(-scores)[:per_node]
        for parent in idx:
            if np.isfinite(scores[parent]) and scores[parent] > 0:
                u, v = sorted((int(parent), child))
                if u != v:
                    candidates.add((u, v))
    return sorted(candidates, key=lambda edge: (-float(mi[edge[0], edge[1]]), edge[0], edge[1]))


def edit_budget(fraction: float, item_count: int) -> int:
    """Convert an edit fraction to a budget, keeping tiny positive budgets usable."""
    if item_count <= 0:
        return 0
    raw_budget = int(fraction * item_count)
    if raw_budget < 0:
        return 1
    if fraction > 0 and raw_budget < 1:
        return 1
    return raw_budget


def refine_one_pass(
    graph: np.ndarray,
    mi: np.ndarray,
    records: np.ndarray,
    p_init: float,
    p_min: float,
    min_improvement: float,
    max_delete_fraction: float,
    max_add_fraction: float,
    delete_mi_threshold: float,
    add_top_candidates_per_node: int,
    add_max_loss_increase: float,
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    """Run one brute-force batch edit pass from the current graph."""
    graph = symmetrize_graph(graph)
    p_matrix = probability_matrix_from_mi(graph, mi, p_min, p_init)
    log_sum = graph_log_sum(graph, p_matrix, records)
    mi_min, mi_max = mi_probability_scale(mi)

    current_edges = undirected_edge_array(graph)
    delete_budget = edit_budget(max_delete_fraction, len(current_edges))
    delete_moves = []
    for u, v in current_edges:
        u = int(u)
        v = int(v)
        c_u_to_v = contribution(records, u, p_matrix[u, v])
        c_v_to_u = contribution(records, v, p_matrix[v, u])
        old_loss_v = column_loss(records, v, log_sum[:, v], p_init)
        new_loss_v = column_loss(records, v, log_sum[:, v] - c_u_to_v, p_init)
        old_loss_u = column_loss(records, u, log_sum[:, u], p_init)
        new_loss_u = column_loss(records, u, log_sum[:, u] - c_v_to_u, p_init)
        delta = (new_loss_v - old_loss_v) + (new_loss_u - old_loss_u)
        mi_score = float(mi[u, v])
        low_mi = mi_score <= delete_mi_threshold
        if delta > min_improvement or low_mi:
            delete_moves.append((float(delta), mi_score, low_mi, u, v, c_u_to_v, c_v_to_u))
    delete_moves.sort(key=lambda x: (not x[2], -x[0], x[1]))

    deleted = 0
    for delta, _mi_score, _low_mi, u, v, c_u_to_v, c_v_to_u in delete_moves[:delete_budget]:
        if graph[u, v] == 0 and graph[v, u] == 0:
            continue
        graph[u, v] = 0
        graph[v, u] = 0
        p_matrix[u, v] = 0
        p_matrix[v, u] = 0
        log_sum[:, v] -= c_u_to_v
        log_sum[:, u] -= c_v_to_u
        deleted += 1

    add_budget = edit_budget(max_add_fraction, undirected_edge_count(graph))
    add_candidates = top_add_candidates(mi, graph, add_top_candidates_per_node)
    add_moves = []
    for u, v in add_candidates:
        p_value = probability_from_mi_score(float(mi[u, v]), mi_min, mi_max, p_min, p_init)
        c_u_to_v = contribution(records, u, p_value)
        c_v_to_u = contribution(records, v, p_value)
        old_loss_v = column_loss(records, v, log_sum[:, v], p_init)
        new_loss_v = column_loss(records, v, log_sum[:, v] + c_u_to_v, p_init)
        old_loss_u = column_loss(records, u, log_sum[:, u], p_init)
        new_loss_u = column_loss(records, u, log_sum[:, u] + c_v_to_u, p_init)
        delta = (new_loss_v - old_loss_v) + (new_loss_u - old_loss_u)
        if delta <= min_improvement:
            continue
        if delta > add_max_loss_increase:
            continue
        add_moves.append((float(delta), float(mi[u, v]), u, v, c_u_to_v, c_v_to_u))
    add_moves.sort(key=lambda x: -x[0])

    added = 0
    for delta, _mi_score, u, v, c_u_to_v, c_v_to_u in add_moves[:add_budget]:
        if graph[u, v] == 1 or graph[v, u] == 1:
            continue
        graph[u, v] = 1
        graph[v, u] = 1
        p_value = probability_from_mi_score(float(mi[u, v]), mi_min, mi_max, p_min, p_init)
        p_matrix[u, v] = p_value
        p_matrix[v, u] = p_value
        log_sum[:, v] += c_u_to_v
        log_sum[:, u] += c_v_to_u
        added += 1

    current_loss = likelihood_loss_from_sum(
        records,
        log_sum,
        p_init,
    )

    return graph, p_matrix, current_loss, deleted, added


def refine_graph_by_likelihood(
    shared: np.ndarray,
    mi: np.ndarray,
    records: np.ndarray,
    p_init: float,
    p_min: float,
    min_improvement: float,
    max_delete_fraction: float,
    max_add_fraction: float,
    delete_mi_threshold: float,
    add_top_candidates_per_node: int,
    add_max_loss_increase: float,
    refine_iters: int,
    task: int,
) -> tuple[np.ndarray, np.ndarray, float, int, int, int]:
    """Iteratively refine and return the final graph after the refinement loop."""
    graph = symmetrize_graph(shared)
    total_deleted = 0
    total_added = 0
    p_matrix = probability_matrix_from_mi(graph, mi, p_min, p_init)
    loss = likelihood_loss_from_sum(
        records,
        graph_log_sum(graph, p_matrix, records),
        p_init,
    )

    used_refine_iters = 0
    for refine_iter in range(1, refine_iters + 1):
        used_refine_iters = refine_iter
        graph, p_matrix, loss, deleted, added = refine_one_pass(
            graph=graph,
            mi=mi,
            records=records,
            p_init=p_init,
            p_min=p_min,
            min_improvement=min_improvement,
            max_delete_fraction=max_delete_fraction,
            max_add_fraction=max_add_fraction,
            delete_mi_threshold=delete_mi_threshold,
            add_top_candidates_per_node=add_top_candidates_per_node,
            add_max_loss_increase=add_max_loss_increase,
        )
        total_deleted += deleted
        total_added += added

        stopped = deleted == 0 and added == 0
        print(
            f"task={task} refine_iter={refine_iter} "
            f"edges={undirected_edge_count(graph)} loss={loss:.6f} "
            f"deleted={deleted} added={added}",
            flush=True,
        )
        if stopped:
            break

    return (
        graph,
        p_matrix,
        loss,
        total_deleted,
        total_added,
        used_refine_iters,
    )



def parse_ratio_label(value: str) -> str:
    text = str(value).strip()
    if text.endswith("%"):
        percent = float(text[:-1])
    else:
        parsed = float(text)
        percent = parsed if parsed > 1 else parsed * 100.0
    if percent.is_integer():
        return f"{int(percent)}%"
    return f"{percent:g}%"


def format_topology_id(dataset_name: str) -> str:
    topology_id = re.sub(r"[^0-9A-Za-z_.-]+", "_", dataset_name.strip())
    return topology_id.strip("_") or "topology"


def minmax_normalize(values: np.ndarray, low: float, high: float) -> np.ndarray:
    span = high - low
    if span <= 1e-12:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values - low) / span, 0.0, 1.0).astype(np.float32, copy=False)


def finite_min_max(values: np.ndarray, cur_min: float, cur_max: float) -> tuple[float, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return cur_min, cur_max
    return min(cur_min, float(finite.min())), max(cur_max, float(finite.max()))


def all_pair_count(node_num: int) -> int:
    return node_num * (node_num - 1) // 2


def row_offset(u: int, node_num: int) -> int:
    return u * node_num - u * (u + 1) // 2


def row_slice(u: int, node_num: int) -> slice:
    start = row_offset(u, node_num)
    return slice(start, start + node_num - u - 1)


def flat_pair_index(u: int, v: int, node_num: int) -> int:
    if u > v:
        u, v = v, u
    if u == v:
        raise ValueError("self-loop has no pair-cache index")
    return row_offset(u, node_num) + (v - u - 1)


def pair_cache_dir(args: argparse.Namespace) -> Path:
    if args.context_cache_dir is not None:
        return args.context_cache_dir
    return args.out_dir / "context_pair_cache"


def pair_cache_metadata_path(args: argparse.Namespace) -> Path:
    return pair_cache_dir(args) / "metadata.json"


def open_pair_cache(
    args: argparse.Namespace,
    node_num: int,
    ranges: PairScoreRanges | None = None,
    mode: str = "r+",
) -> PairScoreCache:
    cache_dir = pair_cache_dir(args)
    if mode != "r":
        cache_dir.mkdir(parents=True, exist_ok=True)
    pair_count = all_pair_count(node_num)

    def mmap(name: str, dtype) -> np.memmap:
        return np.memmap(cache_dir / name, mode=mode, dtype=dtype, shape=(pair_count,))

    if ranges is None:
        ranges = PairScoreRanges(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return PairScoreCache(
        node_num=node_num,
        pair_count=pair_count,
        mi_raw=mmap("mi_raw.float32.mmap", np.float32),
        persona_raw=mmap("persona_raw.float32.mmap", np.float32),
        edge=mmap("edge.uint8.mmap", np.uint8),
        score=mmap("score.float32.mmap", np.float32),
        ranges=ranges,
    )


def llm_cache_signature(args: argparse.Namespace, record_hash: str, node_num: int) -> dict:
    return {
        "node_num": int(node_num),
        "record_hash": record_hash,
        "mi_mode": int(args.mi_mode),
        "mi_weight": float(args.context_mi_weight),
        "persona_weight": float(args.context_persona_weight),
        "edge_weight": float(args.context_edge_weight),
        "persona_embedding_path": str(resolve_persona_embedding_path(args)),
        "edge_context_path": str(resolve_edge_context_path(args)),
        "disable_persona_context": bool(args.disable_persona_context),
        "disable_edge_context": bool(args.disable_edge_context),
    }


def metadata_matches(metadata: dict, signature: dict) -> bool:
    if not metadata or not metadata.get("complete"):
        return False
    old_signature = metadata.get("signature", {})
    return old_signature == signature


def load_pair_cache_if_valid(
    args: argparse.Namespace,
    node_num: int,
    record_hash: str,
) -> PairScoreCache | None:
    if args.force_shared_recompute or args.context_rebuild_cache:
        return None
    meta_path = pair_cache_metadata_path(args)
    if not meta_path.exists():
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    signature = llm_cache_signature(args, record_hash, node_num)
    if not metadata_matches(metadata, signature):
        return None

    pair_count = all_pair_count(node_num)
    cache_dir = pair_cache_dir(args)
    expected = {
        "mi_raw.float32.mmap": np.dtype(np.float32).itemsize,
        "persona_raw.float32.mmap": np.dtype(np.float32).itemsize,
        "edge.uint8.mmap": np.dtype(np.uint8).itemsize,
        "score.float32.mmap": np.dtype(np.float32).itemsize,
    }
    for name, item_size in expected.items():
        path = cache_dir / name
        if not path.exists() or path.stat().st_size != pair_count * item_size:
            return None

    ranges = PairScoreRanges(
        mi_min=float(metadata["mi_min"]),
        mi_max=float(metadata["mi_max"]),
        persona_min=float(metadata["persona_min"]),
        persona_max=float(metadata["persona_max"]),
        score_min=float(metadata["score_min"]),
        score_max=float(metadata["score_max"]),
    )
    print(f"loaded context pair cache: {cache_dir}", flush=True)
    return open_pair_cache(args, node_num, ranges=ranges, mode="r")


def write_pair_cache_metadata(
    args: argparse.Namespace,
    cache: PairScoreCache,
    record_hash: str,
) -> None:
    payload = {
        "complete": True,
        "mi_min": cache.ranges.mi_min,
        "mi_max": cache.ranges.mi_max,
        "persona_min": cache.ranges.persona_min,
        "persona_max": cache.ranges.persona_max,
        "score_min": cache.ranges.score_min,
        "score_max": cache.ranges.score_max,
        "signature": llm_cache_signature(args, record_hash, cache.node_num),
    }
    pair_cache_metadata_path(args).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def resolve_persona_embedding_path(args: argparse.Namespace) -> Path:
    if args.persona_embedding_path is not None:
        return args.persona_embedding_path
    root = args.diffusion_root if args.diffusion_root is not None else DEFAULT_LLM_DIFFUSION_ROOT
    return root / "profile_fields_embedding_Qwen_8B.npy"


def resolve_edge_context_path(args: argparse.Namespace) -> Path:
    if args.edge_context_path is not None:
        return args.edge_context_path
    root = args.diffusion_root if args.diffusion_root is not None else DEFAULT_LLM_DIFFUSION_ROOT
    return root / "edges.csv"


def l2_norms(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1).astype(np.float32, copy=False)
    norms[norms <= 1e-12] = 1.0
    return norms


def load_persona_embedding_context(
    args: argparse.Namespace,
    node_num: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    needs_embeddings = (
        not args.disable_persona_context and args.context_persona_weight > 0
    ) or (
        not args.disable_content_context and args.context_content_weight > 0
    )
    if not needs_embeddings:
        return None, None
    path = resolve_persona_embedding_path(args)
    if not path.exists():
        if args.require_context:
            raise FileNotFoundError(f"Persona embedding file not found: {path}")
        print(f"context persona embeddings unavailable; omitting persona/content: {path}", flush=True)
        return None, None
    embeddings = np.load(path, mmap_mode="r")
    if embeddings.shape[0] != node_num:
        raise ValueError(
            f"Persona embeddings rows={embeddings.shape[0]} do not match node_num={node_num}"
        )
    norms = l2_norms(embeddings)
    return embeddings, norms


def read_edge_pairs_csv(path: Path, node_num: int) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 2), dtype=np.int32)
    edges: set[tuple[int, int]] = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "src_u_id" in row and "dst_u_id" in row:
                u = int(row["src_u_id"])
                v = int(row["dst_u_id"])
            else:
                values = list(row.values())
                if len(values) < 2:
                    continue
                u = int(values[0])
                v = int(values[1])
            if u == v:
                continue
            if 0 <= u < node_num and 0 <= v < node_num:
                edges.add(tuple(sorted((u, v))))
    if not edges:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(sorted(edges), dtype=np.int32)


def edge_context_matrix(args: argparse.Namespace, node_num: int) -> sparse.csr_matrix:
    if args.disable_edge_context or args.context_edge_weight <= 0:
        return sparse.csr_matrix((node_num, node_num), dtype=np.uint8)
    path = resolve_edge_context_path(args)
    edge_array = read_edge_pairs_csv(path, node_num)
    if edge_array.size == 0:
        if args.require_context:
            raise FileNotFoundError(f"No usable context edges found: {path}")
        print(f"context edges unavailable; omitting P_edge: {path}", flush=True)
        return sparse.csr_matrix((node_num, node_num), dtype=np.uint8)
    rows = np.concatenate([edge_array[:, 0], edge_array[:, 1]])
    cols = np.concatenate([edge_array[:, 1], edge_array[:, 0]])
    data = np.ones(rows.shape[0], dtype=np.uint8)
    return sparse.csr_matrix((data, (rows, cols)), shape=(node_num, node_num))


def mutual_information_block(
    records: np.ndarray,
    col_sum: np.ndarray,
    beta: int,
    start: int,
    end: int,
    mode: int,
) -> np.ndarray:
    block_records = records[:, start:end]
    n11 = (block_records.T @ records).astype(np.float64, copy=False)
    n1_x = col_sum[start:end].reshape(end - start, 1)
    n0_x = beta - n1_x
    n1_y = col_sum.reshape(1, records.shape[1])
    n0_y = beta - n1_y
    n10 = n1_x - n11
    n01 = n1_y - n11
    n00 = beta - n1_x - n1_y + n11

    same = _mi_term(n11, n1_x, n1_y, beta) + _mi_term(n00, n0_x, n0_y, beta)
    term10 = _mi_term(n10, n1_x, n0_y, beta)
    term01 = _mi_term(n01, n0_x, n1_y, beta)
    if mode == 1:
        block = same - term10 - term01
    elif mode == 2:
        block = same - np.abs(term10) - np.abs(term01)
    else:
        block = same + term10 + term01
    for local_row in range(end - start):
        block[local_row, start + local_row] = 0.0
    return block.astype(np.float32, copy=False)


def persona_similarity_block(
    embeddings: np.ndarray | None,
    norms: np.ndarray | None,
    start: int,
    end: int,
    node_num: int,
) -> np.ndarray:
    if embeddings is None or norms is None:
        return np.zeros((end - start, node_num), dtype=np.float32)
    block = np.asarray(embeddings[start:end], dtype=np.float32)
    sims = block @ np.asarray(embeddings.T, dtype=np.float32)
    sims /= norms[start:end, None]
    sims /= norms[None, :]
    np.clip(sims, -1.0, 1.0, out=sims)
    for local_row in range(end - start):
        sims[local_row, start + local_row] = 0.0
    return sims.astype(np.float32, copy=False)


def build_context_raw_cache(
    records: np.ndarray,
    embeddings: np.ndarray | None,
    embedding_norms: np.ndarray | None,
    edge_matrix: sparse.csr_matrix,
    args: argparse.Namespace,
) -> tuple[PairScoreCache, float, float, float, float]:
    records = (records != 0).astype(np.float32, copy=False)
    beta, node_num = records.shape
    cache = open_pair_cache(args, node_num, mode="w+")
    col_sum = records.sum(axis=0, dtype=np.float64)
    mi_min, mi_max = math.inf, -math.inf
    persona_min, persona_max = math.inf, -math.inf

    block_starts = range(0, node_num, args.mi_block_size)
    total_blocks = (node_num + args.mi_block_size - 1) // args.mi_block_size
    if tqdm is not None:
        block_starts = tqdm(block_starts, desc="context raw cache", unit="block", total=total_blocks, mininterval=1.0)

    for start in block_starts:
        end = min(start + args.mi_block_size, node_num)
        mi = mutual_information_block(records, col_sum, beta, start, end, args.mi_mode)
        persona = persona_similarity_block(embeddings, embedding_norms, start, end, node_num)
        edge_block = edge_matrix[start:end].toarray().astype(np.uint8, copy=False)
        for local_row in range(end - start):
            u = start + local_row
            if u + 1 >= node_num:
                continue
            sl = row_slice(u, node_num)
            mi_values = mi[local_row, u + 1:]
            persona_values = persona[local_row, u + 1:]
            cache.mi_raw[sl] = mi_values
            cache.persona_raw[sl] = persona_values
            cache.edge[sl] = edge_block[local_row, u + 1:]
            mi_min, mi_max = finite_min_max(mi_values, mi_min, mi_max)
            persona_min, persona_max = finite_min_max(persona_values, persona_min, persona_max)

    if not np.isfinite(mi_min):
        mi_min = mi_max = 0.0
    if not np.isfinite(persona_min):
        persona_min = persona_max = 0.0
    cache.mi_raw.flush()
    cache.persona_raw.flush()
    cache.edge.flush()
    return cache, mi_min, mi_max, persona_min, persona_max


def build_context_score_cache(
    cache: PairScoreCache,
    ranges: PairScoreRanges,
    args: argparse.Namespace,
) -> tuple[float, float]:
    score_min, score_max = math.inf, -math.inf
    starts = range(0, cache.pair_count, args.context_cache_chunk_size)
    if tqdm is not None:
        starts = tqdm(starts, desc="context score cache", unit="chunk", mininterval=1.0)
    for start in starts:
        end = min(start + args.context_cache_chunk_size, cache.pair_count)
        mi_norm = minmax_normalize(cache.mi_raw[start:end], ranges.mi_min, ranges.mi_max)
        persona_norm = minmax_normalize(cache.persona_raw[start:end], ranges.persona_min, ranges.persona_max)
        edge = cache.edge[start:end].astype(np.float32, copy=False)
        score = (
            args.context_mi_weight * mi_norm
            + args.context_persona_weight * persona_norm
            + args.context_edge_weight * edge
        ).astype(np.float32, copy=False)
        cache.score[start:end] = score
        score_min, score_max = finite_min_max(score, score_min, score_max)
    if not np.isfinite(score_min):
        score_min = score_max = 0.0
    cache.score.flush()
    return score_min, score_max


def load_or_build_context_pair_cache(
    args: argparse.Namespace,
    record_data_list: list[np.ndarray],
    node_num: int,
) -> PairScoreCache:
    record_hash = records_fingerprint(record_data_list)
    cached = load_pair_cache_if_valid(args, node_num, record_hash)
    if cached is not None:
        return cached

    print("building context pair cache from MI/persona/edge signals...", flush=True)
    all_records = np.concatenate(record_data_list, axis=0)
    embeddings, embedding_norms = load_persona_embedding_context(args, node_num)
    if args.disable_persona_context or embeddings is None:
        persona_weight = args.context_persona_weight
        args.context_persona_weight = 0.0
    else:
        persona_weight = None
    edge_matrix = edge_context_matrix(args, node_num)
    if args.disable_edge_context or edge_matrix.nnz == 0:
        edge_weight = args.context_edge_weight
        args.context_edge_weight = 0.0
    else:
        edge_weight = None

    try:
        raw_embeddings = embeddings if args.context_persona_weight > 0 else None
        raw_embedding_norms = embedding_norms if raw_embeddings is not None else None
        cache, mi_min, mi_max, persona_min, persona_max = build_context_raw_cache(
            all_records,
            raw_embeddings,
            raw_embedding_norms,
            edge_matrix,
            args,
        )
        provisional = PairScoreRanges(mi_min, mi_max, persona_min, persona_max, 0.0, 0.0)
        score_min, score_max = build_context_score_cache(cache, provisional, args)
        cache.ranges = PairScoreRanges(mi_min, mi_max, persona_min, persona_max, score_min, score_max)
        write_pair_cache_metadata(args, cache, record_hash)
    finally:
        if persona_weight is not None:
            args.context_persona_weight = persona_weight
        if edge_weight is not None:
            args.context_edge_weight = edge_weight

    print(
        "context pair cache ready: "
        f"mi=({cache.ranges.mi_min:.6g},{cache.ranges.mi_max:.6g}) "
        f"persona=({cache.ranges.persona_min:.6g},{cache.ranges.persona_max:.6g}) "
        f"score=({cache.ranges.score_min:.6g},{cache.ranges.score_max:.6g})",
        flush=True,
    )
    return cache


def fit_pair_score_kmeans(cache: PairScoreCache, args: argparse.Namespace) -> np.ndarray:
    centers = np.asarray([cache.ranges.score_min, cache.ranges.score_max], dtype=np.float64)
    if abs(float(centers[1] - centers[0])) <= 1e-12:
        return centers
    for iteration in range(1, args.context_kmeans_iters + 1):
        sums = np.zeros(2, dtype=np.float64)
        counts = np.zeros(2, dtype=np.int64)
        starts = range(0, cache.pair_count, args.context_cache_chunk_size)
        if tqdm is not None:
            starts = tqdm(starts, desc=f"context kmeans {iteration}", unit="chunk", mininterval=1.0)
        for start in starts:
            end = min(start + args.context_cache_chunk_size, cache.pair_count)
            vals = np.asarray(cache.score[start:end], dtype=np.float64)
            eligible = vals >= args.global_score_threshold
            if not np.any(eligible):
                continue
            vals = vals[eligible]
            labels = np.argmin(np.abs(vals[:, None] - centers[None, :]), axis=1)
            for label in (0, 1):
                mask = labels == label
                if np.any(mask):
                    sums[label] += float(vals[mask].sum())
                    counts[label] += int(mask.sum())
        next_centers = centers.copy()
        for label in (0, 1):
            if counts[label] > 0:
                next_centers[label] = sums[label] / counts[label]
        print(
            f"context kmeans iter={iteration} centers={next_centers.tolist()} counts={counts.tolist()}",
            flush=True,
        )
        if np.allclose(next_centers, centers, atol=1e-8, rtol=0.0):
            centers = next_centers
            break
        centers = next_centers
    return centers


def select_initial_context_edges(
    cache: PairScoreCache,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    centers = fit_pair_score_kmeans(cache, args)
    edge_label = int(np.argmax(centers))
    max_edges = args.context_max_initial_edges
    heap: list[tuple[float, int, int]] = []
    chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []
    selected_count = 0

    row_iter = range(cache.node_num - 1)
    if tqdm is not None:
        row_iter = tqdm(row_iter, desc="select context edges", unit="row", mininterval=1.0)
    for u in row_iter:
        sl = row_slice(u, cache.node_num)
        score = np.asarray(cache.score[sl], dtype=np.float32)
        eligible = score >= args.global_score_threshold
        if not np.any(eligible):
            continue
        labels = np.argmin(np.abs(score[:, None] - centers[None, :]), axis=1)
        idx = np.flatnonzero(eligible & (labels == edge_label))
        if idx.size == 0:
            continue
        selected_count += int(idx.size)
        vs = idx + u + 1
        if max_edges is None or max_edges <= 0:
            us = np.full(idx.size, u, dtype=np.int32)
            chunks.append(np.column_stack([us, vs.astype(np.int32, copy=False)]))
            score_chunks.append(score[idx].astype(np.float32, copy=False))
            continue

        threshold = heap[0][0] if len(heap) >= max_edges else -math.inf
        if len(heap) >= max_edges:
            idx = idx[score[idx] > threshold]
            if idx.size == 0:
                continue
        if idx.size > max_edges:
            idx = idx[np.argpartition(score[idx], -max_edges)[-max_edges:]]
        for local_idx in idx:
            value = float(score[local_idx])
            v = int(u + 1 + local_idx)
            if len(heap) < max_edges:
                heapq.heappush(heap, (value, int(u), v))
            elif value > heap[0][0]:
                heapq.heapreplace(heap, (value, int(u), v))

    if max_edges is not None and max_edges > 0:
        rows = sorted(heap, key=lambda item: (-item[0], item[1], item[2]))
        edges = np.asarray([[u, v] for _score, u, v in rows], dtype=np.int32)
        scores = np.asarray([score for score, _u, _v in rows], dtype=np.float32)
    elif chunks:
        edges = np.vstack(chunks).astype(np.int32, copy=False)
        scores = np.concatenate(score_chunks).astype(np.float32, copy=False)
    else:
        edges = np.zeros((0, 2), dtype=np.int32)
        scores = np.zeros(0, dtype=np.float32)

    print(
        "context initial topology: "
        f"kmeans_centers={centers.tolist()} selected_before_cap={selected_count} "
        f"kept={edges.shape[0]} cap={max_edges}",
        flush=True,
    )
    return edges, scores


def save_sparse_edge_list(path: Path, edges: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    if edges.size == 0:
        path.write_text("", encoding="utf-8")
        return
    np.savetxt(path, edges, fmt="%d", delimiter="\t")


def edge_set_from_array(edges: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, edge)) for edge in np.asarray(edges, dtype=np.int32).reshape(-1, 2)}


def sparse_graph_log_sum(
    records: np.ndarray,
    edges: np.ndarray,
    p_values: np.ndarray,
    node_num: int,
) -> np.ndarray:
    if edges.size == 0:
        return np.zeros((records.shape[0], node_num), dtype=np.float64)
    p_values = np.clip(np.asarray(p_values, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    weights = np.log1p(-p_values)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([weights, weights])
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(node_num, node_num))
    return (records != 0).astype(np.float64, copy=False) @ matrix


def pair_scores_from_cache(cache: PairScoreCache, edges: np.ndarray) -> np.ndarray:
    scores = np.zeros(edges.shape[0], dtype=np.float32)
    for idx, (u, v) in enumerate(edges):
        scores[idx] = float(cache.score[flat_pair_index(int(u), int(v), cache.node_num)])
    return scores


def probability_from_context_score(score: float, args: argparse.Namespace) -> float:
    max_score = (
        args.context_score_weight
        * (args.context_mi_weight + args.context_persona_weight + args.context_edge_weight)
        + args.context_content_weight
    )
    if max_score <= 1e-12:
        return args.p_min
    normalized = min(1.0, max(0.0, score / max_score))
    return float(args.p_min + normalized * (args.p_init - args.p_min))


def load_topic_content_scores(
    root: Path,
    topic: str,
    embeddings: np.ndarray | None,
    embedding_norms: np.ndarray | None,
    args: argparse.Namespace,
    node_num: int,
) -> np.ndarray:
    if (
        args.disable_content_context
        or args.context_content_weight <= 0
        or embeddings is None
        or embedding_norms is None
    ):
        return np.zeros(node_num, dtype=np.float32)
    topic_dir = root / topic
    path = topic_dir / f"{topic}_news_content_embedding_content_Qwen_8B.npy"
    if not path.exists():
        if args.require_context:
            raise FileNotFoundError(f"Topic content embedding file not found: {path}")
        print(f"topic content embeddings unavailable for {topic}; omitting P_content: {path}", flush=True)
        return np.zeros(node_num, dtype=np.float32)
    news_embeddings = np.load(path, mmap_mode="r")
    if news_embeddings.ndim != 2:
        raise ValueError(f"{path} must be a 2-D embedding matrix, got {news_embeddings.shape}")
    if args.topic_content_mode == "first":
        topic_vec = np.asarray(news_embeddings[0], dtype=np.float32)
    else:
        topic_vec = np.asarray(news_embeddings[:], dtype=np.float32).mean(axis=0)
    topic_norm = float(np.linalg.norm(topic_vec))
    if topic_norm <= 1e-12:
        return np.zeros(node_num, dtype=np.float32)
    raw = (np.asarray(embeddings, dtype=np.float32) @ topic_vec) / (embedding_norms * topic_norm)
    raw = np.clip(raw, -1.0, 1.0).astype(np.float32, copy=False)
    return minmax_normalize(raw, float(raw.min()), float(raw.max()))


def combined_topic_scores(
    base_scores: np.ndarray,
    edges: np.ndarray,
    content_scores: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if edges.size == 0:
        return np.zeros(0, dtype=np.float32)
    avg_content = (content_scores[edges[:, 0]] + content_scores[edges[:, 1]]) * 0.5
    return (
        args.context_score_weight * base_scores
        + args.context_content_weight * avg_content
    ).astype(np.float32, copy=False)


def p_values_for_context_edges(
    edges: np.ndarray,
    base_scores: np.ndarray,
    content_scores: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    scores = combined_topic_scores(base_scores, edges, content_scores, args)
    return np.asarray([probability_from_context_score(float(score), args) for score in scores], dtype=np.float64)


def collect_context_add_candidates(
    cache: PairScoreCache,
    active_set: set[tuple[int, int]],
    content_scores: np.ndarray,
    args: argparse.Namespace,
) -> list[tuple[int, int, float, float]]:
    topm = args.context_add_topm
    if topm is None or topm <= 0:
        topm = max(1, len(active_set))
    heap: list[tuple[float, int, int, float]] = []
    node_num = cache.node_num
    row_iter = range(node_num - 1)
    if tqdm is not None:
        row_iter = tqdm(row_iter, desc="context add prefilter", unit="row", mininterval=1.0)
    for u in row_iter:
        sl = row_slice(u, node_num)
        base = np.asarray(cache.score[sl], dtype=np.float32)
        vs = np.arange(u + 1, node_num, dtype=np.int32)
        combined = (
            args.context_score_weight * base
            + args.context_content_weight * ((content_scores[u] + content_scores[vs]) * 0.5)
        )
        threshold = heap[0][0] if len(heap) >= topm else -math.inf
        eligible = combined > threshold
        if args.global_score_threshold > 0:
            eligible &= base >= args.global_score_threshold
        idx = np.flatnonzero(eligible)
        if idx.size == 0:
            continue
        row_keep = args.add_top_candidates_per_node if args.add_top_candidates_per_node > 0 else idx.size
        if idx.size > row_keep:
            idx = idx[np.argpartition(combined[idx], -row_keep)[-row_keep:]]
        for local_idx in idx:
            v = int(u + 1 + local_idx)
            edge = (int(u), v)
            if edge in active_set:
                continue
            score = float(combined[local_idx])
            base_score = float(base[local_idx])
            if len(heap) < topm:
                heapq.heappush(heap, (score, edge[0], edge[1], base_score))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, edge[0], edge[1], base_score))
    rows = [(u, v, score, base_score) for score, u, v, base_score in heap]
    rows.sort(key=lambda item: (-item[2], item[0], item[1]))
    return rows


def refine_context_edges(
    initial_edges: np.ndarray,
    initial_base_scores: np.ndarray,
    records: np.ndarray,
    cache: PairScoreCache,
    content_scores: np.ndarray,
    args: argparse.Namespace,
    topic: str,
) -> tuple[np.ndarray, np.ndarray, float, int, int, int]:
    edges = np.asarray(initial_edges, dtype=np.int32).reshape(-1, 2)
    base_scores = np.asarray(initial_base_scores, dtype=np.float32).reshape(-1)
    p_values = p_values_for_context_edges(edges, base_scores, content_scores, args)
    total_deleted = 0
    total_added = 0
    loss = 0.0
    used_iters = 0

    for refine_iter in range(1, args.refine_iters + 1):
        used_iters = refine_iter
        log_sum = sparse_graph_log_sum(records, edges, p_values, cache.node_num)
        delete_budget = edit_budget(args.max_delete_fraction, edges.shape[0])
        delete_moves = []
        for edge_idx, (u, v) in enumerate(edges):
            p = float(p_values[edge_idx])
            c_u_to_v = contribution(records, int(u), p)
            c_v_to_u = contribution(records, int(v), p)
            old_loss_v = column_loss(records, int(v), log_sum[:, v], args.p_init)
            new_loss_v = column_loss(records, int(v), log_sum[:, v] - c_u_to_v, args.p_init)
            old_loss_u = column_loss(records, int(u), log_sum[:, u], args.p_init)
            new_loss_u = column_loss(records, int(u), log_sum[:, u] - c_v_to_u, args.p_init)
            delta = (new_loss_v - old_loss_v) + (new_loss_u - old_loss_u)
            if delta > args.min_improvement:
                delete_moves.append((float(delta), int(edge_idx)))
        delete_moves.sort(key=lambda item: -item[0])
        deleted_indices = {idx for _delta, idx in delete_moves[:delete_budget]}
        if deleted_indices:
            keep = np.asarray([idx not in deleted_indices for idx in range(edges.shape[0])], dtype=bool)
            edges = edges[keep]
            base_scores = base_scores[keep]
            p_values = p_values[keep]

        add_budget = edit_budget(args.max_add_fraction, edges.shape[0])
        added = 0
        if add_budget > 0:
            active_set = edge_set_from_array(edges)
            log_sum = sparse_graph_log_sum(records, edges, p_values, cache.node_num)
            candidates = collect_context_add_candidates(cache, active_set, content_scores, args)
            add_moves = []
            for u, v, combined_score, base_score in candidates:
                p = probability_from_context_score(float(combined_score), args)
                c_u_to_v = contribution(records, int(u), p)
                c_v_to_u = contribution(records, int(v), p)
                old_loss_v = column_loss(records, int(v), log_sum[:, v], args.p_init)
                new_loss_v = column_loss(records, int(v), log_sum[:, v] + c_u_to_v, args.p_init)
                old_loss_u = column_loss(records, int(u), log_sum[:, u], args.p_init)
                new_loss_u = column_loss(records, int(u), log_sum[:, u] + c_v_to_u, args.p_init)
                delta = (new_loss_v - old_loss_v) + (new_loss_u - old_loss_u)
                if delta > args.min_improvement and delta <= args.add_max_loss_increase:
                    add_moves.append((float(delta), u, v, base_score, p))
            add_moves.sort(key=lambda item: -item[0])
            chosen = add_moves[:add_budget]
            if chosen:
                new_edges = np.asarray([[u, v] for _delta, u, v, _base, _p in chosen], dtype=np.int32)
                new_scores = np.asarray([base for _delta, _u, _v, base, _p in chosen], dtype=np.float32)
                new_p = np.asarray([p for _delta, _u, _v, _base, p in chosen], dtype=np.float64)
                edges = np.vstack([edges, new_edges])
                base_scores = np.concatenate([base_scores, new_scores])
                p_values = np.concatenate([p_values, new_p])
                added = new_edges.shape[0]

        loss = likelihood_loss_from_sum(
            records,
            sparse_graph_log_sum(records, edges, p_values, cache.node_num),
            args.p_init,
        )
        total_deleted += len(deleted_indices)
        total_added += added
        print(
            f"topic={topic} context_refine_iter={refine_iter} "
            f"edges={edges.shape[0]} loss={loss:.6f} "
            f"deleted={len(deleted_indices)} added={added}",
            flush=True,
        )
        if len(deleted_indices) == 0 and added == 0:
            break

    return edges, p_values, loss, total_deleted, total_added, used_iters


def load_llm_topic_records(
    topic_dir: Path,
    topic: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int]:
    sim_files = []
    pattern = re.compile(rf"^{re.escape(topic)}_\d+$")
    for path in sorted(topic_dir.iterdir()):
        if path.is_file() and pattern.match(path.name):
            sim_files.append(path)
    raw_count = len(sim_files)
    if args.records_per_task is not None:
        sim_files = sim_files[: args.records_per_task]
    if not sim_files:
        raise FileNotFoundError(f"No LLM diffusion pickle files found under {topic_dir}")

    records = []
    for sim_path in sim_files:
        payload = load_pickle(sim_path)
        mat = np.asarray(payload["test_dataset_all"])
        if mat.ndim != 2:
            raise ValueError(f"{sim_path} test_dataset_all must be 2-D, got {mat.shape}")
        if mat.shape[1] <= args.final_col:
            raise ValueError(f"{sim_path} has no final-state column {args.final_col}; shape={mat.shape}")
        records.append(mat[:, args.final_col])
    record_array = np.stack(records, axis=0).astype(np.int8)
    record_array[record_array != 0] = 1
    return record_array.astype(np.float32), raw_count


def run_llm_context_inference(args: argparse.Namespace) -> None:
    root = args.diffusion_root if args.diffusion_root is not None else DEFAULT_LLM_DIFFUSION_ROOT
    if not root.exists():
        raise FileNotFoundError(f"LLM diffusion root not found: {root}")
    args.diffusion_root = root
    args.out_dir.mkdir(parents=True, exist_ok=True)

    topics = args.topics or DEFAULT_LLM_TOPICS
    shared_record_data_list: list[np.ndarray] = []
    final_records_by_topic: dict[str, np.ndarray] = {}
    raw_counts: dict[str, int] = {}
    node_num: int | None = None

    for topic in topics:
        topic_dir = root / topic
        records, raw_count = load_llm_topic_records(topic_dir, topic, args)
        if node_num is None:
            node_num = records.shape[1]
        elif records.shape[1] != node_num:
            raise ValueError(f"{topic} node count {records.shape[1]} != expected {node_num}")
        shared_record_data_list.append(records)
        final_records_by_topic[topic] = records
        raw_counts[topic] = raw_count
        print(
            f"loaded LLM topic={topic} records={records.shape[0]}/{raw_count} nodes={records.shape[1]}",
            flush=True,
        )

    if node_num is None:
        raise ValueError("No LLM topics loaded")

    cache = load_or_build_context_pair_cache(args, shared_record_data_list, node_num)
    initial_edges, initial_scores = select_initial_context_edges(cache, args)
    save_sparse_edge_list(args.out_dir / "context_initial_topology_edges.txt", initial_edges)

    embeddings, embedding_norms = load_persona_embedding_context(args, node_num)
    print(
        f"context initial topology edges={initial_edges.shape[0]}",
        flush=True,
    )

    rows = []
    begin = time.time()
    for topic in topics:
        content_scores = load_topic_content_scores(
            root,
            topic,
            embeddings,
            embedding_norms,
            args,
            node_num,
        )
        edges, p_values, loss, deleted, added, used_refine_iters = refine_context_edges(
            initial_edges,
            initial_scores,
            final_records_by_topic[topic],
            cache,
            content_scores,
            args,
            topic,
        )
        save_sparse_edge_list(args.out_dir / f"{topic}_graph.txt", edges)
        np.save(args.out_dir / f"{topic}_p.npy", p_values)
        print(
            f"topic={topic} context_topology edges={edges.shape[0]} "
            f"loss={loss:.6f} deleted={deleted} added={added}",
            flush=True,
        )
        rows.append(
            {
                "topic": topic,
                "raw_records": raw_counts[topic],
                "records": final_records_by_topic[topic].shape[0],
                "nodes": node_num,
                "initial_edges": initial_edges.shape[0],
                "edges": edges.shape[0],
                "loss": loss,
                "deleted": deleted,
                "added": added,
                "refine_iters": used_refine_iters,
            }
        )

    with (args.out_dir / "topology_summary.csv").open("w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"LLM context inference done in {time.time() - begin:.2f}s; outputs written to {args.out_dir}")


def load_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def symmetrize_graph(graph: np.ndarray) -> np.ndarray:
    undirected = ((graph != 0) | (graph.T != 0)).astype(np.float32)
    np.fill_diagonal(undirected, 0)
    return undirected


def undirected_edge_count(graph: np.ndarray) -> int:
    undirected = (graph != 0) | (graph.T != 0)
    return int(np.count_nonzero(np.triu(undirected, k=1)))


def undirected_edge_array(graph: np.ndarray) -> np.ndarray:
    undirected = (graph != 0) | (graph.T != 0)
    return np.argwhere(np.triu(undirected, k=1))


def adjacency_to_dense(adj_mat) -> np.ndarray:
    if sparse.issparse(adj_mat):
        graph = adj_mat.toarray().astype(np.float32)
    else:
        graph = np.asarray(adj_mat, dtype=np.float32)
    graph[graph != 0] = 1.0
    return symmetrize_graph(graph)


def find_adjacency_file(task_dir: Path) -> Path | None:
    candidates = sorted(task_dir.glob("*_adj_mat.pkl"))
    if candidates:
        return candidates[0]
    return None


def list_simulation_files(task_dir: Path) -> list[Path]:
    files = []
    for path in sorted(task_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name.endswith("_adj_mat.pkl"):
            continue
        if path.suffix.lower() in {".json", ".jsonl", ".txt", ".csv", ".log", ".npy"}:
            continue
        files.append(path)
    return files


def diffusion_file_limit(args: argparse.Namespace) -> int | None:
    return args.records_per_task


def load_task_records(task_dir: Path, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    sim_files = list_simulation_files(task_dir)
    if not sim_files:
        raise FileNotFoundError(f"No simulation pickle files found under {task_dir}")

    adj_path = find_adjacency_file(task_dir)
    if adj_path is not None:
        graph = adjacency_to_dense(load_pickle(adj_path)["adj_mat"])
    else:
        first_payload = load_pickle(sim_files[0])
        if "adj_mat" not in first_payload:
            raise FileNotFoundError(
                f"No *_adj_mat.pkl found under {task_dir}, and simulation file has no adj_mat fallback"
            )
        graph = adjacency_to_dense(first_payload["adj_mat"])

    records: list[np.ndarray] = []
    max_files = diffusion_file_limit(args)
    if max_files is not None:
        sim_files = sim_files[:max_files]

    for sim_path in sim_files:
        payload = load_pickle(sim_path)
        if "test_dataset_all" not in payload:
            continue
        mat = np.asarray(payload["test_dataset_all"])
        if mat.ndim != 2:
            raise ValueError(f"{sim_path} test_dataset_all must be 2-D, got shape={mat.shape}")

        if mat.shape[1] <= args.final_col:
            raise ValueError(
                f"{sim_path} has no final-state column {args.final_col}; shape={mat.shape}"
            )
        records.append(mat[:, args.final_col])

    if not records:
        raise ValueError(f"No records loaded from {task_dir}")
    record_array = np.stack(records, axis=0).astype(np.int8)
    record_array[record_array != 0] = 1
    return record_array.astype(np.float32), graph


def collect_combo_tasks(
    diffusion_root: Path,
    dataset_name: str,
    mode: str,
    ratio_label: str,
    max_topologies: int | None,
) -> list[tuple[str, Path]]:
    topology_id = format_topology_id(dataset_name)
    combo_dir = diffusion_root / topology_id / f"{topology_id}_{mode}" / f"{topology_id}_{mode}_{ratio_label}"
    if not combo_dir.exists():
        raise FileNotFoundError(f"Combination directory not found: {combo_dir}")

    task_dirs = []
    for path in sorted(combo_dir.iterdir()):
        if not path.is_dir():
            continue
        if find_adjacency_file(path) is None and not list_simulation_files(path):
            continue
        task_dirs.append((path.name, path))
    if max_topologies is not None:
        task_dirs = task_dirs[:max_topologies]
    if not task_dirs:
        raise FileNotFoundError(f"No topology task directories found under {combo_dir}")
    return task_dirs


def run_inference_for_tasks(
    args: argparse.Namespace,
    shared_record_data_list: list[np.ndarray],
    final_record_data_list: list[np.ndarray],
    graph_label_list: list[np.ndarray],
    task_names: list[str],
    raw_record_counts: list[int] | None = None,
) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)

    begin = time.time()
    node_num = shared_record_data_list[0].shape[1]
    for task, final_records in enumerate(final_record_data_list):
        if final_records.shape[1] != node_num:
            raise ValueError(
                f"task={task} final records node count {final_records.shape[1]} "
                f"does not match shared records node count {node_num}"
            )

    mi, threshold, shared = load_or_build_shared_topology(args, shared_record_data_list, node_num)
    save_edge_list(args.out_dir / "shared_topology_edges.txt", shared)
    print(f"shared topology: threshold={threshold:.8f}, edges={undirected_edge_count(shared)}", flush=True)

    rows = []
    for task, records in enumerate(final_record_data_list):
        raw_record_count = raw_record_counts[task] if raw_record_counts is not None else records.shape[0]
        shared_record_metrics = per_record_f1(graph_label_list[task], shared, records)
        shared_metric_summary = summarize_record_metrics(shared_record_metrics)
        shared_precision = shared_metric_summary["precision_mean"]
        shared_recall = shared_metric_summary["recall_mean"]
        shared_f1 = shared_metric_summary["f1_mean"]
        shared_p = probability_matrix_from_mi(shared, mi, args.p_min, args.p_init)
        shared_loss = likelihood_loss_from_sum(
            records,
            graph_log_sum(shared, shared_p, records),
            args.p_init,
        )
        print(
            f"task={task} name={task_names[task]} shared precision={shared_precision:.6f} "
            f"recall={shared_recall:.6f} f1={shared_f1:.6f} "
            f"edges={undirected_edge_count(shared)} loss={shared_loss:.6f}",
            flush=True,
        )
        (
            refined,
            p_matrix,
            loss,
            deleted,
            added,
            used_refine_iters,
        ) = refine_graph_by_likelihood(
            shared=shared,
            mi=mi,
            records=records,
            p_init=args.p_init,
            p_min=args.p_min,
            min_improvement=args.min_improvement,
            max_delete_fraction=args.max_delete_fraction,
            max_add_fraction=args.max_add_fraction,
            delete_mi_threshold=args.delete_mi_threshold,
            add_top_candidates_per_node=args.add_top_candidates_per_node,
            add_max_loss_increase=args.add_max_loss_increase,
            refine_iters=args.refine_iters,
            task=task,
        )
        refined_record_metrics = per_record_f1(graph_label_list[task], refined, records)
        refined_metric_summary = summarize_record_metrics(refined_record_metrics)
        precision = refined_metric_summary["precision_mean"]
        recall = refined_metric_summary["recall_mean"]
        f1 = refined_metric_summary["f1_mean"]
        print(
            f"task={task} name={task_names[task]} final_refined precision={precision:.6f} "
            f"recall={recall:.6f} f1={f1:.6f} edges={undirected_edge_count(refined)} "
            f"loss={loss:.6f} deleted={deleted} added={added} "
            f"refine_iters={used_refine_iters}",
            flush=True,
        )

        np.save(args.out_dir / f"task_{task}_{task_names[task]}_p.npy", p_matrix)
        save_edge_list(args.out_dir / f"task_{task}_{task_names[task]}_graph.txt", refined)
        rows.append(
            {
                "task": task,
                "task_name": task_names[task],
                "raw_records": raw_record_count,
                "records": records.shape[0],
                "shared_precision": shared_precision,
                "shared_recall": shared_recall,
                "shared_f1": shared_f1,
                "shared_edges": undirected_edge_count(shared),
                "shared_loss": shared_loss,
                "precision": refined_metric_summary["precision_mean"],
                "recall": refined_metric_summary["recall_mean"],
                "f1": refined_metric_summary["f1_mean"],
                "edges": undirected_edge_count(refined),
                "loss": loss,
                "deleted": deleted,
                "added": added,
                "refine_iters": used_refine_iters,
            }
        )

    with (args.out_dir / "metrics.csv").open("w", newline="") as f:
        fieldnames = [
            "task",
            "task_name",
            "raw_records",
            "records",
            "shared_precision",
            "shared_recall",
            "shared_f1",
            "shared_edges",
            "shared_loss",
            "precision",
            "recall",
            "f1",
            "edges",
            "loss",
            "deleted",
            "added",
            "refine_iters",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"done in {time.time() - begin:.2f}s; outputs written to {args.out_dir}")


def resolve_diffusion_root(args: argparse.Namespace, dataset_name: str, mode: str) -> Path:
    if args.diffusion_root is not None:
        return args.diffusion_root
    key = (format_topology_id(dataset_name), mode.upper())
    try:
        return DEFAULT_DIFFUSION_ROOTS[key]
    except KeyError as exc:
        known = ", ".join(f"{dataset}/{diff_mode}" for dataset, diff_mode in sorted(DEFAULT_DIFFUSION_ROOTS))
        raise ValueError(
            f"No default diffusion root for dataset={dataset_name!r}, mode={mode!r}. "
            f"Use --diffusion-root or choose one of: {known}"
        ) from exc



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diffusion-root",
        type=Path,
        default=None,
        help="Optional override. By default LLM mode uses Dataset-diffusion/LLM-diffusion relative to VF-SL.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional override. By default LLM mode writes to Dataset-Graph-infer/graph-infer-llm-context relative to VF-SL.",
    )
    parser.add_argument(
        "--dataset-format",
        choices=["classical", "llm"],
        default="classical",
        help="Select the input layout explicitly: classical simulated-diffusion or LLM-diffusion topic data.",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=DEFAULT_LLM_TOPICS,
        help="LLM-diffusion topic directories to process.",
    )
    parser.add_argument(
        "--dataset",
        "--datasets",
        dest="dataset",
        default="power_grid",
        help="Classical dataset name. Only one dataset is run per invocation.",
    )
    parser.add_argument(
        "--mode",
        "--modes",
        dest="mode",
        default="IC",
        help="Classical diffusion mode. Only one mode is run per invocation.",
    )
    parser.add_argument(
        "--source-ratio",
        "--source-ratios",
        dest="source_ratio",
        default="1%",
        help="Classical source ratio. Only one ratio is run per invocation.",
    )

    parser.add_argument("--max-topologies", type=int, default=None)
    parser.add_argument(
        "--records-per-task",
        type=int,
        default=300,
        help="Use only the first N complete diffusion files for each task.",
    )
    parser.add_argument("--final-col", type=int, default=1)
    parser.add_argument("--mi-mode", type=int, default=2)
    parser.add_argument("--mi-block-size", type=int, default=512)
    parser.add_argument("--mi-threshold-scale", type=float, default=1.0)
    parser.add_argument(
        "--global-score-threshold",
        type=float,
        default=0.0,
        help="Minimum context pair score eligible for KMeans edge selection in LLM mode.",
    )
    parser.add_argument("--context-mi-weight", type=float, default=0.34)
    parser.add_argument("--context-persona-weight", type=float, default=0.33)
    parser.add_argument("--context-edge-weight", type=float, default=0.33)
    parser.add_argument("--context-content-weight", type=float, default=0.2)
    parser.add_argument("--context-score-weight", type=float, default=1.0)
    parser.add_argument("--context-kmeans-iters", type=int, default=10)
    parser.add_argument("--context-cache-chunk-size", type=int, default=5_000_000)
    parser.add_argument(
        "--context-max-initial-edges",
        type=int,
        default=300_000,
        help="Cap LLM initial topology edges after KMeans by keeping highest context scores; <=0 keeps all.",
    )
    parser.add_argument(
        "--context-add-topm",
        type=int,
        default=100_000,
        help="Global heap size for LLM context add-candidate prefilter; <=0 uses current edge count.",
    )
    parser.add_argument("--context-cache-dir", type=Path, default=None)
    parser.add_argument("--context-rebuild-cache", action="store_true")
    parser.add_argument("--persona-embedding-path", type=Path, default=None)
    parser.add_argument("--edge-context-path", type=Path, default=None)
    parser.add_argument("--disable-persona-context", action="store_true")
    parser.add_argument("--disable-edge-context", action="store_true")
    parser.add_argument("--disable-content-context", action="store_true")
    parser.add_argument("--require-context", action="store_true")
    parser.add_argument("--topic-content-mode", choices=["first", "mean"], default="mean")
    parser.add_argument("--shared-top-in-per-node", type=int, default=None)
    parser.add_argument(
        "--p-init",
        type=float,
        default=0.05,
        help="Maximum edge probability after MI normalization.",
    )
    parser.add_argument(
        "--p-min",
        type=float,
        default=1e-6,
        help="Minimum edge probability after MI normalization.",
    )
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--max-delete-fraction", type=float, default=0.01)
    parser.add_argument(
        "--delete-mi-threshold",
        type=float,
        default=1e-12,
        help="Also delete existing edges whose MI score is at or below this threshold.",
    )
    parser.add_argument("--max-add-fraction", type=float, default=0.01)
    parser.add_argument("--add-top-candidates-per-node", type=int, default=50)
    parser.add_argument("--add-max-loss-increase", type=float, default=float("inf"))
    parser.add_argument("--refine-iters", type=int, default=8)
    parser.add_argument("--force-shared-recompute", action="store_true")
    args = parser.parse_args()
    if args.records_per_task is not None and args.records_per_task <= 0:
        parser.error("--records-per-task must be positive when set")
    if args.final_col < 0:
        parser.error("--final-col must be non-negative")
    if args.mi_block_size <= 0:
        parser.error("--mi-block-size must be positive")
    if args.context_cache_chunk_size <= 0:
        parser.error("--context-cache-chunk-size must be positive")
    if args.context_kmeans_iters <= 0:
        parser.error("--context-kmeans-iters must be positive")
    for name in [
        "context_mi_weight",
        "context_persona_weight",
        "context_edge_weight",
        "context_content_weight",
        "context_score_weight",
    ]:
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if not 0 < args.p_init < 1:
        parser.error("--p-init is used as max normalized edge probability and must be in (0, 1)")
    if not 0 <= args.p_min < args.p_init:
        parser.error("--p-min must satisfy 0 <= p-min < p-init")
    return args


def main() -> int:
    args = parse_args()
    use_llm = args.dataset_format == "llm"
    if use_llm:
        if args.diffusion_root is None:
            args.diffusion_root = DEFAULT_LLM_DIFFUSION_ROOT
        if args.out_dir is None:
            args.out_dir = DEFAULT_LLM_OUTPUT_ROOT
    elif args.out_dir is None:
        args.out_dir = DEFAULT_OUTPUT_ROOT

    base_out_dir = args.out_dir
    base_out_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(base_out_dir)
    print(f"record_mode=final", flush=True)
    print(f"args={vars(args)}", flush=True)

    if use_llm:
        print(
            f"\n=== LLM context topology inference diffusion_root={args.diffusion_root} ===",
            flush=True,
        )
        run_llm_context_inference(args)
        print(f"log written to {log_path}")
        return 0

    dataset_name = args.dataset
    topology_id = format_topology_id(dataset_name)
    mode = args.mode.upper()
    ratio_label = parse_ratio_label(args.source_ratio)
    combo_diffusion_root = resolve_diffusion_root(args, dataset_name, mode)
    print(
        f"\n=== dataset={dataset_name} mode={mode} ratio={ratio_label} "
        f"record_mode=final "
        f"diffusion_root={combo_diffusion_root} ===",
        flush=True,
    )
    task_items = collect_combo_tasks(
        diffusion_root=combo_diffusion_root,
        dataset_name=dataset_name,
        mode=mode,
        ratio_label=ratio_label,
        max_topologies=args.max_topologies,
    )
    shared_record_data_list = []
    final_record_data_list = []
    graph_label_list = []
    task_names = []
    raw_record_counts = []
    for task_name, task_dir in task_items:
        records, graph = load_task_records(task_dir, args)
        loaded_record_count = records.shape[0]
        shared_record_data_list.append(records)
        final_record_data_list.append(records)
        graph_label_list.append(graph)
        task_names.append(task_name)
        raw_record_counts.append(loaded_record_count)
        print(
            f"loaded task={task_name} "
            f"diffusion_file_limit={diffusion_file_limit(args)} "
            f"records={records.shape} "
            f"graph={graph.shape} from {task_dir}",
            flush=True,
        )

    combo_out = base_out_dir / topology_id / mode / ratio_label / "final"
    old_out = args.out_dir
    args.out_dir = combo_out
    run_inference_for_tasks(
        args,
        shared_record_data_list,
        final_record_data_list,
        graph_label_list,
        task_names,
        raw_record_counts=raw_record_counts,
    )
    args.out_dir = old_out

    print(f"log written to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Classical W/O context
# python Topology_Inference.py \
#   --dataset karate \
#   --mode IC \
#   --source-ratio 1% \
#   --records-per-task 300

# LLM WITH context
# python Topology_Inference.py \
# --dataset-format llm