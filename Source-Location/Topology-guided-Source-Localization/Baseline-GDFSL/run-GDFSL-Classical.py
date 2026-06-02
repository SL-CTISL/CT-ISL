import importlib.util
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SCRIPT_DIR = Path(__file__).resolve().parent
TGS_ROOT = SCRIPT_DIR.parent
CT_ISL_DIR = TGS_ROOT / "CT-ISL"
GDFSL_CODE_DIR = SCRIPT_DIR / "code"
BASE_RUNNER_PATH = CT_ISL_DIR / "run-CT-ISL-Classical.py"

for path in (GDFSL_CODE_DIR, TGS_ROOT, CT_ISL_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from utils import Metric
from project_paths import ensure_within_vfsl, rel_to_vfsl, resolve_vfsl_path


def _load_gdfsl_model_module():
    spec = importlib.util.spec_from_file_location("gdfsl2025_model", GDFSL_CODE_DIR / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gdfsl_model = _load_gdfsl_model_module()
GaussianDiffusionForwardTrainer_future_un = gdfsl_model.GaussianDiffusionForwardTrainer_future_un
GaussianDiffusionSampler_un = gdfsl_model.GaussianDiffusionSampler_un


def _load_base_runner():
    spec = importlib.util.spec_from_file_location("graphsl_classical_base_runner", BASE_RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base_runner = _load_base_runner()

CKPT_OUTPUT_ROOT = ensure_within_vfsl(SCRIPT_DIR / "ckpt&outputs")


def resolve_checkpoint_path(checkpoint_path):
    path = Path(checkpoint_path)
    if path.is_absolute():
        return resolve_vfsl_path(path)

    for base in (CKPT_OUTPUT_ROOT, SCRIPT_DIR):
        candidate = base / path
        if candidate.exists():
            return ensure_within_vfsl(candidate)

    if path.parts and path.parts[0] in {"ckpt&outputs", "classical-gdfsl2025"}:
        base = SCRIPT_DIR if path.parts[0] == "ckpt&outputs" else CKPT_OUTPUT_ROOT
        return ensure_within_vfsl(base / path)

    return resolve_vfsl_path(path)


@dataclass
class GDFSLRecord:
    adj: torch.Tensor
    labels: torch.Tensor
    observation: torch.Tensor
    features: torch.Tensor
    node_idx: np.ndarray
    full_labels: np.ndarray


def _normalize_columns(arr):
    arr = arr.astype(np.float32, copy=False)
    min_vals = arr.min(axis=0, keepdims=True)
    max_vals = arr.max(axis=0, keepdims=True)
    denom = max_vals - min_vals
    denom[denom < 1e-6] = 1.0
    return (arr - min_vals) / denom


def _pad_to_dim(arr, dim):
    if arr.shape[1] >= dim:
        return arr[:, :dim].astype(np.float32, copy=False)
    padded = np.zeros((arr.shape[0], dim), dtype=np.float32)
    padded[:, :arr.shape[1]] = arr
    return padded


def _graph_feature_matrix(adj):
    csr = adj.tocsr().astype(np.float32, copy=True)
    csr.setdiag(0)
    csr.eliminate_zeros()

    degree = np.asarray(csr.sum(axis=1)).reshape(-1).astype(np.float32, copy=False)
    max_degree = max(float(degree.max()) if degree.size else 0.0, 1.0)
    neighbor_degree_sum = np.asarray(csr @ degree).reshape(-1).astype(np.float32, copy=False)
    neighbor_degree_mean = np.divide(
        neighbor_degree_sum,
        degree,
        out=np.zeros_like(neighbor_degree_sum, dtype=np.float32),
        where=degree > 0,
    )
    second_hop_mean = np.asarray(csr @ neighbor_degree_mean).reshape(-1).astype(np.float32, copy=False)
    second_hop_mean = np.divide(
        second_hop_mean,
        degree,
        out=np.zeros_like(second_hop_mean, dtype=np.float32),
        where=degree > 0,
    )
    node_index = np.linspace(0.0, 1.0, num=csr.shape[0], dtype=np.float32)

    features = np.column_stack(
        [
            degree,
            np.log1p(degree),
            np.sqrt(degree),
            degree / max_degree,
            neighbor_degree_mean,
            second_hop_mean,
            node_index,
        ]
    )
    return _normalize_columns(features)


def _unwrap_dataset(loaded):
    if isinstance(loaded, tuple):
        return loaded[0]
    return loaded


def _candidate_indices(adj, observation, max_nodes, rng):
    observed = np.flatnonzero(observation > 0.5).astype(np.int64)
    if observed.size == 0:
        observed = np.array([int(np.argmax(observation))], dtype=np.int64)

    if observed.size >= max_nodes:
        return observed

    remaining = max_nodes - observed.size
    observed_set = set(observed.tolist())

    neighbor_pool = adj[observed].nonzero()[1].astype(np.int64)
    if neighbor_pool.size:
        neighbor_pool = np.unique(neighbor_pool)
        neighbor_pool = np.array(
            [node for node in neighbor_pool if node not in observed_set],
            dtype=np.int64,
        )

    selected = []
    if neighbor_pool.size:
        take = min(remaining, neighbor_pool.size)
        selected.extend(rng.choice(neighbor_pool, size=take, replace=False).tolist())
        remaining -= take

    if remaining > 0:
        selected_set = observed_set | set(selected)
        pool = np.setdiff1d(
            np.arange(observation.shape[0], dtype=np.int64),
            np.fromiter(selected_set, dtype=np.int64),
        )
        if pool.size:
            take = min(remaining, pool.size)
            selected.extend(rng.choice(pool, size=take, replace=False).tolist())

    if not selected:
        return observed
    return np.concatenate([observed, np.asarray(selected, dtype=np.int64)])


def _build_features(graph_features, influ_mat, node_idx):
    structural = graph_features[node_idx].astype(np.float32, copy=False)
    final_infection = influ_mat[node_idx, 1:2].detach().cpu().numpy()
    infection_context = _pad_to_dim(final_infection, 7)
    return np.concatenate([structural, infection_context], axis=1).astype(np.float32, copy=False)


def _build_records(adj, dataset_subset, graph_features, max_nodes, random_seed):
    rng = np.random.default_rng(random_seed)
    records = []
    for sample_idx, influ_mat in enumerate(dataset_subset):
        if torch.is_tensor(influ_mat):
            mat = influ_mat.detach().cpu()
        else:
            mat = torch.as_tensor(influ_mat)

        labels_full = mat[:, 0].numpy().astype(np.float32, copy=False)
        observation_full = mat[:, 1].numpy().astype(np.float32, copy=False)
        node_idx = _candidate_indices(adj, observation_full, max_nodes=max_nodes, rng=rng)

        sub_adj = adj[node_idx][:, node_idx].toarray().astype(np.float32, copy=False)
        np.fill_diagonal(sub_adj, 0.0)
        features = _build_features(graph_features, mat, node_idx)

        records.append(
            GDFSLRecord(
                adj=torch.from_numpy(sub_adj),
                labels=torch.from_numpy(labels_full[node_idx].astype(np.float32, copy=False)),
                observation=torch.from_numpy(observation_full[node_idx].astype(np.float32, copy=False)),
                features=torch.from_numpy(features),
                node_idx=node_idx,
                full_labels=labels_full,
            )
        )
        if (sample_idx + 1) % 50 == 0 or sample_idx + 1 == len(dataset_subset):
            print(f"Prepared GDFSL records: {sample_idx + 1}/{len(dataset_subset)}")
    return records


def _full_scores(record, candidate_scores):
    scores = np.zeros(record.full_labels.shape[0], dtype=np.float32)
    scores[record.node_idx] = candidate_scores.astype(np.float32, copy=False)
    return scores


def _predict_top_k(labels, scores, top_k=None):
    if top_k is None:
        k = int(np.sum(labels))
    else:
        k = int(top_k)
    k = max(0, min(k, scores.shape[0]))

    pred = np.zeros(labels.shape[0], dtype=bool)
    if k > 0:
        top_idx = np.argsort(-scores, kind="mergesort")[:k]
        pred[top_idx] = True
    return pred


def _metric_from_top_k(records, score_list, top_k=None):
    acc = pr = re = f1 = auc = 0.0
    valid_auc = 0
    preds = []

    for record, scores in zip(records, score_list):
        labels = record.full_labels
        pred = _predict_top_k(labels, scores, top_k=top_k)
        preds.append(pred)
        acc += accuracy_score(labels, pred)
        pr += precision_score(labels, pred, zero_division=0)
        re += recall_score(labels, pred, zero_division=0)
        f1 += f1_score(labels, pred, zero_division=0)
        if np.unique(labels).size > 1:
            auc += roc_auc_score(labels, scores)
            valid_auc += 1

    count = max(len(records), 1)
    auc = auc / valid_auc if valid_auc > 0 else 0.0
    return Metric(acc / count, pr / count, re / count, f1 / count, auc), preds


class GDFSL2025Classical:
    def __init__(
            self,
            epochs=60,
            lr=5e-4,
            beta_1=0.0001,
            beta_T=0.01,
            T=100,
            feature_hidden_dim=64,
            max_nodes=512,
            num_thres=50,
            top_k=None,
            sample_times=1,
            infection_rate=0.1,
            random_seed=0):
        self.epochs = epochs
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T
        self.feature_hidden_dim = feature_hidden_dim
        self.max_nodes = max_nodes
        self.num_thres = num_thres
        self.top_k = top_k
        self.sample_times = sample_times
        self.infection_rate = infection_rate
        self.random_seed = random_seed
        self.t_start = int(1 / 10 * T)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.forward_trainer = GaussianDiffusionForwardTrainer_future_un(
            t_start=self.t_start,
            beta_1=beta_1,
            beta_T=beta_T,
            T=T,
            feature_hidden_dim=feature_hidden_dim,
        ).to(self.device)
        self.sampler = GaussianDiffusionSampler_un(
            model=self.forward_trainer,
            beta_1=beta_1,
            beta_T=beta_T,
            T=T,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.forward_trainer.parameters(), lr=lr)

    def load_state_dict(self, state_dict):
        self.forward_trainer.load_state_dict(state_dict)

    def _record_to_device(self, record):
        return (
            record.adj.to(self.device),
            record.labels.to(self.device),
            record.observation.to(self.device),
            record.features.to(self.device),
        )

    def _train_one_record(self, record):
        graph_topo, _, observation, features = self._record_to_device(record)
        observation = observation.unsqueeze(-1).float()
        features = features.float()
        final_miu = torch.ones_like(observation, device=self.device) + observation
        t_interval = int(torch.randint(self.T - self.t_start, size=(1,), device=self.device).item())

        x_t, noise = self.forward_trainer.forward_model_IC(
            observation,
            t_interval,
            final_miu,
            graph_topo,
            infection_rate=self.infection_rate,
        )
        pred_noise = self.forward_trainer.model(
            x_t.float(),
            t_interval,
            features[:, 7:].float(),
            graph_topo.float(),
            features.float(),
        ).float()
        return F.mse_loss(pred_noise, noise.float())

    def _predict_records(self, records):
        self.forward_trainer.eval()
        self.sampler.eval()
        score_list = []

        with torch.no_grad():
            for record in records:
                graph_topo, _, observation, features = self._record_to_device(record)
                observation = observation.unsqueeze(-1).float()
                features = features.float()
                final_miu = torch.ones_like(observation, device=self.device) + observation
                sample_scores = torch.zeros_like(observation)

                for _ in range(self.sample_times):
                    x_T = torch.randn_like(final_miu, device=self.device) + final_miu
                    pred_x_0 = self.sampler(
                        x_T,
                        final_miu.clone(),
                        features[:, 7:].float(),
                        graph_topo.float(),
                        features.float(),
                        observation=observation,
                    )
                    sample_scores += pred_x_0.float()

                candidate_scores = (sample_scores / self.sample_times).squeeze(-1).detach().cpu().numpy()
                score_list.append(_full_scores(record, candidate_scores))

        return score_list

    def train(self, adj, train_dataset_all, graph_features):
        if self.device.type != "cuda":
            raise RuntimeError("GDFSL implementation uses CUDA-only operations; run with a visible GPU.")

        train_records = _build_records(
            adj,
            train_dataset_all,
            graph_features,
            max_nodes=self.max_nodes,
            random_seed=self.random_seed,
        )

        print("train GDFSL2025:")
        for epoch in range(self.epochs):
            self.forward_trainer.train()
            total_loss = 0.0
            for record in train_records:
                self.optimizer.zero_grad()
                loss = self._train_one_record(record)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(len(train_records), 1)
            print(f"Epoch [{epoch + 1}/{self.epochs}], loss = {avg_loss:.6f}")

        train_scores = self._predict_records(train_records)
        train_metric, _ = _metric_from_top_k(train_records, train_scores, top_k=self.top_k)
        top_k_label = self.top_k if self.top_k is not None else "label_count"
        print(f"GDFSL2025 eval mode = top-k ({top_k_label})")
        return self.top_k, train_metric.auc, train_metric.f1, train_scores

    def test(self, adj, test_dataset_all, graph_features, top_k):
        test_records = _build_records(
            adj,
            test_dataset_all,
            graph_features,
            max_nodes=self.max_nodes,
            random_seed=self.random_seed + 100000,
        )
        test_scores = self._predict_records(test_records)
        return _metric_from_top_k(test_records, test_scores, top_k=top_k)


def _load_gdfsl_checkpoint(model, checkpoint_path):
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload.get("model_states", {}).get("gdfsl_model")
    if state_dict is None:
        raise KeyError(f"Cannot find 'gdfsl_model' state_dict in checkpoint: {checkpoint_path}")
    model.load_state_dict(state_dict)
    top_k = payload.get("extra_params", {}).get("top_k")
    print(f"Loaded GDFSL2025 checkpoint: {rel_to_vfsl(checkpoint_path)}")
    return top_k, payload


def run_experiment(
        data_name='karate',
        infect_prob=0.05,
        diff_type='IC',
        time_step=10,
        recover_prob=0.0,
        sim_num=None,
        seed_ratio=0.01,
        top_rate=0.90,
        vis=False,
        source_ratio='1%',
        topology_id=1,
        load_model=False,
        checkpoint_path=None,
        retrain=True,
        test_mode=False,
        gdfsl_epochs=60,
        gdfsl_lr=5e-4,
        gdfsl_max_nodes=512,
        gdfsl_num_thres=50,
        gdfsl_top_k=None,
        gdfsl_sample_times=1):
    del top_rate, vis

    if topology_id == 'all':
        for topology in base_runner.discover_topologies(data_name, diff_type, source_ratio):
            run_experiment(
                data_name=data_name,
                infect_prob=infect_prob,
                diff_type=diff_type,
                time_step=time_step,
                recover_prob=recover_prob,
                sim_num=sim_num,
                seed_ratio=seed_ratio,
                source_ratio=source_ratio,
                topology_id=topology,
                load_model=load_model,
                checkpoint_path=checkpoint_path,
                retrain=retrain,
                test_mode=test_mode,
                gdfsl_epochs=gdfsl_epochs,
                gdfsl_lr=gdfsl_lr,
                gdfsl_max_nodes=gdfsl_max_nodes,
                gdfsl_num_thres=gdfsl_num_thres,
                gdfsl_top_k=gdfsl_top_k,
                gdfsl_sample_times=gdfsl_sample_times,
            )
        return

    experiment_dt = datetime.now()
    experiment_time = experiment_dt.strftime("%Y-%m-%d %H:%M:%S")
    run_id = experiment_dt.strftime("%Y%m%d_%H%M%S")
    run_root = (
        CKPT_OUTPUT_ROOT
        / "classical-gdfsl2025"
        / base_runner._safe_name(data_name)
        / base_runner._safe_name(diff_type)
        / base_runner._safe_name(source_ratio)
    )
    results_dir = ensure_within_vfsl(run_root / "results")
    checkpoints_dir = ensure_within_vfsl(run_root / "checkpoints")
    checkpoint_search_dirs = [checkpoints_dir]
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    print(f"\n{'=' * 50}\nLoading dataset: {data_name}\n{'=' * 50}")
    dataset = _unwrap_dataset(base_runner.load_dataset(
        data_name=data_name,
        diff_type=diff_type,
        source_ratio=source_ratio,
        topology_id=topology_id,
        sim_num=sim_num,
    ))
    topology = base_runner._topology_name(data_name, topology_id)
    adj, _, _, train_dataset_all, test_dataset_all = base_runner.split_dataset(
        dataset,
        train_ratio=0.8,
    )
    graph_features = _graph_feature_matrix(adj)

    model_name = "GDFSL2025"
    model = GDFSL2025Classical(
        epochs=gdfsl_epochs,
        lr=gdfsl_lr,
        max_nodes=gdfsl_max_nodes,
        num_thres=gdfsl_num_thres,
        top_k=gdfsl_top_k,
        sample_times=gdfsl_sample_times,
    )

    train_auc = 0.0
    train_f1 = 0.0
    top_k = gdfsl_top_k
    loaded_checkpoint_path = None
    checkpoint_payload = None
    trained_new_model = False

    if load_model:
        selected_checkpoint = checkpoint_path
        if selected_checkpoint in (None, "", "latest"):
            selected_checkpoint = base_runner.find_latest_checkpoint(
                checkpoint_search_dirs, model_name, data_name, topology, diff_type
            )
        if selected_checkpoint:
            selected_checkpoint = resolve_checkpoint_path(selected_checkpoint)
            loaded_top_k, checkpoint_payload = _load_gdfsl_checkpoint(model, selected_checkpoint)
            if loaded_top_k is not None:
                top_k = loaded_top_k
            model.top_k = top_k
            loaded_checkpoint_path = selected_checkpoint
            train_metrics = checkpoint_payload.get("train_metrics", {})
            train_auc = float(train_metrics.get("auc", 0.0))
            train_f1 = float(train_metrics.get("f1", 0.0))
        elif not retrain:
            raise FileNotFoundError(
                f"No checkpoint found for {model_name} {data_name}/{topology}/{diff_type} "
                f"in {rel_to_vfsl(checkpoints_dir)}. Set checkpoint_path to a .pt file, or set retrain=True."
            )

    print(f"\n{'=' * 50}\nRunning {model_name} on {data_name}/{diff_type}\n{'=' * 50}")
    if test_mode:
        if loaded_checkpoint_path is None:
            raise ValueError("test_mode=True requires load_model=True with a valid GDFSL2025 checkpoint.")
        print(f"Loaded checkpoint for {model_name}: {rel_to_vfsl(loaded_checkpoint_path)} with train F1: {train_f1:.3f}")
    else:
        if loaded_checkpoint_path:
            print(f"Continuing from checkpoint for {model_name}: {rel_to_vfsl(loaded_checkpoint_path)}")
        else:
            print(f"Training {model_name}...")
        top_k, train_auc, train_f1, _ = model.train(
            adj,
            train_dataset_all,
            graph_features,
        )
        trained_new_model = True

    print(f"Train results - AUC: {train_auc:.3f}, F1: {train_f1:.3f}")

    print(f"Testing {model_name}...")
    metric, preds = model.test(
        adj,
        test_dataset_all,
        graph_features,
        top_k,
    )
    del preds
    print(f"Test results - Acc: {metric.acc:.3f}, PR: {metric.pr:.3f}, "
          f"RE: {metric.re:.3f}, F1: {metric.f1:.3f}, AUC: {metric.auc:.3f}")

    results = {
        "timestamp": experiment_time,
        "dataset": data_name,
        "topology": topology,
        "infect_model": diff_type,
        "model": model_name,
        "train_auc": f"{train_auc:.3f}",
        "train_f1": f"{train_f1:.3f}",
        "test_acc": f"{metric.acc:.3f}",
        "test_pr": f"{metric.pr:.3f}",
        "test_re": f"{metric.re:.3f}",
        "test_f1": f"{metric.f1:.3f}",
        "test_auc": f"{metric.auc:.3f}",
        "infect_prob": infect_prob,
        "seed_ratio": seed_ratio,
        "source_ratio": source_ratio,
        "recover_prob": recover_prob,
        "time_step": time_step,
        "sim_num": sim_num,
    }

    if loaded_checkpoint_path is not None and not trained_new_model:
        results["checkpoint"] = rel_to_vfsl(loaded_checkpoint_path)
        results["loaded_model"] = True
    else:
        saved_checkpoint_path = base_runner.save_model_checkpoint(
            checkpoints_dir,
            model_name,
            run_id,
            metadata={
                "timestamp": experiment_time,
                "dataset": data_name,
                "topology": topology,
                "diff_type": diff_type,
                "source_ratio": source_ratio,
                "infect_prob": infect_prob,
                "seed_ratio": seed_ratio,
                "recover_prob": recover_prob,
                "time_step": time_step,
                "sim_num": sim_num,
                "num_nodes": adj.shape[0],
                "num_edges": adj.nnz // 2,
                "topology_snapshot_root": rel_to_vfsl(base_runner.GRAPH_INFER_ROOT),
                "gdfsl_feature_dim": 14,
                "gdfsl_structural_feature_dim": 7,
                "gdfsl_final_infection_feature_dim": 1,
                "gdfsl_epochs": gdfsl_epochs,
                "gdfsl_lr": gdfsl_lr,
                "gdfsl_max_nodes": gdfsl_max_nodes,
                "gdfsl_num_thres": gdfsl_num_thres,
                "gdfsl_top_k": top_k,
                "gdfsl_sample_times": gdfsl_sample_times,
                "feature_inputs": "GDFSL classical subgraph runner: observed final-infected nodes plus sampled candidates; 7 structural graph features + final infection status padded to the expected feature width.",
            },
            model_states={"gdfsl_model": base_runner._cpu_state_dict(model.forward_trainer)},
            extra_params={
                "top_k": top_k,
                "epochs": gdfsl_epochs,
                "lr": gdfsl_lr,
                "max_nodes": gdfsl_max_nodes,
                "num_thres": gdfsl_num_thres,
                "sample_times": gdfsl_sample_times,
            },
            threshold=top_k,
            train_metrics={"auc": train_auc, "f1": train_f1},
            test_metrics={
                "acc": metric.acc,
                "pr": metric.pr,
                "re": metric.re,
                "f1": metric.f1,
                "auc": metric.auc,
            },
        )
        results["checkpoint"] = rel_to_vfsl(saved_checkpoint_path)
        results["loaded_model"] = False

    base_runner.save_to_csv(results, os.path.join(results_dir, "results.csv"))


if __name__ == "__main__":
    run_experiment(
        data_name='karate',
        infect_prob=0.05,
        diff_type='IC',
        time_step=10,
        recover_prob=0.0,
        sim_num=None,
        seed_ratio=0.01,
        source_ratio='1%',
        topology_id=1,
        load_model=False,
        checkpoint_path=None,
        retrain=True,
        test_mode=False,
        gdfsl_epochs=60,
        gdfsl_lr=5e-4,
        gdfsl_max_nodes=512,
        gdfsl_num_thres=50,
        gdfsl_top_k=None,
        gdfsl_sample_times=1,
    )
