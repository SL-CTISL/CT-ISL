

import csv
import os
import pickle
import re
from datetime import datetime
from pathlib import Path
import sys

CT_ISL_DIR = Path(__file__).resolve().parent
TGS_ROOT = CT_ISL_DIR.parent
BASELINE_OTHERS_DIR = TGS_ROOT / "Baseline-others"
for path in (TGS_ROOT, BASELINE_OTHERS_DIR, CT_ISL_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from baseline.IVGD.main import IVGD
from baseline.GCNSI.main import GCNSI
from Prescribed import NetSleuth, OJC
from utils import split_dataset
from model.main_new import SLCVAE, SLCVAE_model
from model.model_new import CVAE as SLCVAE_CVAE, GNN as SLCVAE_GNN
from baseline.SLVAE.main import SLVAE
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from scipy.sparse import coo_matrix

from project_paths import (
    CLASSICAL_DIFFUSION_ROOT,
    CLASSICAL_INFERRED_GRAPH_ROOT,
    ensure_within_vfsl,
    rel_to_vfsl,
    resolve_vfsl_path,
)

DIFFUSION_ROOT = CLASSICAL_DIFFUSION_ROOT
GRAPH_INFER_ROOT = CLASSICAL_INFERRED_GRAPH_ROOT
CKPT_OUTPUT_ROOT = ensure_within_vfsl(CT_ISL_DIR / "ckpt&outputs")


def resolve_checkpoint_path(checkpoint_path):
    path = Path(checkpoint_path)
    if path.is_absolute():
        return resolve_vfsl_path(path)

    for base in (CKPT_OUTPUT_ROOT, CT_ISL_DIR):
        candidate = base / path
        if candidate.exists():
            return ensure_within_vfsl(candidate)

    if path.parts and path.parts[0] in {"ckpt&outputs", "classical", "llm", "logs"}:
        base = CT_ISL_DIR if path.parts[0] == "ckpt&outputs" else CKPT_OUTPUT_ROOT
        return ensure_within_vfsl(base / path)

    return resolve_vfsl_path(path)

def visualize_source_prediction(adj, predictions, save_dir, save_name, figsize=(5, 5)):
    G = nx.from_scipy_sparse_array(adj)
    num_nodes = adj.shape[0]
    pos = nx.spring_layout(G, seed=43)

    print(np.sum(predictions))
    non_source_nodes = [node for node in G.nodes() if predictions[node] == 0]
    source_nodes = [node for node in G.nodes() if predictions[node] == 1]

    plt.figure(figsize=figsize)

    nx.draw_networkx_edges(
        G, pos, edge_color="gray", width=0.6, alpha=0.8, arrows=False
    )


    nx.draw_networkx_nodes(
        G, pos, nodelist=non_source_nodes,
        node_color='lightgreen', node_size=110, alpha=1, edgecolors='black', linewidths=0.8
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=source_nodes,
        node_color='coral', node_size=280, alpha=1, edgecolors='black', linewidths=0.8
    )

    plt.axis('off')
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = Path(save_dir) / f"{save_name}.png"
    print('the vis of the prediction is saved in:', rel_to_vfsl(path))

    plt.savefig(path, dpi=300)
    plt.close()

def save_to_csv(results, filename="experiment_results.csv"):
  
    if not os.path.isfile(filename) or os.path.getsize(filename) == 0:
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=results.keys())
            writer.writeheader()
            writer.writerow(results)
        return

    with open(filename, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    new_fields = [key for key in results.keys() if key not in fieldnames]
    if new_fields:
        fieldnames.extend(new_fields)
        rows.append(results)
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return

    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(results)

def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _cpu_state_dict(model):
    if model is None or not hasattr(model, "state_dict"):
        return None
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in model.state_dict().items()
    }


def _to_cpu_tensor(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


def save_model_checkpoint(
        checkpoint_dir,
        model_name,
        run_id,
        metadata,
        model_states=None,
        extra_params=None,
        seed_vae_train=None,
        train_metrics=None,
        test_metrics=None,
        threshold=None):
    checkpoint_dir = ensure_within_vfsl(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = (
        f"{run_id}_{_safe_name(metadata['dataset'])}_{_safe_name(metadata['topology'])}_"
        f"{_safe_name(metadata['diff_type'])}_{_safe_name(model_name)}.pt"
    )
    path = ensure_within_vfsl(Path(checkpoint_dir) / filename)
    payload = {
        "model_name": model_name,
        "metadata": metadata,
        "model_states": model_states or {},
        "extra_params": extra_params or {},
        "seed_vae_train": _to_cpu_tensor(seed_vae_train),
        "threshold": threshold,
        "train_metrics": train_metrics or {},
        "test_metrics": test_metrics or {},
    }
    torch.save(payload, path)
    print(f"Saved model checkpoint: {rel_to_vfsl(path)}")
    return str(path)

def find_latest_checkpoint(checkpoint_dir, model_name, data_name, topology, diff_type):
    pattern = (
        f"*_{_safe_name(data_name)}_{_safe_name(topology)}_"
        f"{_safe_name(diff_type)}_{_safe_name(model_name)}.pt"
    )
    if isinstance(checkpoint_dir, (str, os.PathLike)):
        checkpoint_dirs = [checkpoint_dir]
    else:
        checkpoint_dirs = list(checkpoint_dir)
    matches = []
    for directory in checkpoint_dirs:
        directory = Path(directory)
        if directory.is_dir():
            matches.extend(directory.glob(pattern))
    matches = sorted(matches, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)
    return str(matches[0]) if matches else None


def _extract_state_dict(payload, model_key="slvae_model"):
    model_states = payload.get("model_states", {})
    if model_key in model_states:
        return model_states[model_key]
    if "slcvae_model" in model_states:
        return model_states["slcvae_model"]
    if model_states and all(torch.is_tensor(value) for value in model_states.values()):
        return model_states
    raise KeyError(f"Cannot find {model_key!r} state_dict in checkpoint.")


def build_slcvae_from_state_dict(adj, state_dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cvae = SLCVAE_CVAE(adj, input_dim=1, cond_dim=64).to(device)
    gnn = SLCVAE_GNN(adj_matrix=adj).to(device)
    slcvae_model = SLCVAE_model(cvae, gnn).to(device)
   
    slcvae_model.load_state_dict(state_dict, strict=False)
    slcvae_model.eval()
    for param in slcvae_model.parameters():
        param.requires_grad = False
    return slcvae_model


def load_slcvae_checkpoint(checkpoint_path, adj):
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location='cpu')
    
    state_dict = _extract_state_dict(payload, model_key="slvae_model")
    slcvae_model = build_slcvae_from_state_dict(adj, state_dict)
    seed_vae_train = _to_cpu_tensor(payload.get("seed_vae_train"))
    thres = payload.get("threshold")
    if thres is None:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain a threshold.")
    print(f"Loaded model checkpoint: {rel_to_vfsl(checkpoint_path)}")
    return slcvae_model, seed_vae_train, thres, payload

def _ratio_to_label(ratio):
    if isinstance(ratio, str):
        return ratio
    percentage = ratio * 100
    if float(percentage).is_integer():
        return f"{int(percentage)}%"
    return f"{percentage:g}%"


def _topology_name(data_name, topology_id):
    if isinstance(topology_id, str):
        return topology_id
    return f"{data_name}{topology_id}"


def _natural_key(text):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def discover_topologies(
        data_name='karate',
        diff_type='IC',
        source_ratio='1%',
        diffusion_root=DIFFUSION_ROOT):
    ratio_label = _ratio_to_label(source_ratio)
    diffusion_root = ensure_within_vfsl(diffusion_root)
    base_dir = (
        Path(diffusion_root)
        / data_name
        / f"{data_name}_{diff_type}"
        / f"{data_name}_{diff_type}_{ratio_label}"
    )
    suffix = f"_{diff_type}_{ratio_label}"
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Diffusion topology directory not found: {rel_to_vfsl(base_dir)}")
    topologies = [
        path.name[:-len(suffix)]
        for path in base_dir.iterdir()
        if path.is_dir() and path.name.endswith(suffix)
    ]
    return sorted(topologies, key=_natural_key)


def _read_inferred_adj(edge_path, num_nodes, symmetric=True):
    row, col = [], []
    with open(edge_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"[\s,]+", line)
            if len(parts) < 2:
                continue
            i, j = int(parts[0]), int(parts[1])
            if i == j:
                continue
            if i >= num_nodes or j >= num_nodes:
                raise ValueError(
                    f"Edge ({i}, {j}) in {edge_path} exceeds diffusion node count {num_nodes}."
                )
            row.append(i)
            col.append(j)
            if symmetric:
                row.append(j)
                col.append(i)

    data = np.ones(len(row), dtype=np.float32)
    adj = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32)
    adj.sum_duplicates()
    adj.data[:] = 1.0
    return adj.tocsr()


def _load_diffusion_matrix(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or "test_dataset_all" not in obj:
        raise ValueError(f"{path} is not a supported diffusion pickle.")
    arr = np.asarray(obj["test_dataset_all"], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{path} must contain a [num_nodes, >=2] diffusion matrix.")
    return arr


def load_dataset(
        data_name='karate',
        data_dir=None,
        diff_type='IC',
        source_ratio='1%',
        topology_id=1,
        diffusion_root=DIFFUSION_ROOT,
        graph_infer_root=GRAPH_INFER_ROOT,
        sim_num=None):
    
    del data_dir
    ratio_label = _ratio_to_label(source_ratio)
    diffusion_root = ensure_within_vfsl(diffusion_root)
    graph_infer_root = ensure_within_vfsl(graph_infer_root)
    topology = _topology_name(data_name, topology_id)

    diffusion_dir = (
        Path(diffusion_root)
        / data_name
        / f"{data_name}_{diff_type}"
        / f"{data_name}_{diff_type}_{ratio_label}"
        / f"{topology}_{diff_type}_{ratio_label}"
    )
    if not diffusion_dir.is_dir():
        raise FileNotFoundError(f"Diffusion directory not found: {rel_to_vfsl(diffusion_dir)}")

    pattern = re.compile(rf"^{re.escape(topology)}_{re.escape(diff_type)}_{re.escape(ratio_label)}_\d+$")
    diffusion_files = sorted(p for p in diffusion_dir.iterdir() if p.is_file() and pattern.match(p.name))
    if sim_num is not None:
        diffusion_files = diffusion_files[:sim_num]
    if not diffusion_files:
        raise FileNotFoundError(f"No diffusion files found in {rel_to_vfsl(diffusion_dir)}")

    diff_mats_all = [_load_diffusion_matrix(path) for path in diffusion_files]
    num_nodes = diff_mats_all[0].shape[0]
    for path, mat in zip(diffusion_files, diff_mats_all):
        if mat.shape[0] != num_nodes:
            raise ValueError(f"Node count mismatch in {rel_to_vfsl(path)}: expected {num_nodes}, got {mat.shape[0]}")

    graph_dir = Path(graph_infer_root) / data_name / diff_type / ratio_label / "final"
    graph_matches = sorted(graph_dir.glob(f"task_*_{topology}_{diff_type}_{ratio_label}_graph.txt"))
    if not graph_matches:
        raise FileNotFoundError(f"Inferred graph file not found under: {rel_to_vfsl(graph_dir)}")
    adj_mat = _read_inferred_adj(graph_matches[0], num_nodes=num_nodes, symmetric=True)

    diff_mat_all = torch.tensor(np.stack(diff_mats_all, axis=0), dtype=torch.float32)
    diff_mat = diff_mat_all[:, :, :2]
    dataset = {
        'adj_mat': adj_mat,
        'diff_mat': diff_mat,
        'diff_mat_all': diff_mat_all,
    }
    print(
        f"Loaded {topology}: adj={adj_mat.shape}, inferred_edges={adj_mat.nnz // 2}, "
        f"diff_mat={tuple(diff_mat.shape)}, diff_mat_all={tuple(diff_mat_all.shape)}"
    )
    return dataset, None

def run_experiment(
        data_name='jazz',
        infect_prob=0.1,
        diff_type='SI',
        time_step=10,
        recover_prob=0.05,
        sim_num=500,
        seed_ratio=0.1,
        top_rate=0.90,
        vis=False,
        source_ratio='1%',
        topology_id=1,
        load_model=True,
        checkpoint_path=None,
        retrain=False,
        test_mode=False):
  
    if topology_id == 'all':
        for topology in discover_topologies(data_name, diff_type, source_ratio):
            run_experiment(
                data_name=data_name,
                infect_prob=infect_prob,
                diff_type=diff_type,
                time_step=time_step,
                recover_prob=recover_prob,
                sim_num=sim_num,
                seed_ratio=seed_ratio,
                top_rate=top_rate,
                vis=vis,
                source_ratio=source_ratio,
                topology_id=topology,
                load_model=load_model,
                checkpoint_path=checkpoint_path,
                retrain=retrain,
                test_mode = test_mode
            )
        return


    experiment_dt = datetime.now()
    experiment_time = experiment_dt.strftime("%Y-%m-%d %H:%M:%S")
    run_id = experiment_dt.strftime("%Y%m%d_%H%M%S")
    run_root = CKPT_OUTPUT_ROOT / "classical" / _safe_name(data_name) / _safe_name(diff_type) / _safe_name(source_ratio)
    results_dir = ensure_within_vfsl(run_root / "results")
    vis_dir = ensure_within_vfsl(run_root / "visualizations")
    checkpoints_dir = ensure_within_vfsl(run_root / "checkpoints")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    


    print(f"\n{'='*50}\nLoading dataset: {data_name}\n{'='*50}")
    dataset, user_embeddings_tensor = load_dataset(
        data_name=data_name,
        diff_type=diff_type,
        source_ratio=source_ratio,
        topology_id=topology_id,
        sim_num=sim_num,
    )
    topology = _topology_name(data_name, topology_id)
    
   
    adj, train_dataset, test_dataset , train_dataset_all, test_dataset_all = split_dataset(dataset, train_ratio=0.8)

    if vis:  # -6/1
        visualize_source_prediction(
            adj,
            test_dataset[8][:,0].numpy(),
            save_dir=vis_dir,
            save_name="true_source_prediction"
        )
        visualize_source_prediction(
            adj,
            test_dataset[-6][:,0].numpy(),
            save_dir=vis_dir,
            save_name="true_source_prediction2"
        )
        visualize_source_prediction(
            adj,
            test_dataset[14][:,0].numpy(),
            save_dir=vis_dir,
            save_name="true_source_prediction3"
        )
        # return


    models = {
        # "NetSleuth": NetSleuth(),
        # "OJC": OJC(),
        # "GCNSI": GCNSI(),
        #"IVGD": IVGD(),
        # "SLVAE": SLVAE(),
        "CVAE": SLCVAE()
    }
    

    for model_name, model in models.items():
        print(f"\n{'='*50}\nRunning {model_name}\n{'='*50}")
        
      
        train_auc=0
        train_f1=0
        model_states = {}
        extra_params = {}
        seed_vae_train = None
        thres = None
        loaded_checkpoint_path = None
        checkpoint_payload = None

        slvae_model = None

        if load_model and model_name == "CVAE":
            selected_checkpoint = checkpoint_path
            if selected_checkpoint in (None, "", "latest"):
                selected_checkpoint = find_latest_checkpoint(
                    checkpoints_dir, model_name, data_name, topology, diff_type
                )
            if selected_checkpoint:
                selected_checkpoint = resolve_checkpoint_path(selected_checkpoint)
                slvae_model, seed_vae_train, thres, checkpoint_payload = load_slcvae_checkpoint(
                    selected_checkpoint, adj
                )
                loaded_checkpoint_path = selected_checkpoint
                train_metrics = checkpoint_payload.get("train_metrics", {})
                train_auc = train_metrics.get("auc", 0)
                train_f1 = train_metrics.get("f1", 0)
            elif not retrain:
                raise FileNotFoundError(
                    f"No checkpoint found for {model_name} {data_name}/{topology}/{diff_type} "
                    f"in {rel_to_vfsl(checkpoints_dir)}. Set checkpoint_path to a .pt file, or set "
                    f"load_model=False/retrain=True to train a new model."
                )

        if loaded_checkpoint_path is None:
            print(f"Training {model_name}...")
        else:
            print(f"Loaded checkpoint for {model_name}: {rel_to_vfsl(loaded_checkpoint_path)} with train F1: {train_f1:.3f}")

        if test_mode == True:
            pass
        elif model_name == "NetSleuth":
            #k = train_dataset[0].shape[0] * 0.05

            k, train_auc, train_f1 = model.train(adj, train_dataset)
            extra_params = {"k": k}
            
        elif model_name == "OJC":
            #Y = train_dataset[0].shape[0] * 0.01
            Y, train_auc, train_f1 = model.train(adj, train_dataset)
            extra_params = {"Y": Y}
        elif model_name == "GCNSI":
            gcnsi_model, thres, train_auc, train_f1, pred = model.train(adj, train_dataset, lr=1e-4, num_epoch=50)  
            model_states = {"gcnsi_model": _cpu_state_dict(gcnsi_model)}
          
            pred = (pred >= thres)

        elif model_name == "IVGD":
         
            diffusion_model = model.train_diffusion(adj, train_dataset, num_epoch=50, lr=1e-3) 
            ivgd_model, thres, train_auc, train_f1, pred = model.train(
                adj, train_dataset, diffusion_model, lr=1e-3, num_epoch=200)
            model_states = {
                "diffusion_model": _cpu_state_dict(diffusion_model),
                "ivgd_model": _cpu_state_dict(ivgd_model),
            }
           
            pred = (pred >= thres)

        elif model_name == "SLVAE":
            slvae_model, seed_vae_train, thres, train_auc, train_f1, pred = model.train(
                adj, train_dataset, num_epoch=100, lr=1e-4, print_epoch=1)
            model_states = {"slvae_model": _cpu_state_dict(slvae_model)}
    
        else:
            ## karate epoch=50 lr=1e-3
            ## power_grid epoch=1000 lr=1e-3
            slvae_model, seed_vae_train, thres, train_auc, train_f1, pred = model.train(
                adj, train_dataset_all, user_embeddings=user_embeddings_tensor, num_epoch=100, lr=5e-4, slcvae_model_reload=slvae_model)  
            model_states = {"slvae_model": _cpu_state_dict(slvae_model)}
    
        print(f"Train results - AUC: {train_auc:.3f}, F1: {train_f1:.3f}")
        
  
        print(f"Testing {model_name}...")
        if model_name == "NetSleuth":
            metric, preds = model.test(adj, test_dataset, k)
        elif model_name == "OJC":
            metric, preds = model.test(adj, test_dataset, Y)
        elif model_name == "GCNSI":
            metric, preds = model.test(adj, test_dataset, gcnsi_model, thres)
        elif model_name == "IVGD":
            metric, preds = model.test(adj, test_dataset, diffusion_model, ivgd_model, thres)
        elif model_name == "SLVAE":
            metric = model.infer(test_dataset, slvae_model, seed_vae_train, thres, num_epoch=1, lr=1e-3)
            preds = None
        else:
            metric, pred, y_true, y_pred, preds = model.infer(test_dataset_all, slvae_model, seed_vae_train, adj, user_embeddings=user_embeddings_tensor, thres=thres, num_epoch=10, lr=1e-2)  #num_epoch = 100
        if vis and preds is not None:
            visualize_source_prediction(
                adj,
                preds[8],
                save_dir=vis_dir,
                save_name=f"{model_name}_source_prediction"
            )
            visualize_source_prediction(
                adj,
                preds[-6],
                save_dir=vis_dir,
                save_name=f"{model_name}_source_prediction2"
            )

            visualize_source_prediction(
                adj,
                preds[14],
                save_dir=vis_dir,
                save_name=f"{model_name}_source_prediction3"
            )

        
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
            "sim_num": sim_num
        }
        
        if loaded_checkpoint_path is not None:
            results["checkpoint"] = rel_to_vfsl(loaded_checkpoint_path)
            results["loaded_model"] = True
        else:
            saved_checkpoint_path = save_model_checkpoint(
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
                },
                model_states=model_states,
                extra_params=extra_params,
                seed_vae_train=seed_vae_train,
                threshold=thres,
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
        save_to_csv(results, os.path.join(results_dir, "results.csv"))
            


if __name__ == "__main__":

    run_experiment(
        data_name='power_grid',
        infect_prob=0.05,
        diff_type='SI',
        time_step=10,
        recover_prob=0.0,
        sim_num=None,
        seed_ratio=0.01,
        top_rate=0.9,
        vis=False,
        source_ratio='1%',
        topology_id=4,
        load_model=False,
        checkpoint_path=None, 
        retrain=True,
        test_mode=False
    )
