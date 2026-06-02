# CT-ISL: Topology-Guided Information Source Localization

CT-ISL is a research workspace for information source localization on diffusion
cascades. It supports classical IC/SI diffusion datasets and LLM-assisted news
diffusion datasets. The repository includes diffusion data, topology inference,
and multiple source-localization methods, including CT-ISL and several
baselines.

## Repository Layout

```text
VF-SL/
|-- Dataset-diffusion/
|   |-- Classical-diffusion/            # Classical IC/SI diffusion records
|   `-- LLM-diffusion/                  # Topic-level LLM diffusion records and embeddings
|-- Topology-Inference/
|   |-- Topology_Inference.py           # MI/context-based topology inference
|   |-- results/                        # Classical inferred topology outputs
|   `-- results-llm-context/            # LLM-context inferred topology outputs
|-- LLM-assisted-Dataset-Construction/
|   `-- Pokec-simulate_llm_diffusion_BBC.py
`-- Source-Location/
    |-- input_data(inferred_topology,  context)/
    |   |-- classical/                  # Inferred topology inputs for classical runs
    |   `-- context-llm/                # Inferred topology inputs for LLM runs
    `-- Topology-guided-Source-Localization/
        |-- project_paths.py            # Centralized path handling
        |-- CT-ISL/                     # Proposed CT-ISL method
        |-- Baseline-others/            # IVGD, NetSleuth, GCNSI, SLVAE, OJC
        `-- Baseline-GDFSL/             # GDFSL2025 baseline
```

## Data

- `Dataset-diffusion/Classical-diffusion/simulated-diffusion/` stores simulated
  IC/SI diffusion records for classical graph datasets such as `karate`,
  `cora_ml`, and `power_grid`.
- `Dataset-diffusion/LLM-diffusion/` stores topic-level LLM diffusion records
  for `business`, `tech`, `entertainment`, `politics`, and `sport`.
- LLM diffusion records use two columns: source labels and final infection
  states. Source localization is performed from the final infection state.
- Inferred topology files consumed by source-localization scripts are stored
  under `Source-Location/input_data(inferred_topology,  context)/`.

## Main Components

- `Topology-Inference/Topology_Inference.py` performs topology inference from
  final-state diffusion records. It supports classical topology inference and
  LLM-context topology inference.
- `Source-Location/Topology-guided-Source-Localization/CT-ISL/` runs the
  proposed CT-ISL method on classical and LLM diffusion datasets.
- `Source-Location/Topology-guided-Source-Localization/Baseline-others/` runs
  baseline source-localization methods: OJC, IVGD, NetSleuth, GCNSI, and SLVAE.
- `Source-Location/Topology-guided-Source-Localization/Baseline-GDFSL/` runs the
  GDFSL2025 baseline on classical and LLM diffusion datasets.
- `LLM-assisted-Dataset-Construction/Pokec-simulate_llm_diffusion_BBC.py`
  constructs LLM-assisted BBC-topic diffusion data on the Pokec user graph.

## Dependencies

No pinned top-level environment file is included. The scripts use common
scientific Python and graph-learning packages, including:

```text
numpy, scipy, pandas, scikit-learn, torch, torch-geometric,
networkx, matplotlib, tqdm, requests, transformers, sentence-transformers
```

The LLM-assisted data-construction pipeline expects a local Qwen model path and
an OpenAI-compatible local chat API. Check the `Config` class in
`LLM-assisted-Dataset-Construction/Pokec-simulate_llm_diffusion_BBC.py` before
running that pipeline.

## Quick Start

Run commands from the repository root unless a command changes directory
explicitly.

### 1. Infer Classical Topologies

```bash
cd Topology-Inference
python Topology_Inference.py \
  --dataset-format classical \
  --dataset power_grid \
  --mode IC \
  --source-ratio 1% \
  --records-per-task 300 \
  --out-dir results
```

### 2. Infer LLM-Context Topologies

```bash
cd Topology-Inference
python Topology_Inference.py \
  --dataset-format llm \
  --topics business tech entertainment politics sport \
  --out-dir results-llm-context
```

Source-localization scripts read topology inputs from
`Source-Location/input_data(inferred_topology,  context)/`. If new topology
outputs are generated, place or export the selected topology files into that
input directory before running source localization.

### 3. Run CT-ISL

```bash
cd Source-Location/Topology-guided-Source-Localization/CT-ISL

# Classical IC/SI source localization
python run-CT-ISL-Classical.py

# LLM diffusion source localization
python run-CT-ISL-LLM.py
```

### 4. Run Baseline Methods

```bash
cd Source-Location/Topology-guided-Source-Localization/Baseline-others

# Classical IC/SI baselines
python run-baseline-Classical.py

# LLM diffusion baselines
python run-baseline-LLM.py
```

### 5. Run GDFSL2025

```bash
cd Source-Location/Topology-guided-Source-Localization/Baseline-GDFSL

# Classical IC/SI GDFSL2025
python run-GDFSL-Classical.py

# LLM diffusion GDFSL2025
python run-GDFSL-LLM.py
```

The GDFSL2025 runners can be computationally heavy with their default
`__main__` settings. For smoke tests or debugging, reduce parameters such as
`sim_num`, `gdfsl_epochs`, and `gdfsl_max_nodes` in the script entry block.

### 6. Generate LLM-Assisted Diffusion Data

```bash
cd LLM-assisted-Dataset-Construction
python Pokec-simulate_llm_diffusion_BBC.py \
  --topic business \
  --news-start 0 \
  --news-end 300 \
  --repeat 1
```

Useful options include `--api-url`, `--embed-device`, `--infection-threshold`,
`--min-final-infected`, `--max-final-infected`, `--max-steps`,
`--force-recompute-profile-embeddings`, and
`--force-recompute-source-candidates`.

## Outputs

- Topology inference:
  `Topology-Inference/results/` and `Topology-Inference/results-llm-context/`
- CT-ISL source localization:
  `Source-Location/Topology-guided-Source-Localization/CT-ISL/ckpt&outputs/`
- Other source-localization baselines:
  `Source-Location/Topology-guided-Source-Localization/Baseline-others/ckpt&outputs/`
- GDFSL2025:
  `Source-Location/Topology-guided-Source-Localization/Baseline-GDFSL/ckpt&outputs/`
- LLM-assisted diffusion-construction caches:
  `LLM-assisted-Dataset-Construction/Pokec/data/processed/`

## Path Policy

Source-localization paths are centralized in
`Source-Location/Topology-guided-Source-Localization/project_paths.py`. Runtime
inputs and outputs are expected to stay inside the VF-SL project directory, and
scripts should be run with relative project paths rather than machine-specific
absolute paths.

## References

Topology inference baselines:

- Hao Huang et al. "Learning Diffusions under Uncertainty." *AAAI*, 2024.
  PIND: https://github.com/DiffusionNetworkInference/PIND
- Ting Gan et al. "Multi-task Inference of Diffusion Networks." *The Web
  Conference*, 2026. MIND: https://github.com/DiffusionNetwork/MIND
