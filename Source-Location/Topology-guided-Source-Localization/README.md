# Topology-guided Source Localization

This folder contains the source-localization experiments.
It includes the CT-ISL method, several classical baselines, and the GDFSL2025
baseline. Experiments use inferred topology graphs together with diffusion
cascade data.

## Directory Layout

- `CT-ISL/`

  Our main CT-ISL experiment code.

  - `run-CT-ISL-Classical.py`: run CT-ISL on classical IC/SI diffusion data
  - `run-CT-ISL-LLM.py`: run CT-ISL on LLM diffusion data
  - `model/`: CT-ISL model code
  - `ckpt&outputs/`: CT-ISL results, checkpoints, and visualizations

- `Baseline-others/`

  Baseline methods for comparison.

  - `run-baseline-Classical.py`: run baselines on classical IC/SI data
  - `run-baseline-LLM.py`: run baselines on LLM data
  - `baseline/`: IVGD, GCNSI, SLVAE and related model code
  - `Prescribed.py`: NetSleuth, OJC

- `Baseline-GDFSL/`

  GDFSL2025 baseline code.

  - `run-GDFSL-Classical.py`: run GDFSL2025 on classical IC/SI data
  - `run-GDFSL-LLM.py`: run GDFSL2025 on LLM data
  - `code/`: GDFSL2025 model code

- `project_paths.py`

  Shared path configuration for datasets, inferred graphs, and outputs.

## Data

The scripts read data from these locations:

```text
Dataset-diffusion/Classical-diffusion/simulated-diffusion/
Dataset-diffusion/LLM-diffusion/
Source-Location/input_data(inferred_topology,  context)/classical/
Source-Location/input_data(inferred_topology,  context)/context-llm/
```

Classical experiments use datasets such as `karate`, `cora_ml`, and
`power_grid`. LLM experiments use topics such as `business`, `tech`,
`entertainment`, `politics`, and `sport`.

## Quick Start

Run CT-ISL:

```bash
python CT-ISL/run-CT-ISL-Classical.py
python CT-ISL/run-CT-ISL-LLM.py
```

Run other baselines:

```bash
python Baseline-others/run-baseline-Classical.py
python Baseline-others/run-baseline-LLM.py
```

Run GDFSL2025:

```bash
python Baseline-GDFSL/run-GDFSL-Classical.py
python Baseline-GDFSL/run-GDFSL-LLM.py
```

Edit the `if __name__ == "__main__"` block in each script to change datasets
and run settings.

## Common Parameters

- `data_name`: dataset or LLM topic name
- `diff_type`: `IC`, `SI`, or `LLM`
- `source_ratio`: source-node ratio, for example `1%`
- `topology_id`: inferred topology id; `all` runs all discovered topologies
- `sim_num`: number of cascades to use; `None` uses all cascades
- `vis`: save source-prediction visualizations

Some scripts also support checkpoint options such as `load_model`,
`checkpoint_path`, `retrain`, and `test_mode`.

## Baseline Methods

`Baseline-others` includes:

- `NetSleuth`
- `OJC`
- `GCNSI`
- `IVGD`
- `SLVAE`

To run only selected baselines, edit the `models` dictionary in the
corresponding baseline script.

## Outputs

Each script saves outputs inside its own folder:

```text
CT-ISL/ckpt&outputs/
Baseline-others/ckpt&outputs/
Baseline-GDFSL/ckpt&outputs/
```

Typical output files:

- `results/results.csv`: metrics and run metadata
- `checkpoints/*.pt`: saved model states or parameters
- `visualizations/*.png`: optional prediction plots

## Checkpoint Zip Files

Some saved checkpoints may be provided as `checkpoints.zip`. Unzip each file
into the corresponding `checkpoints/` directory under the same experiment
folder.

For example, unzip:

```text
CT-ISL/ckpt&outputs/classical/cora_ml/IC/1_/checkpoints.zip
```

to:

```text
CT-ISL/ckpt&outputs/classical/cora_ml/IC/1_/checkpoints/
```

## Metrics

The scripts report training AUC/F1 and testing accuracy, precision, recall,
F1, and AUC.
