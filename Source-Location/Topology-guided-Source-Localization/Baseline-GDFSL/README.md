# GDFSL Baseline

This directory contains the GDFSL baseline used for topology-guided source
localization experiments.

The implementation is based on the following paper:

Dongpeng Hou, Yuchen Wang, Chao Gao, and Xianghua Li. "A Generalized Diffusion
Framework with Learnable Propagation Dynamics for Source Localization." In
*Proceedings of the 34th International Joint Conference on Artificial
Intelligence (IJCAI 2025)*, 2919-2927, 2025.

The original project is available at:
https://github.com/cgao-comp/GDFSL

## Directory Structure

- `code/` contains the original GDFSL implementation. This version provides the
  IC-propagation-model-based GDFSL baseline.
- `run-GDFSL-Classical.py` adapts GDFSL to classical IC/SI diffusion datasets.
- `run-GDFSL-LLM.py` adapts GDFSL to LLM diffusion datasets.

## Usage

Run the classical IC/SI version:

```bash
python run-GDFSL-Classical.py
```

Run the LLM diffusion version:

```bash
python run-GDFSL-LLM.py
```

Edit each script's `if __name__ == "__main__"` block to change datasets,
topology ids, simulation counts, and GDFSL training settings.

## Notes

This baseline is included for comparison with CT-ISL and other source
localization methods in this repository. Paths are expected to be resolved
relative to the VF-SL project directory.
