# IFCG

Implementation of **Interval-Valued Fuzzy Causal Graphs for Counterfactual Multistage Diagnosis**.

## Installation

Use Python 3.11 or 3.12. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

Use controlled Tennessee Eastman interventions or [CIPCaD-Bench](https://github.com/giovanniMen/CPCaD-Bench) simulator exports for causal quantities. [SECOM](https://archive.ics.uci.edu/dataset/179/secom) supplies an observational transfer setting and cannot supply true intervention outcomes or physical source labels by itself.

Create `data/process.csv` with `case,unit,split` and numeric process variables. `split` is `train`, `cal` or `test`, with unit-disjoint assignment. Optional `root` contains the documented initiating variable's exact column name; leave it empty where unavailable. Each row is an aligned staged process observation, and the variable list in `process.example.json` must be in a known topological order. The same JSON defines stage indices, outcome variable, outcome limit, measurement standard deviations and actionable ranges/effect sources. Only earlier-stage variables may be parents in the full method. Inputs must be finite and in consistent physical units.

For intervention evaluation/calibration, create a separate CSV with `case,split,variable,value,outcome`. Each row contains an externally imposed variable/value and its controlled post-intervention outcome for that process case. `cal` and `test` interventions must reference process cases in their corresponding split. Action metadata must identify its simulator or intervention-log effect source. Interventional calibration is optional for observational ranking and mandatory when interpreting nominal effect coverage.

For structural metrics, export the simulator's binary source-row/child-column adjacency to `data/truth_graph.npy`; do not infer this reference from the learned graph. The process and intervention preparation never changes manuscript files.

## Run and evaluate

```bash
ifcg data/process.csv process.example.json --interventions data/interventions.csv --truth-graph data/truth_graph.npy --output outputs/full --seed 0
```

Fill action `effect_source` with actual intervention provenance before requesting recommendations. Omit intervention/truth-graph arguments where unavailable; corresponding reference metrics remain unavailable.

## Ablations

Supported controls: `point`, `no_stages`, `no_conflict`, `no_abstention`. Reuse the same inputs, seed and budget. For retrained controls, evaluate the corresponding saved model.

```bash
ifcg data/process.csv process.example.json --interventions data/interventions.csv --truth-graph data/truth_graph.npy --output outputs/point --seed 0 --variant point
```

## Outputs

`model.pkl`, `edge_probabilities.npy` and `graph.png` retain fitted mechanisms/graphs. `diagnoses.json` records rankings/recommendations; `interventions.csv` and `metrics.json` contain supported intervention and reference-based scores.

Source inspection: Python 3.13.5, numpy 2.1.3, pandas 2.2.3, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
