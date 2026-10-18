# IFCG

Implementation of **Interval-Valued Fuzzy Causal Graphs for Counterfactual Multistage Diagnosis**.

IFCG retains a stage-constrained structural ensemble and interval-valued Takagi–Sugeno mechanisms. It ranks candidate causes by supported outcome change while penalizing interval width and structural disagreement. Supported action search propagates intervals forward and refuses recommendations that cross the outcome limit or fail structural/sensitivity checks.

## Install

```bash
python -m venv .venv
python -m pip install -e .
```

Use Python 3.11 or 3.12 and activate the environment for your shell.

## Prepare process and intervention records

Use controlled Tennessee Eastman interventions or [CIPCaD-Bench](https://github.com/giovanniMen/CPCaD-Bench) simulator exports for causal quantities. [SECOM](https://archive.ics.uci.edu/dataset/179/secom) supplies an observational transfer setting and cannot supply true intervention outcomes or physical source labels by itself.

Create `data/process.csv` with `case,unit,split` and numeric process variables. `split` is `train`, `cal` or `test`, with unit-disjoint assignment. Optional `root` contains the documented initiating variable's exact column name; leave it empty where unavailable. Each row is an aligned staged process observation, and the variable list in `process.example.json` must be in a known topological order. The same JSON defines stage indices, outcome variable, outcome limit, measurement standard deviations and actionable ranges/effect sources. Only earlier-stage variables may be parents in the full method. Inputs must be finite and in consistent physical units.

For intervention evaluation/calibration, create a separate CSV with `case,split,variable,value,outcome`. Each row contains an externally imposed variable/value and its controlled post-intervention outcome for that process case. `cal` and `test` interventions must reference process cases in their corresponding split. Action metadata must identify its simulator or intervention-log effect source. Interventional calibration is optional for observational ranking and mandatory when interpreting nominal effect coverage.

For structural metrics, export the simulator's binary source-row/child-column adjacency to `data/truth_graph.npy`; do not infer this reference from the learned graph. The process and intervention preparation never changes manuscript files.

## Fit, diagnose and search actions

```bash
ifcg data/process.csv process.example.json --interventions data/interventions.csv --truth-graph data/truth_graph.npy --output outputs/full
```

`--ensemble 10` controls structural bootstrap samples, `--sparsity 0.02` controls allowed-edge Lasso selection, `--edge-threshold` removes tiny coefficients and `--rules 4` controls local fuzzy complexity. The normalized measurement intervals and per-mechanism residuals are kept separate. Calibration interventions expand effect intervals using the finite-sample 90th-percentile undercoverage score.

`--width-weight`, `--conflict-weight` penalize uncertain/conflicting cause scores. `--conflict-limit` bounds structural disagreement. `--confounding` applies a common normalized residual perturbation in both signs when checking action robustness. Action values are searched on a bounded grid over registered ranges and restricted to empirical training support. The selected recommendation minimizes normalized outcome plus action cost among accepted choices.

## Controls

```bash
ifcg data/process.csv process.example.json --variant point --output outputs/point
ifcg data/process.csv process.example.json --variant no_stages --output outputs/no_stages
ifcg data/process.csv process.example.json --variant no_conflict --output outputs/no_conflict
ifcg data/process.csv process.example.json --variant no_abstention --output outputs/no_abstention
```

Add the same intervention and truth-graph arguments to every controlled causal comparison. `point` removes fuzzy effect intervals, `no_stages` allows edges within a stage while preserving the registered total variable order, `no_conflict` removes structural conflict from ranking, and `no_abstention` issues the best support-qualified grid candidate without the acceptance checks. These are matched mechanism controls rather than renamed external graph-learning algorithms.

## Outputs and metrics

`model.pkl` saves the structural/mechanism ensemble and scaling state. `edge_probabilities.npy` and `graph.png` show source-to-child bootstrap edge inclusion. `diagnoses.json` contains candidate rankings, effect ranges, structural conflicts and supported recommendations. `interventions.csv` contains test-intervention outcome intervals.

`metrics.json` reports root top-1/top-3 where true roots exist, recommendation coverage, controlled intervention-effect MAE, 90% interval coverage/width, accepted-action validity, structural Hamming distance and transitive path agreement. Comparing predicted and controlled post-intervention outcomes gives the same absolute effect error after subtracting their common factual outcome. Hamming distance counts directed adjacency differences; a reversal therefore counts two. Path agreement includes all ordered node pairs. Unavailable causal references produce absent causal metrics, rather than observational proxy scores.

## Modeling conventions

Stage order guarantees acyclicity before fitting. Lasso supplies sparse parent sets in each bootstrap graph; compact Gaussian Takagi–Sugeno mechanisms then fit nonlinear local responses by weighted ridge regression. Sensor uncertainty widens antecedent grades, and 90th-percentile normal mechanism residuals widen consequents. Endpoint-switch type reduction and low/mid/high parent sampling propagate intervals. Structural variation remains explicit rather than collapsing into one graph. Candidate normalization uses the training median and abducted factual residuals. Hidden-confounding perturbation is a sensitivity scenario, not a procedure that identifies unrestricted latent confounding. An effect-source declaration and bounded support are required for recommendations; observational ranking alone does not establish an actionable intervention.

## Reproduction logic conventions

Interval propagation encloses the complete parent box. Gaussian firing-grade bounds use the nearest/farthest distance to that box, affine consequents use coefficient-sign interval bounds, and type reduction combines these bounds. Mixed-sign parent effects are not approximated by only the all-low/all-high corners.
