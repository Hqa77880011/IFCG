import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso, Ridge


class FuzzyMechanism:
    def __init__(self, x, y, rules, seed, sensor_sd):
        self.centers = KMeans(n_clusters=min(rules, len(x)), random_state=seed, n_init=5).fit(x).cluster_centers_
        self.sigma = x.std(0).clip(.1)
        self.sensor_sd = np.asarray(sensor_sd)
        weights = self.membership(x, 0)
        design = np.c_[x, np.ones(len(x))]
        self.models = [Ridge(alpha=.1).fit(design, y, sample_weight=weights[:, r] + 1e-8) for r in range(len(self.centers))]
        self.residual = float(np.quantile(np.abs(y - self.predict(x)), .9))

    def membership(self, x, direction):
        widths = self.sigma * np.exp(direction * self.sensor_sd / self.sigma)
        grades = np.exp(-.5 * (((x[:, None] - self.centers) / widths) ** 2).sum(-1)).clip(1e-30)
        return grades / grades.sum(1, keepdims=True)

    def predict(self, x):
        values = np.stack([m.predict(np.c_[x, np.ones(len(x))]) for m in self.models], axis=1)
        return (self.membership(x, 0) * values).sum(1)

    def endpoints(self, x):
        values = np.stack([m.predict(np.c_[x, np.ones(len(x))]) for m in self.models], axis=1)
        grade_lo = np.exp(-.5 * (((x[:, None] - self.centers) / (self.sigma * np.exp(-self.sensor_sd / self.sigma))) ** 2).sum(-1)).clip(1e-30)
        grade_hi = np.exp(-.5 * (((x[:, None] - self.centers) / (self.sigma * np.exp(self.sensor_sd / self.sigma))) ** 2).sum(-1)).clip(1e-30)
        endpoints = []
        for left in (True, False):
            order = np.argsort(values, axis=1)
            y = np.take_along_axis(values, order, axis=1)
            lo = np.take_along_axis(grade_lo, order, axis=1)
            hi = np.take_along_axis(grade_hi, order, axis=1)
            estimate = y.mean(1)
            for _ in range(50):
                w = np.where(y <= estimate[:, None], hi if left else lo, lo if left else hi)
                updated = (w * y).sum(1) / w.sum(1)
                if np.max(np.abs(updated - estimate)) < 1e-8:
                    estimate = updated
                    break
                estimate = updated
            endpoints.append(estimate + (-self.residual if left else self.residual))
        return endpoints


    def interval(self, lower, upper):
        """Enclose a fuzzy mechanism over the complete parent box."""
        lower, upper = np.asarray(lower), np.asarray(upper)
        farthest = np.maximum(np.abs(lower - self.centers), np.abs(upper - self.centers))
        nearest = np.maximum(np.maximum(lower - self.centers, self.centers - upper), 0)
        narrow = self.sigma * np.exp(-self.sensor_sd / self.sigma)
        wide = self.sigma * np.exp(self.sensor_sd / self.sigma)
        grade_lo = np.exp(-.5 * ((farthest / narrow) ** 2).sum(-1)).clip(1e-30)
        grade_hi = np.exp(-.5 * ((nearest / wide) ** 2).sum(-1)).clip(1e-30)
        consequence_lo, consequence_hi = [], []
        for model in self.models:
            coefficient = model.coef_[:-1]
            offset = model.intercept_ + model.coef_[-1]
            consequence_lo.append(offset + np.minimum(coefficient * lower, coefficient * upper).sum())
            consequence_hi.append(offset + np.maximum(coefficient * lower, coefficient * upper).sum())
        endpoints = []
        for values, left in ((consequence_lo, True), (consequence_hi, False)):
            order = np.argsort(values)
            values = np.asarray(values)[order]
            lo, hi = grade_lo[order], grade_hi[order]
            estimate = float(values.mean())
            for _ in range(100):
                weights = np.where(values <= estimate, hi if left else lo, lo if left else hi)
                updated = float(weights @ values / weights.sum())
                if abs(updated - estimate) < 1e-10:
                    estimate = updated
                    break
                estimate = updated
            endpoints.append(estimate + (-self.residual if left else self.residual))
        return np.nextafter(endpoints[0], -np.inf), np.nextafter(endpoints[1], np.inf)


def fit(frame, spec, args):
    names = spec["variables"]
    train = frame[frame.split == "train"]
    center = train[names].mean().to_numpy()
    scale = train[names].std().to_numpy().clip(1e-6)
    x = (train[names].to_numpy() - center) / scale
    stages = np.asarray(spec["stages"])
    sensor = np.asarray(spec.get("measurement_sd", [0.] * len(names))) / scale
    rng = np.random.default_rng(args.seed)
    structures = []
    for seed in range(args.ensemble):
        ids = rng.integers(0, len(x), len(x))
        graph = np.zeros((len(names), len(names)), int)
        mechanisms = {}
        for j in range(len(names)):
            allowed = np.where((np.arange(len(names)) < j) & ((stages < stages[j]) if args.variant != "no_stages" else True))[0]
            if not len(allowed):
                mechanisms[j] = None
                continue
            selector = Lasso(alpha=args.sparsity, max_iter=5000).fit(x[ids][:, allowed], x[ids, j])
            parents = allowed[np.abs(selector.coef_) > args.edge_threshold]
            if not len(parents):
                mechanisms[j] = None
                continue
            graph[parents, j] = 1
            mechanisms[j] = (parents, FuzzyMechanism(x[ids][:, parents], x[ids, j], args.rules, seed + args.seed, sensor[parents]))
        structures.append((graph, mechanisms))
    return dict(structures=structures, center=center, scale=scale, healthy=np.median(x, axis=0), lower=np.quantile(x, .01, axis=0), upper=np.quantile(x, .99, axis=0), spec=spec)


def propagate(bundle, factual, do, args, perturbation=0.):
    states = []
    for graph, mechanisms in bundle["structures"]:
        low = factual.copy()
        high = factual.copy()
        for j in range(len(factual)):
            if j in do:
                low[j] = high[j] = do[j]
                continue
            mechanism = mechanisms[j]
            if mechanism is None:
                continue
            parents, model = mechanism
            factual_prediction = model.predict(factual[parents][None])[0]
            residual = factual[j] - factual_prediction + perturbation
            parent_low, parent_high = low[parents].copy(), high[parents].copy()
            left, right = model.interval(parent_low, parent_high)
            low[j], high[j] = left + residual, right + residual
            if args.variant == "point":
                midpoint = ((parent_low + parent_high) / 2)[None]
                low[j] = high[j] = model.predict(midpoint)[0] + residual
        states.append(np.stack([low, high]))
    return np.stack(states)


def diagnose(bundle, factual, args):
    y = bundle["spec"]["variables"].index(bundle["spec"]["outcome"])
    scores, effects, conflicts = [], [], []
    for c in range(len(factual)):
        shifted = propagate(bundle, factual, {c: bundle["healthy"][c]}, args)
        effect = factual[y] - shifted[:, :, y]
        midpoint = effect.mean(1)
        width = np.ptp(effect, axis=1).mean()
        conflict = np.std(midpoint)
        abnormality = abs(factual[c] - bundle["healthy"][c])
        score = abs(midpoint.mean()) * (1 - np.exp(-abnormality)) - args.width_weight * width
        if args.variant != "no_conflict":
            score -= args.conflict_weight * conflict
        if c == y:
            score = -np.inf
        scores.append(score)
        effects.append([float(effect.min()), float(effect.max())])
        conflicts.append(float(conflict))
    return np.argsort(-np.asarray(scores)), effects, conflicts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("spec")
    p.add_argument("--output", default="outputs/full")
    p.add_argument("--interventions")
    p.add_argument("--truth-graph")
    p.add_argument("--variant", choices=["full", "point", "no_stages", "no_conflict", "no_abstention"], default="full")
    p.add_argument("--ensemble", type=int, default=10)
    p.add_argument("--rules", type=int, default=4)
    p.add_argument("--sparsity", type=float, default=.02)
    p.add_argument("--edge-threshold", type=float, default=.01)
    p.add_argument("--width-weight", type=float, default=.1)
    p.add_argument("--conflict-weight", type=float, default=.5)
    p.add_argument("--conflict-limit", type=float, default=.25)
    p.add_argument("--confounding", type=float, default=.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    frame = pd.read_csv(args.data)
    spec = json.loads(Path(args.spec).read_text())
    if frame.groupby("unit").split.nunique().max() != 1:
        raise ValueError("Units must be split before fitting")
    if not all((frame.split == s).any() for s in ("train", "cal", "test")):
        raise ValueError("Supply train, cal and test units")
    bundle = fit(frame, spec, args)
    names = spec["variables"]
    outcome = names.index(spec["outcome"])
    threshold = (spec["outcome_limit"] - bundle["center"][outcome]) / bundle["scale"][outcome]
    correction = 0.
    intervention_frame = pd.read_csv(args.interventions) if args.interventions else pd.DataFrame()
    if not intervention_frame.empty:
        errors = []
        for _, case in intervention_frame[intervention_frame.split == "cal"].iterrows():
            row = frame.loc[frame.case == case.case].iloc[0]
            if row.split != "cal":
                raise ValueError("Intervention calibration may use calibration cases only")
            factual = (row[names].to_numpy(float) - bundle["center"]) / bundle["scale"]
            j = names.index(case.variable)
            value = (case.value - bundle["center"][j]) / bundle["scale"][j]
            predicted = propagate(bundle, factual, {j: value}, args)[:, :, outcome]
            target = (case.outcome - bundle["center"][outcome]) / bundle["scale"][outcome]
            errors.append(max(predicted.min() - target, target - predicted.max(), 0))
        if errors:
            correction = float(np.quantile(errors, min(1, np.ceil((len(errors) + 1) * .9) / len(errors)), method="higher"))
    rankings = []
    for _, row in frame[frame.split == "test"].iterrows():
        factual = (row[names].to_numpy(float) - bundle["center"]) / bundle["scale"]
        order, effects, conflicts = diagnose(bundle, factual, args)
        roots = names.index(row.root) if "root" in row and isinstance(row.root, str) and row.root in names else None
        choices = []
        for variable, support in spec.get("actions", {}).items():
            j = names.index(variable)
            if not support.get("effect_source"):
                continue
            for value_raw in np.linspace(support["lower"], support["upper"], args.rules + 1):
                value = (value_raw - bundle["center"][j]) / bundle["scale"][j]
                if not bundle["lower"][j] <= value <= bundle["upper"][j]:
                    continue
                predicted = propagate(bundle, factual, {j: value}, args)[:, :, outcome]
                low, high = float(predicted.min() - correction), float(predicted.max() + correction)
                sensitivity = [propagate(bundle, factual, {j: value}, args, perturbation=s)[:, :, outcome].mean() for s in (-args.confounding, args.confounding)]
                stable = all(s <= threshold for s in sensitivity)
                identified = conflicts[j] <= args.conflict_limit
                accepted = high <= threshold and identified and stable
                if args.variant == "no_abstention":
                    accepted = True
                cost = support.get("cost", 1.) * abs(value_raw - row[variable])
                choices.append(dict(variable=variable, value=float(value_raw), lower=low * bundle["scale"][outcome] + bundle["center"][outcome], upper=high * bundle["scale"][outcome] + bundle["center"][outcome], accepted=bool(accepted), objective=float(cost + predicted.mean())))
        best = min((v for v in choices if v["accepted"]), key=lambda v: v["objective"], default=None)
        rankings.append(dict(case=str(row.case), ranking=[names[j] for j in order], rank=int(np.where(order == roots)[0][0]) + 1 if roots is not None else None, effects=effects, conflict=conflicts, recommendation=best))
    ranks = [r["rank"] for r in rankings if r["rank"] is not None]
    metrics = dict(top1=float(np.mean(np.asarray(ranks) == 1)) if ranks else None, top3=float(np.mean(np.asarray(ranks) <= 3)) if ranks else None, recommendation_coverage=float(np.mean([r["recommendation"] is not None for r in rankings])), calibration_expansion=correction)
    intervention_results = []
    for _, case in intervention_frame[intervention_frame.split == "test"].iterrows() if not intervention_frame.empty else []:
        row = frame.loc[frame.case == case.case].iloc[0]
        if row.split != "test":
            raise ValueError("Test interventions must refer to test cases")
        factual = (row[names].to_numpy(float) - bundle["center"]) / bundle["scale"]
        j = names.index(case.variable)
        value = (case.value - bundle["center"][j]) / bundle["scale"][j]
        predicted = propagate(bundle, factual, {j: value}, args)[:, :, outcome]
        lo = (predicted.min() - correction) * bundle["scale"][outcome] + bundle["center"][outcome]
        hi = (predicted.max() + correction) * bundle["scale"][outcome] + bundle["center"][outcome]
        accepted = hi <= spec["outcome_limit"] and np.std(predicted.mean(1)) <= args.conflict_limit
        intervention_results.append(dict(case=str(case.case), lower=float(lo), upper=float(hi), target=float(case.outcome), accepted=bool(accepted)))
    if intervention_results:
        results = pd.DataFrame(intervention_results)
        covered = (results.lower <= results.target) & (results.target <= results.upper)
        metrics.update(effect_mae=float(abs((results.lower + results.upper) / 2 - results.target).mean()), effect_coverage=float(covered.mean()), effect_width=float((results.upper - results.lower).mean()), abstention_validity=float((results.target[results.accepted] <= spec["outcome_limit"]).mean()) if results.accepted.any() else None)
    graph = np.mean([g for g, _ in bundle["structures"]], axis=0)
    if args.truth_graph:
        truth = np.load(args.truth_graph)
        fitted = graph >= .5
        metrics["structural_hamming_distance"] = int(np.sum(fitted != truth))
        def reachable(a):
            r = a.astype(bool).copy()
            for k in range(len(a)):
                r |= r[:, k, None] & r[k, None, :]
            return r
        truth_path, fitted_path = reachable(truth), reachable(fitted)
        metrics["path_agreement"] = float(np.mean(truth_path == fitted_path))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "model.pkl").open("wb") as f:
        pickle.dump(bundle, f)
    (out / "diagnoses.json").write_text(json.dumps(rankings, indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    pd.DataFrame(intervention_results).to_csv(out / "interventions.csv", index=False)
    np.save(out / "edge_probabilities.npy", graph)
    plt.imshow(graph, vmin=0, vmax=1, cmap="Blues")
    plt.xticks(range(len(names)), names, rotation=90)
    plt.yticks(range(len(names)), names)
    plt.xlabel("Child")
    plt.ylabel("Parent")
    plt.colorbar(label="Bootstrap inclusion")
    plt.tight_layout()
    plt.savefig(out / "graph.png", dpi=180)


if __name__ == "__main__":
    main()
