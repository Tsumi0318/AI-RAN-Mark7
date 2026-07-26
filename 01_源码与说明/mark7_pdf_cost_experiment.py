#!/usr/bin/env python3
"""Mark7: PDF cost structure with semantic-calibrated LLM coordination.

The task-level semantic predictions are reused because the tasks are unchanged.
With --full, the DeepSeek Game Master is called again for every new game state
because the effective D_comp and M parameters have changed.
"""

from __future__ import annotations

import csv
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
BASE_SRC = ROOT / "01_源码与说明" / "base_ai_ran_components.py"
DATA = ROOT / "00_原始数据" / "GenTD26"
OUT = ROOT / "02_表格数据"


def load_base_module():
    spec = importlib.util.spec_from_file_location("base_ai_ran_components", BASE_SRC)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_SRC}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.DATA = DATA
    return module


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"empty rows: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fit_models(points: pd.DataFrame, mu_pool: float) -> tuple[dict[str, Any], pd.DataFrame]:
    x = points.k_equivalent.to_numpy(float)
    y = np.maximum(points.queue_delay_ms.to_numpy(float) / 1000.0, 1e-6)
    # The multi-GPU interpretation uses the pooled service rate for the
    # baseline execution term.  This is explicitly documented because the
    # per-card rate alone would imply a 22 s baseline for this trace.
    t_base = 1.0 / mu_pool

    def pred_b(z: np.ndarray) -> np.ndarray:
        overhead, scale, lam = z
        rho = np.clip(lam * x / mu_pool, 0.0, 0.999999)
        return overhead + (1.0 / mu_pool) * scale / (1.0 - rho)

    fit_b = least_squares(lambda z: np.log(pred_b(z)) - np.log(y), [0.0, 1.0, 0.003], bounds=([0.0, 0.0, 0.0], [10.0, 100.0, 0.2]), loss="soft_l1")
    models = {
        "B_utilization_proxy": {
            "formula": "D=D_overhead+(1/(c*mu_card))*scale/(1-rho), rho=lambda*K/(c*mu_card)",
            "D_overhead_seconds": float(fit_b.x[0]),
            "scale": float(fit_b.x[1]),
            "lambda": float(fit_b.x[2]),
            "prediction": pred_b(fit_b.x),
            "interpretation": "pooled service baseline multiplied by fitted utilization pressure",
        },
    }
    rows = []
    for name, m in models.items():
        p = np.asarray(m.pop("prediction"))
        m.update({
            "r2_seconds": float(r2_score(y, p)),
            "mae_ms": float(mean_absolute_error(y, p) * 1000.0),
            "rmse_ms": float(math.sqrt(mean_squared_error(y, p)) * 1000.0),
            "median_absolute_error_ms": float(np.median(np.abs(y - p)) * 1000.0),
        })
        for k_val, observed, predicted in zip(x, y, p):
            rows.append({"model": name, "k_equivalent": float(k_val), "observed_delay_ms": float(observed * 1000), "predicted_delay_ms": float(predicted * 1000), "residual_ms": float((predicted - observed) * 1000)})
    return models, pd.DataFrame(rows)


class ModelB:
    def __init__(self, overhead, scale, lam, mu_pool): self.overhead, self.scale, self.lam, self.mu_pool = overhead, scale, lam, mu_pool
    def __call__(self, k):
        if k <= 0: return 0.0
        rho = min(max(self.lam * k / self.mu_pool, 0.0), 0.999999)
        return self.overhead + (1.0 / self.mu_pool) * self.scale / (1.0 - rho)


def semantic_calibration(semantic: pd.DataFrame, n: int) -> dict[str, float]:
    """Aggregate task predictions into shared parameters for an exact potential game."""
    return {
        "mean_compute_multiplier": float(semantic.compute_multiplier.iloc[:n].mean()),
        "mean_vram_multiplier": float(semantic.vram_multiplier.iloc[:n].mean()),
    }


def build_pdf_cost_model(m5, c, arrays, semantic, queue, memory, scheduler, n, dcomp, calibration):
    class PdfCostModel(m5.FormulaModel):
        def __init__(self):
            super().__init__(c, arrays, semantic, queue, memory, scheduler, n)
            # Semantic predictions calibrate the shared Dcomp and M parameters;
            # they are not added as an independent player cost.
            self.vreq_fraction = calibration["mean_vram_multiplier"] / c.equivalent_capacity_tasks_assumed
            self.vram_fraction = np.full(n, self.vreq_fraction, dtype=float)
            self.vram_gb = self.vram_fraction * c.vram_capacity_gb_assumed

        def dcomp(self, k: int) -> float:
            return float(dcomp(k))

        def memory_penalty(self, k: int) -> float:
            if k <= 0:
                return 0.0
            exponent = float(np.clip(self.beta * (k * self.vreq_fraction - 1.0), -30, 30))
            return float(self.alpha * math.exp(exponent))

        def center_delta(self, i: int, s_without_i: np.ndarray) -> dict[str, float]:
            k_without = int(s_without_i.sum())
            candidate_k = k_without + 1
            queue_cost = self.dcomp(candidate_k)
            memory_cost = self.memory_penalty(candidate_k)
            return {
                "k_without_i": k_without,
                "candidate_k": candidate_k,
                "vram_without_i_fraction": k_without * self.vreq_fraction,
                "candidate_vram_fraction": candidate_k * self.vreq_fraction,
                "queue_cost": queue_cost,
                "memory_penalty": memory_cost,
                "center_cost_increment": queue_cost + memory_cost,
            }

        def potential(self, s: np.ndarray) -> float:
            k = int(s.sum())
            energy = np.where(s == 1, self.a["e_tx"], self.a["e_loc"]).sum()
            congestion = sum(self.dcomp(j) + self.memory_penalty(j) for j in range(1, k + 1))
            return float(energy + congestion)

        def individual_costs(self, s: np.ndarray) -> np.ndarray:
            k = int(s.sum())
            shared = self.dcomp(k) + self.memory_penalty(k) if k else 0.0
            return np.where(s == 1, self.a["e_tx"] + shared, self.a["e_loc"])

        def vram_load_fraction(self, s: np.ndarray) -> float:
            return float(int(s.sum()) * self.vreq_fraction)

    return PdfCostModel()


def run_game(m5, queue_params: dict[str, Any], model_name: str, dcomp, calibration: dict[str, float], live: bool) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    c = m5.Config()
    pool = m5.load_request_pool(c)
    intents, arrays = m5.build_intents(pool, c)
    semantic_path = OUT / "semantic_resource_predictions_reused.csv"
    if not semantic_path.exists():
        raise FileNotFoundError(f"Missing recorded semantic predictions: {semantic_path}")
    semantic = pd.read_csv(semantic_path)
    memory, _ = m5.fit_vram_barrier(c)
    scheduler = m5.scheduler_summary()
    queue = dict(queue_params)
    queue["lambda_per_task_per_second"] = 0.0
    base = build_pdf_cost_model(m5, c, arrays, semantic, queue, memory, scheduler, c.n_main, dcomp, calibration)
    if live:
        client = m5.DeepSeekClient(c)
        coordinator = m5.DeepSeekGameMaster(client, base, OUT / "deepseek_game_master_cache_mark7.json")
        eq, trace = m5.run_best_response(base, c, c.seed, intents=intents[:c.n_main], coordinator=coordinator, record_trace=True)
        events = pd.DataFrame(coordinator.events)
        new_events = events.loc[events.persistent_cache_hit == 0].drop_duplicates("state_hash")
        recorded_api = pd.DataFrame(list(coordinator.cache.values()))
        latencies = recorded_api.latency_ms.dropna().to_numpy(float)
        overhead = {
            "mode": "live_deepseek_game_master",
            "requested_model": client.model,
            "resolved_models": ";".join(sorted(set(recorded_api.resolved_model.dropna().astype(str)))),
            "semantic_predictions_reused": len(semantic),
            "semantic_real_api_calls_this_run": 0,
            "game_logical_calls": len(events),
            "game_real_api_calls_this_run": len(new_events),
            "game_cache_hits": int(events.persistent_cache_hit.sum()),
            "recorded_unique_real_api_calls": len(recorded_api),
            "recorded_total_tokens": int(recorded_api.total_tokens.fillna(0).sum()),
            "recorded_mean_latency_ms": float(latencies.mean()) if len(latencies) else 0.0,
            "recorded_p95_latency_ms": float(np.quantile(latencies, 0.95)) if len(latencies) else 0.0,
            "numeric_mismatches_corrected": int((events.llm_numeric_within_tolerance == 0).sum()),
            "api_key_saved": False,
        }
    else:
        eq, trace = m5.run_best_response(base, c, c.seed, record_trace=True)
        events = pd.DataFrame([{"mode": "deterministic_no_live_llm"}])
        overhead = {"mode": "deterministic_no_live_llm", "semantic_predictions_reused": len(semantic), "game_logical_calls": 0, "game_real_api_calls_this_run": 0, "total_tokens_this_run": 0, "api_key_saved": False}
    metrics = base.metrics(eq)
    metrics.update({"model": model_name, "updates": len(trace)-1, "strategy_changes": sum(int(r["changed"]) for r in trace), "verified_psne": int(base.is_psne(eq))})
    strategy = [{"model": model_name, "node": i, "s_star": int(eq[i]), "meaning": "offload" if eq[i] else "local"} for i in range(c.n_main)]
    trace_df = pd.DataFrame(trace).assign(model=model_name)
    return metrics, pd.DataFrame(strategy), trace_df, events, overhead


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true", help="Run the live DeepSeek Game Master with the selected model")
    mode.add_argument("--deterministic", action="store_true", help="Run without live API calls for local verification")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    m5 = load_base_module()
    q5, points = m5.fit_multigpu_delay(m5.Config())
    semantic = pd.read_csv(OUT / "semantic_resource_predictions_reused.csv")
    calibration = semantic_calibration(semantic, m5.Config().n_main)
    mu_card_effective = float(q5["mu_card_per_second"]) / calibration["mean_compute_multiplier"]
    mu_pool_effective = float(q5["card_count_c"]) * mu_card_effective
    models, fit_df = fit_models(points, mu_pool_effective)
    for model in models.values():
        model.update({
            **calibration,
            "mu_card_data_per_second": float(q5["mu_card_per_second"]),
            "mu_card_effective_per_second": mu_card_effective,
            "mu_pool_effective_per_second": mu_pool_effective,
            "vreq_fraction_effective": calibration["mean_vram_multiplier"] / m5.Config().equivalent_capacity_tasks_assumed,
            "cost_structure": "PDF: E_i plus s_i times shared Dcomp(K) and M(K); no independent C_compute",
            "semantic_aggregation": "arithmetic mean over the 100 fixed main-game tasks",
        })
    write_csv(OUT / "selected_model_B_fit_points.csv", fit_df.to_dict("records"))
    write_csv(OUT / "selected_model_B_metrics.csv", [{"model": k, **{a: b for a, b in v.items() if a != "prediction"}} for k, v in models.items()])
    write_csv(OUT / "semantic_parameter_calibration.csv", [{
        **calibration,
        "mu_card_data_per_second": float(q5["mu_card_per_second"]),
        "mu_card_effective_per_second": mu_card_effective,
        "mu_pool_effective_per_second": mu_pool_effective,
        "vreq_base_fraction": 1.0 / m5.Config().equivalent_capacity_tasks_assumed,
        "vreq_effective_fraction": calibration["mean_vram_multiplier"] / m5.Config().equivalent_capacity_tasks_assumed,
        "aggregation_scope": "first 100 fixed main-game tasks",
        "reason": "preserve one shared Dcomp(K) and M(K) for the PDF exact-potential structure",
    }])

    base_q = {"card_count_c": q5["card_count_c"], "mu_card_per_second": mu_card_effective, "lambda_per_task_per_second": 0.0}
    gb = models["B_utilization_proxy"]
    metrics, strategies, traces = [], [], []
    met, st, tr, events, overhead = run_game(m5, base_q, "B_utilization_proxy_pdf_cost", ModelB(gb["D_overhead_seconds"], gb["scale"], gb["lambda"], mu_pool_effective), calibration, live=args.full)
    metrics.append(met); strategies.append(st); traces.append(tr)
    write_csv(OUT / "model_game_metrics.csv", metrics)
    pd.concat(strategies, ignore_index=True).to_csv(OUT / "model_equilibrium_strategies.csv", index=False)
    pd.concat(traces, ignore_index=True).to_csv(OUT / "model_convergence_traces.csv", index=False)
    event_name = "llm_feedback_events.csv" if args.full else "deterministic_feedback_events.csv"
    overhead_name = "llm_coordination_overhead.csv" if args.full else "deterministic_coordination_overhead.csv"
    events.to_csv(OUT / event_name, index=False)
    write_csv(OUT / overhead_name, [overhead])
    pool = m5.load_request_pool(m5.Config())
    intents, _ = m5.build_intents(pool, m5.Config())
    write_csv(OUT / "semantic_intents.csv", intents)
    selected = "B_utilization_proxy"
    (OUT / "selected_model.json").write_text(json.dumps({"selected_model": selected, "selection_rule": "scheme B retained from the preceding model comparison", "reason": "Mark7 changes the cost integration, not the previously selected queue family"}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status":"completed", "experiment":"Mark7 PDF cost structure with semantic-calibrated utilization-proxy queue and live DeepSeek Game Master" if args.full else "Mark7 PDF cost structure deterministic run", "selected_model": selected, "semantic_parameter_calibration":calibration, "models":models, "game_metrics":metrics, "llm_overhead":overhead, "claim_boundary":"Finite-game PSNE verification; not global optimum or deployment validation."}
    summary_name = "run_summary.json" if args.full else "deterministic_run_summary.json"
    (OUT / summary_name).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
