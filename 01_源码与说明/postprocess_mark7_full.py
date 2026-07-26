#!/usr/bin/env python3
"""Generate the complete non-API validation outputs for the finished Mark7 run."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "01_源码与说明" / "mark7_pdf_cost_experiment.py"
OUT = ROOT / "02_表格数据"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    mk7 = load(SRC, "mark7_selected")
    m5 = mk7.load_base_module()
    c = m5.Config()
    queue5, _ = m5.fit_multigpu_delay(c)
    memory, vram_points = m5.fit_vram_barrier(c)
    scheduler = m5.scheduler_summary()
    selected = pd.read_csv(OUT / "queue_model_metrics.csv").iloc[0]
    queue_stub = {"card_count_c": queue5["card_count_c"], "mu_card_per_second": queue5["mu_card_per_second"], "lambda_per_task_per_second": 0.0}
    pool = m5.load_request_pool(c)
    _, arrays = m5.build_intents(pool, c)
    semantic_path = OUT / "semantic_resource_predictions.csv"
    if not semantic_path.exists():
        raise FileNotFoundError(
            "Run mark7_pdf_cost_experiment.py --full first to generate "
            "semantic_resource_predictions.csv from the Alibaba task pool."
        )
    semantic = pd.read_csv(semantic_path)
    calibration = mk7.semantic_calibration(semantic, c.n_main)
    mu_card_effective = float(selected.mu_card_effective_per_second)
    queue_stub["mu_card_per_second"] = mu_card_effective
    dcomp_args = (selected.D_overhead_seconds, selected.scale, selected["lambda"], selected.mu_pool_effective_per_second)

    def make_model(n: int):
        return mk7.build_pdf_cost_model(m5, c, arrays, semantic, queue_stub, memory, scheduler, n, mk7.PooledUtilizationModel(*dcomp_args), calibration)

    main_model = make_model(c.n_main)
    strategy_df = pd.read_csv(OUT / "model_equilibrium_strategies.csv")
    equilibrium = strategy_df.s_star.to_numpy(np.int8)
    m5.write_csv(OUT / "algorithm_comparison.csv", m5.baseline_rows(c, main_model, equilibrium))
    m5.write_csv(OUT / "pareto_front.csv", m5.pareto_rows(main_model))
    m5.write_csv(OUT / "formula_audit.csv", m5.formula_audit(main_model))

    rows = []
    for n in range(c.sweep_n_min, c.sweep_n_max + 1, c.sweep_n_step):
        model = make_model(n)
        counts = {name: 0 for name in ["deepseek_best_response_psne", "all_local", "all_offload", "random_p_0.5", "greedy_local_information_only"]}
        mean_k = {name: [] for name in counts}
        for trial in range(c.sweep_trials):
            eq, _ = m5.run_best_response(model, c, c.seed + n * 100 + trial, record_trace=False)
            rng = np.random.default_rng(c.seed + n * 1000 + trial)
            strategies = {
                "deepseek_best_response_psne": eq,
                "all_local": np.zeros(n, dtype=np.int8),
                "all_offload": np.ones(n, dtype=np.int8),
                "random_p_0.5": rng.integers(0, 2, n, dtype=np.int8),
                "greedy_local_information_only": (model.a["e_tx"] < model.a["e_loc"]).astype(np.int8),
            }
            for name, strategy in strategies.items():
                counts[name] += int(model.vram_load_fraction(strategy) > 1.0)
                mean_k[name].append(int(strategy.sum()))
        for name in counts:
            rows.append({"n_nodes": n, "algorithm": name, "trials": c.sweep_trials, "memory_violation_rate_simulated": counts[name] / c.sweep_trials, "mean_k_offload": float(np.mean(mean_k[name])), "capacity_gb_assumed": c.vram_capacity_gb_assumed, "measurement_status": "software_simulator_proxy_not_physical_oom"})
    m5.write_csv(OUT / "memory_violation_sweep.csv", rows)
    m5.write_csv(OUT / "data_manifest.csv", m5.data_manifest())
    m5.write_csv(OUT / "scheduler_latency_summary.csv", [scheduler])
    m5.write_csv(OUT / "vram_barrier_fit_points.csv", vram_points.to_dict("records"))
    assumption_rows = m5.assumptions_rows(c, queue5, memory)
    assumption_rows.extend([
        {"parameter": "llm_thinking_mode", "value": "disabled", "status": "API_configuration"},
        {"parameter": "mean_compute_multiplier", "value": calibration["mean_compute_multiplier"], "status": "DeepSeek prediction mean over 100 tasks"},
        {"parameter": "mean_vram_multiplier", "value": calibration["mean_vram_multiplier"], "status": "DeepSeek prediction mean over 100 tasks"},
        {"parameter": "mu_card_effective", "value": mu_card_effective, "status": "mu_card_data divided by mean_compute_multiplier"},
        {"parameter": "vreq_effective_fraction", "value": calibration["mean_vram_multiplier"] / c.equivalent_capacity_tasks_assumed, "status": "base slot fraction multiplied by mean_vram_multiplier"},
    ])
    m5.write_csv(OUT / "assumptions_and_parameters.csv", assumption_rows)
    print(json.dumps({"status":"completed", "outputs":["algorithm_comparison.csv", "pareto_front.csv", "formula_audit.csv", "memory_violation_sweep.csv"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
