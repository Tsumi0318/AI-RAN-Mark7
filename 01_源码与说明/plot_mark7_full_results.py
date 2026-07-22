#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "02_表格数据"
F = ROOT / "03_结果图"
for folder in ["PNG", "PDF", "SVG", "TIFF"]:
    (F / folder).mkdir(parents=True, exist_ok=True)
BLUE, RED, TEAL, GRAY = "#3f6fb5", "#c94f53", "#278c82", "#777777"


def save(fig, name):
    fig.tight_layout()
    for ext in ["png", "pdf", "svg", "tiff"]:
        fig.savefig(F / ext.upper() / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


trace = pd.read_csv(T / "model_convergence_traces.csv")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
axes[0].plot(trace.iteration, trace.k_offload, color=BLUE, lw=1.7)
axes[0].set(xlabel="Best-response update", ylabel="Offloaded nodes K", title="A  Strategy convergence")
axes[1].plot(trace.iteration, trace.potential, color=TEAL, lw=1.7)
axes[1].set(xlabel="Best-response update", ylabel="Potential function", title="B  Potential descent")
for ax in axes: ax.grid(alpha=.2)
save(fig, "02_convergence_stability")

alg = pd.read_csv(T / "algorithm_comparison.csv")
fig, ax = plt.subplots(figsize=(8.2, 4.8))
label_map = {
    "deepseek_best_response_psne": "LLM best response",
    "all_local": "All Local",
    "all_offload": "All Offload",
    "greedy_local_information_only": "Greedy: local information only",
    "random_p_0.5_mean": "Random p=0.5 mean",
}
labels = alg.algorithm.map(label_map)
bars = ax.barh(labels, alg.system_total_cost_normalized, color=[BLUE, GRAY, RED, "#8b8b8b", TEAL])
ax.set_xscale("log")
ax.set_xlabel("Normalized system total cost (log scale)")
ax.set_title("System-cost comparison", loc="left", fontweight="bold")
ax.invert_yaxis()
ax.grid(axis="x", alpha=.2, which="both")
for b, value in zip(bars, alg.system_total_cost_normalized):
    ax.text(value * 1.08, b.get_y() + b.get_height()/2, f"{value:.1f}", ha="left", va="center", fontsize=8)
save(fig, "03_system_cost_comparison")

mem = pd.read_csv(T / "memory_violation_sweep.csv")
fig, ax = plt.subplots(figsize=(7.8, 4.2))
combined = [
    ("deepseek_best_response_psne", "LLM best response / All Local", BLUE),
    ("all_offload", "All Offload / Greedy", RED),
    ("random_p_0.5", "Random p=0.5", "#8e63bd"),
]
for name, label, color in combined:
    g = mem.loc[mem.algorithm == name]
    ax.plot(g.n_nodes, g.memory_violation_rate_simulated, marker="o", ms=3, lw=1.6, color=color, label=label)
ax.set(xlabel="Number of nodes N", ylabel="Simulated memory violation rate", ylim=(-.03, 1.03))
ax.set_title("Memory-capacity stress test", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=7)
ax.grid(alpha=.2)
save(fig, "04_memory_violation_rate")

o = pd.read_csv(T / "llm_coordination_overhead.csv").iloc[0]
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
axes[0].bar(["Logical", "Real API", "Cache hits"], [o.game_logical_calls, o.game_real_api_calls_this_run, o.game_cache_hits], color=[BLUE, RED, GRAY])
axes[0].set_ylabel("Calls / states"); axes[0].set_title("A  Coordination volume")
axes[1].bar(["Mean", "P95"], [o.recorded_mean_latency_ms, o.recorded_p95_latency_ms], color=[TEAL, RED])
axes[1].set_ylabel("Latency (ms)"); axes[1].set_title("B  Measured API latency")
axes[1].text(.03, .95, f"Tokens: {int(o.recorded_total_tokens):,}\nModel: {o.resolved_models}", transform=axes[1].transAxes, va="top", fontsize=8)
save(fig, "05_llm_coordination_overhead")
