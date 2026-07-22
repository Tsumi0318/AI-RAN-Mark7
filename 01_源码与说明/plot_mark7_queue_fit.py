#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "02_表格数据"
FIG = ROOT / "03_结果图"
for folder in ["PNG", "PDF", "SVG", "TIFF"]:
    (FIG / folder).mkdir(parents=True, exist_ok=True)
df = pd.read_csv(TABLES / "selected_model_B_fit_points.csv")
metrics = pd.read_csv(TABLES / "selected_model_B_metrics.csv")
model = "B_utilization_proxy"
color = "#4c72b0"
label = "B: utilization proxy (selected)"
fig, ax = plt.subplots(figsize=(7.4, 4.5))
g = df.loc[df.model == model].copy()
ax.scatter(g.k_equivalent, g.observed_delay_ms, s=13, alpha=0.22, color="#777777", label="Matched observations")
fit = g.groupby("k_equivalent", as_index=False).predicted_delay_ms.mean().sort_values("k_equivalent")
row = metrics.loc[metrics.model == model].iloc[0]
ax.plot(fit.k_equivalent, fit.predicted_delay_ms, lw=2.0, color=color, label=f"{label} (R²={row.r2_seconds:.3f})")
ax.set_xlabel("Equivalent congestion K (scaled proxy)")
ax.set_ylabel("Observed / fitted delay (ms)")
ax.set_title("Mark7 queue model: scheme B", loc="left", fontweight="bold")
ax.legend(frameon=False)
ax.grid(alpha=0.2)
fig.tight_layout()
for ext, dpi in [("png", 300), ("pdf", 300), ("svg", 300), ("tiff", 300)]:
    fig.savefig(FIG / ext.upper() / f"01_selected_model_B.{ext}", dpi=dpi, bbox_inches="tight")
plt.close(fig)
