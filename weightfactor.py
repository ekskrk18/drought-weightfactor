"""
Revised DPI Weight Optimization Analysis
========================================

Main fixes compared with the first version:
1) DPI components are calculated using a longer pre-event calibration window
   (default: 180 days), while the performance metrics are evaluated only in
   the last 90 days before the drought event. This prevents AUC from becoming
   all-NaN when P_WINDOW=90.
2) Pareto plotting no longer drops all rows when one metric has NaNs.
3) Contour/ternary plots are skipped safely when valid points are insufficient.
4) Vertex labels and output figure titles are cleaned for publication use.

DPI_w = wP*(-Z_P) + wV*(Z_VPD) + wW*(Z_WS),  wP+wV+wW=1
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS
# ============================================================
EVENTS_CSV = Path(r"E:\20260411\00 KONKUK\02 Papers\01 SCIE\32th Drought (Weight factor)\python\drought_events.csv")
KMA_DIR = Path(r"E:\20260411\00 KONKUK\02 Papers\01 SCIE\32th Drought (Weight factor)\python\kma_daily_outputs")
OUT_DIR = Path(r"E:\20260411\00 KONKUK\02 Papers\01 SCIE\32th Drought (Weight factor)\python\dpi_weight_optimization_revised")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_IDS = list(range(1, 11))

# Fixed temporal windows used for weight optimization.
P_WINDOW = 90       # precipitation accumulation window, days
V_WINDOW = 7        # VPD moving-average window, days
W_WINDOW = 7        # wind-speed moving-average window, days

# Key correction: calculate rolling components and z-scores using a longer
# pre-event calibration window, then evaluate metrics in the last 90 days.
COMPONENT_LOOKBACK_DAYS = 180
METRIC_LOOKBACK_DAYS = 90

WEIGHT_STEP = 0.05  # 0.05 -> 231 combinations; 0.02 -> smoother but slower
DPI_THRESHOLD = 0.9
CONSEC_DAYS = 3

OBJECTIVE_WEIGHTS = {
    "lead": 1.0,
    "auc": 1.0,
    "duration": 1.0,
    "stability": 0.5,
}

REQUIRED_KMA_COLS = ["date_kst", "rn_day_2355", "ta_0900", "hm_0900", "ws_10m_0900"]

# ============================================================
# Basic functions
# ============================================================
def compute_vpd_kpa(T_c: pd.Series, RH_pct: pd.Series) -> pd.Series:
    T_c = pd.to_numeric(T_c, errors="coerce")
    RH_pct = pd.to_numeric(RH_pct, errors="coerce")
    es = 0.6108 * np.exp((17.27 * T_c) / (T_c + 237.3))
    return es * (1.0 - RH_pct / 100.0)


def zscore_nan(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    vals = s.to_numpy(dtype=float)
    valid = np.isfinite(vals)
    if valid.sum() == 0:
        return pd.Series(np.nan, index=s.index, dtype=float)
    mu = np.nanmean(vals)
    sd = np.nanstd(vals, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        out = np.zeros(len(s), dtype=float)
        out[~valid] = np.nan
        return pd.Series(out, index=s.index)
    return (s - mu) / sd


def zscore_column(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    vals = s.to_numpy(dtype=float)
    valid = np.isfinite(vals)
    if valid.sum() == 0:
        z = pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    else:
        mu = np.nanmean(vals)
        sd = np.nanstd(vals, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            z = pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
        else:
            z = (s - mu) / sd
            z = z.fillna(0.0)
    return z if higher_is_better else -z


def min_periods_for_window(win: int, kind: str) -> int:
    return max(3, win // 3) if kind == "sum" else max(3, win // 2)


def generate_weight_grid(step: float = 0.05) -> pd.DataFrame:
    vals = np.round(np.arange(0.0, 1.0 + step / 2, step), 10)
    rows = []
    for wP in vals:
        for wV in vals:
            wW = 1.0 - wP - wV
            if wW < -1e-9 or wW > 1.0 + 1e-9:
                continue
            rows.append({"wP": float(wP), "wV": float(wV), "wW": round(float(wW), 10)})
    return pd.DataFrame(rows)


def load_events(path: Path) -> pd.DataFrame:
    ev = pd.read_csv(path, encoding="utf-8-sig")
    required = {"id", "event_time"}
    if not required.issubset(ev.columns):
        raise ValueError(f"events csv requires columns {required}, current={ev.columns.tolist()}")
    ev["id"] = pd.to_numeric(ev["id"], errors="coerce")
    ev["event_time"] = pd.to_datetime(ev["event_time"], errors="coerce")
    ev = ev.dropna(subset=["id", "event_time"]).copy()
    ev["id"] = ev["id"].astype(int)
    ev["event_date"] = ev["event_time"].dt.floor("D")
    return ev


def load_kma_event(kma_dir: Path, event_id: int) -> pd.DataFrame:
    path = kma_dir / f"kma_daily_event_{event_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing KMA file: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in REQUIRED_KMA_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    df["date_kst"] = pd.to_datetime(df["date_kst"], errors="coerce")
    return df.dropna(subset=["date_kst"]).sort_values("date_kst").reset_index(drop=True)

# ============================================================
# DPI component calculation
# ============================================================
def compute_components(df: pd.DataFrame, event_date: pd.Timestamp) -> pd.DataFrame:
    start_date = event_date - pd.Timedelta(days=COMPONENT_LOOKBACK_DAYS)
    pre = df[(df["date_kst"] >= start_date) & (df["date_kst"] < event_date)].copy()
    pre = pre.sort_values("date_kst").reset_index(drop=True)
    if pre.empty:
        return pd.DataFrame(columns=["date_kst", "CP", "CV", "CW"])

    p_acc = pd.to_numeric(pre["rn_day_2355"], errors="coerce").rolling(
        P_WINDOW, min_periods=min_periods_for_window(P_WINDOW, "sum")
    ).sum()
    vpd = compute_vpd_kpa(pre["ta_0900"], pre["hm_0900"])
    vpd_mean = vpd.rolling(V_WINDOW, min_periods=min_periods_for_window(V_WINDOW, "mean")).mean()
    ws_mean = pd.to_numeric(pre["ws_10m_0900"], errors="coerce").rolling(
        W_WINDOW, min_periods=min_periods_for_window(W_WINDOW, "mean")
    ).mean()

    comp = pd.DataFrame({
        "date_kst": pre["date_kst"],
        "CP": -zscore_nan(p_acc),
        "CV": zscore_nan(vpd_mean),
        "CW": zscore_nan(ws_mean),
    })
    return comp


def build_weighted_dpi(comp: pd.DataFrame, wP: float, wV: float, wW: float) -> pd.Series:
    return wP * comp["CP"] + wV * comp["CV"] + wW * comp["CW"]

# ============================================================
# Performance metrics in the final METRIC_LOOKBACK_DAYS only
# ============================================================
def metric_window(comp: pd.DataFrame, dpi: pd.Series, event_date: pd.Timestamp) -> pd.DataFrame:
    start = event_date - pd.Timedelta(days=METRIC_LOOKBACK_DAYS)
    tmp = pd.DataFrame({"date_kst": comp["date_kst"], "dpi": dpi})
    return tmp[(tmp["date_kst"] >= start) & (tmp["date_kst"] < event_date)].copy()


def warning_lead_time_days(win: pd.DataFrame) -> float:
    win = win.dropna(subset=["dpi"]).copy()
    if win.shape[0] < CONSEC_DAYS:
        return np.nan
    above = (win["dpi"].to_numpy() >= DPI_THRESHOLD).astype(int)
    run = np.convolve(above, np.ones(CONSEC_DAYS, dtype=int), mode="valid")
    idx = np.where(run >= CONSEC_DAYS)[0]
    if len(idx) == 0:
        return np.nan
    first_cross = pd.Timestamp(win.iloc[idx[0]]["date_kst"]).floor("D")
    event_date = pd.Timestamp(win["event_date"].iloc[0]).floor("D")
    return float((event_date - first_cross).days)


def auc_positive(win: pd.DataFrame) -> float:
    x = pd.to_numeric(win["dpi"], errors="coerce").to_numpy(dtype=float)
    if np.isfinite(x).sum() < int(0.7 * METRIC_LOOKBACK_DAYS):
        return np.nan
    return float(np.nansum(np.maximum(0.0, x)))


def duration_over_threshold(win: pd.DataFrame) -> float:
    x = pd.to_numeric(win["dpi"], errors="coerce").dropna()
    if len(x) < int(0.7 * METRIC_LOOKBACK_DAYS):
        return np.nan
    return float((x >= DPI_THRESHOLD).sum())


def evaluate_one_event(comp: pd.DataFrame, event_date: pd.Timestamp, weights_df: pd.DataFrame) -> List[Dict[str, float]]:
    records = []
    for row in weights_df.itertuples(index=False):
        dpi = build_weighted_dpi(comp, row.wP, row.wV, row.wW)
        win = metric_window(comp, dpi, event_date)
        win["event_date"] = event_date
        records.append({
            "wP": row.wP,
            "wV": row.wV,
            "wW": row.wW,
            "lead_time": warning_lead_time_days(win),
            "auc": auc_positive(win),
            "duration": duration_over_threshold(win),
        })
    return records

# ============================================================
# Ranking and Pareto
# ============================================================
def summarize_weights(metrics_event_df: pd.DataFrame) -> pd.DataFrame:
    g = metrics_event_df.groupby(["wP", "wV", "wW"], as_index=False)
    out = g.agg(
        mean_lead=("lead_time", "mean"),
        std_lead=("lead_time", "std"),
        mean_auc=("auc", "mean"),
        std_auc=("auc", "std"),
        mean_duration=("duration", "mean"),
        std_duration=("duration", "std"),
        n_events=("event_id", "nunique"),
    )

    out["z_lead"] = zscore_column(out["mean_lead"], True)
    out["z_auc"] = zscore_column(out["mean_auc"], True)
    out["z_duration"] = zscore_column(out["mean_duration"], True)
    out["std_mean"] = out[["std_lead", "std_auc", "std_duration"]].mean(axis=1, skipna=True)
    out["z_stability"] = zscore_column(out["std_mean"], higher_is_better=False)
    out["composite_score"] = (
        OBJECTIVE_WEIGHTS["lead"] * out["z_lead"]
        + OBJECTIVE_WEIGHTS["auc"] * out["z_auc"]
        + OBJECTIVE_WEIGHTS["duration"] * out["z_duration"]
        + OBJECTIVE_WEIGHTS["stability"] * out["z_stability"]
    )
    out = out.sort_values("composite_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def pareto_mask(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    vals = df[cols].to_numpy(dtype=float)
    n = vals.shape[0]
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        dominates_i = np.all(vals >= vals[i], axis=1) & np.any(vals > vals[i], axis=1)
        dominates_i[i] = False
        if np.any(dominates_i):
            is_pareto[i] = False
    return is_pareto

# ============================================================
# Plotting utilities
# ============================================================
def barycentric_to_cartesian(wP, wV, wW) -> Tuple[np.ndarray, np.ndarray]:
    # Triangle vertices: VPD=(0,0), Precipitation=(1,0), Wind=(0.5,sqrt(3)/2)
    wP = np.asarray(wP, dtype=float)
    wV = np.asarray(wV, dtype=float)
    wW = np.asarray(wW, dtype=float)
    x = wP * 1.0 + wV * 0.0 + wW * 0.5
    y = wW * (np.sqrt(3) / 2.0)
    return x, y


def draw_triangle(ax):
    h = np.sqrt(3) / 2.0
    ax.plot([0, 1], [0, 0], color="black", linewidth=1.5)
    ax.plot([1, 0.5], [0, h], color="black", linewidth=1.5)
    ax.plot([0.5, 0], [h, 0], color="black", linewidth=1.5)
    ax.text(1.03, -0.025, "wP\nPrecipitation", ha="center", va="top", fontsize=10)
    ax.text(-0.03, -0.025, "wV\nVPD", ha="center", va="top", fontsize=10)
    ax.text(0.5, h + 0.035, "wW\nWind", ha="center", va="bottom", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, h + 0.09)
    ax.axis("off")


def save_top_table(df: pd.DataFrame, out_path: Path, n: int = 20) -> None:
    cols = ["rank", "wP", "wV", "wW", "composite_score", "mean_lead", "mean_auc", "mean_duration", "std_mean", "n_events"]
    df[cols].head(n).to_csv(out_path, index=False, encoding="utf-8-sig")


def plot_weight_sensitivity(summary: pd.DataFrame, out_dir: Path) -> None:
    labels = {"wP": "Precipitation weight", "wV": "VPD weight", "wW": "Wind-speed weight"}
    for col, label in labels.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(summary[col], summary["composite_score"], alpha=0.75, s=35)
        ax.set_xlabel(label)
        ax.set_ylabel("Composite performance score")
        ax.set_title(f"Sensitivity of performance to {label}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"sensitivity_{col}_composite.png", dpi=300)
        plt.close(fig)


def plot_tornado_like_sensitivity(summary: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for col, name in [("wP", "Precipitation"), ("wV", "VPD"), ("wW", "Wind speed")]:
        valid = summary[[col, "composite_score"]].dropna()
        corr = valid[col].corr(valid["composite_score"]) if len(valid) >= 3 else np.nan
        rows.append((name, abs(corr) if np.isfinite(corr) else 0.0))
    rows = sorted(rows, key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([r[0] for r in rows], [r[1] for r in rows])
    ax.set_xlabel("Absolute correlation with composite score")
    ax.set_title("Weight sensitivity importance")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_pareto_front(summary: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    # Use metrics that actually contain values. AUC should be available after the revision,
    # but this fallback prevents an empty Pareto plot.
    candidate_cols = ["mean_lead", "mean_auc", "mean_duration"]
    available_cols = [c for c in candidate_cols if summary[c].notna().sum() >= 3]
    if len(available_cols) < 2:
        print("[WARN] Pareto plot skipped: fewer than two valid objective metrics.")
        return pd.DataFrame()

    valid = summary.dropna(subset=available_cols).copy()
    valid["is_pareto"] = pareto_mask(valid, available_cols)
    pareto = valid[valid["is_pareto"]].copy()

    x_col = "mean_lead"
    y_col = "mean_auc" if "mean_auc" in available_cols else "mean_duration"
    size_col = "mean_duration" if "mean_duration" in available_cols else available_cols[-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(valid[x_col], valid[y_col], c=valid["wP"], s=35, alpha=0.65, label="All weight combinations")
    ax.scatter(pareto[x_col], pareto[y_col], facecolors="none", edgecolors="black", s=100, linewidths=1.5, label="Pareto-optimal")
    for _, r in pareto.sort_values("composite_score", ascending=False).head(5).iterrows():
        ax.annotate(f"({r['wP']:.2f},{r['wV']:.2f},{r['wW']:.2f})", (r[x_col], r[y_col]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Mean warning lead time (days)")
    ax.set_ylabel("Mean AUC" if y_col == "mean_auc" else "Mean duration (days)")
    ax.set_title("Pareto front for DPI weight optimization")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Precipitation-deficit weight, wP")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return valid


def plot_response_surface_contour(summary: pd.DataFrame, out_path: Path) -> None:
    df = summary.dropna(subset=["composite_score"]).copy()
    if df.shape[0] < 3:
        print("[WARN] Response surface skipped: insufficient valid points.")
        return
    x = df["wP"].to_numpy()
    y = df["wV"].to_numpy()
    z = df["composite_score"].to_numpy()
    best = df.loc[df["composite_score"].idxmax()]
    top10_thr = df["composite_score"].quantile(0.90)
    top10 = df[df["composite_score"] >= top10_thr]

    fig, ax = plt.subplots(figsize=(8, 6))
    tcf = ax.tricontourf(x, y, z, levels=18)
    ax.tricontour(x, y, z, levels=10, linewidths=0.4)
    ax.scatter(top10["wP"], top10["wV"], facecolors="none", edgecolors="white", s=35, linewidths=0.8, label="Top 10% region")
    ax.scatter(best["wP"], best["wV"], marker="*", s=260, edgecolor="black", label="Best weight")
    ax.plot([0, 1], [1, 0], color="black", linewidth=1.2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("wP: precipitation-deficit weight")
    ax.set_ylabel("wV: VPD weight")
    ax.set_title("Response surface of composite performance\n(wW = 1 - wP - wV)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    cbar = fig.colorbar(tcf, ax=ax)
    cbar.set_label("Composite performance score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_ternary_response_map(summary: pd.DataFrame, out_path: Path) -> None:
    df = summary.dropna(subset=["composite_score"]).copy()
    if df.shape[0] < 3:
        print("[WARN] Ternary map skipped: insufficient valid points.")
        return
    x, y = barycentric_to_cartesian(df["wP"], df["wV"], df["wW"])
    z = df["composite_score"].to_numpy()
    best = df.loc[df["composite_score"].idxmax()]
    bx, by = barycentric_to_cartesian([best["wP"]], [best["wV"]], [best["wW"]])
    top10_thr = df["composite_score"].quantile(0.90)
    top10 = df[df["composite_score"] >= top10_thr]
    tx, ty = barycentric_to_cartesian(top10["wP"], top10["wV"], top10["wW"])

    fig, ax = plt.subplots(figsize=(8, 7))
    tcf = ax.tricontourf(x, y, z, levels=18)
    ax.tricontour(x, y, z, levels=10, linewidths=0.4)
    ax.scatter(tx, ty, facecolors="none", edgecolors="white", s=35, linewidths=0.8, label="Top 10% region")
    ax.scatter(bx, by, marker="*", s=300, edgecolor="black", label="Best weight")
    draw_triangle(ax)
    ax.set_title("Ternary response surface for optimized DPI weights")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2)
    cbar = fig.colorbar(tcf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Composite performance score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_response_maps(summary: pd.DataFrame, out_dir: Path) -> None:
    metric_info = [
        ("mean_lead", "Mean warning lead time (days)", "response_map_lead_time.png"),
        ("mean_auc", "Mean AUC", "response_map_auc.png"),
        ("mean_duration", "Mean duration above threshold (days)", "response_map_duration.png"),
        ("composite_score", "Composite performance score", "response_map_composite.png"),
    ]
    for metric, title, fname in metric_info:
        df = summary.dropna(subset=[metric]).copy()
        if df.shape[0] < 3:
            print(f"[WARN] {fname} skipped: insufficient valid points for {metric}.")
            continue
        x, y = barycentric_to_cartesian(df["wP"], df["wV"], df["wW"])
        z = df[metric].to_numpy()
        fig, ax = plt.subplots(figsize=(8, 7))
        tcf = ax.tricontourf(x, y, z, levels=18)
        ax.tricontour(x, y, z, levels=10, linewidths=0.4)
        draw_triangle(ax)
        ax.set_title(title)
        cbar = fig.colorbar(tcf, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(title)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)

# ============================================================
# Main workflow
# ============================================================
def main() -> None:
    print("[START] Revised DPI weight optimization")
    print(f"Output directory: {OUT_DIR}")
    print(f"Component lookback = {COMPONENT_LOOKBACK_DAYS} days; metric lookback = {METRIC_LOOKBACK_DAYS} days")

    events = load_events(EVENTS_CSV)
    event_date_map = dict(zip(events["id"], events["event_date"]))
    weights_df = generate_weight_grid(WEIGHT_STEP)
    weights_df.to_csv(OUT_DIR / "weight_grid.csv", index=False, encoding="utf-8-sig")
    print(f"[INFO] Number of weight combinations: {len(weights_df)}")

    all_records = []
    for eid in EVENT_IDS:
        if eid not in event_date_map:
            print(f"[WARN] Event id {eid} not found in events CSV. Skipped.")
            continue
        event_date = pd.Timestamp(event_date_map[eid])
        try:
            df = load_kma_event(KMA_DIR, eid)
        except Exception as e:
            print(f"[WARN] Failed to load event {eid}: {e}")
            continue
        comp = compute_components(df, event_date)
        if comp.empty:
            print(f"[WARN] Event {eid}: empty pre-event component table. Skipped.")
            continue
        recs = evaluate_one_event(comp, event_date, weights_df)
        for r in recs:
            r["event_id"] = eid
            r["event_date"] = event_date
            r["P_window"] = P_WINDOW
            r["V_window"] = V_WINDOW
            r["W_window"] = W_WINDOW
            r["component_lookback_days"] = COMPONENT_LOOKBACK_DAYS
            r["metric_lookback_days"] = METRIC_LOOKBACK_DAYS
        all_records.extend(recs)
        print(f"[OK] Event {eid}: evaluated {len(recs)} weight combinations")

    if not all_records:
        raise RuntimeError("No records were generated. Check input paths and file formats.")

    metrics_event = pd.DataFrame(all_records)
    metrics_event.to_csv(OUT_DIR / "metrics_by_event_and_weight.csv", index=False, encoding="utf-8-sig")

    summary = summarize_weights(metrics_event)
    summary.to_csv(OUT_DIR / "weight_optimization_summary.csv", index=False, encoding="utf-8-sig")
    save_top_table(summary, OUT_DIR / "top20_weight_combinations.csv", n=20)

    pareto_df = plot_pareto_front(summary, OUT_DIR / "pareto_front_lead_auc_duration.png")
    pareto_df.to_csv(OUT_DIR / "pareto_flagged_weight_summary.csv", index=False, encoding="utf-8-sig")

    plot_weight_sensitivity(summary, OUT_DIR)
    plot_tornado_like_sensitivity(summary, OUT_DIR / "sensitivity_importance_tornado.png")
    plot_response_surface_contour(summary, OUT_DIR / "response_surface_contour_wP_wV.png")
    plot_ternary_response_map(summary, OUT_DIR / "ternary_response_surface_composite.png")
    plot_metric_response_maps(summary, OUT_DIR)

    best = summary.iloc[0]
    print("\n[DONE] Revised optimization completed.")
    print("Best weight combination:")
    print(
        f"  wP={best['wP']:.2f}, wV={best['wV']:.2f}, wW={best['wW']:.2f}, "
        f"score={best['composite_score']:.3f}, "
        f"lead={best['mean_lead']:.2f}, auc={best['mean_auc']:.2f}, duration={best['mean_duration']:.2f}"
    )
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
