"""Turn results/*.csv from bench.exe into results/plots/*.png and results/summary.md.

Inputs (any may be missing; the corresponding outputs are skipped):
  results/ladder.csv   --mode ladder
  results/tune.csv     --mode tune
  results/configs.csv  --mode info
  results/device.csv   --mode info
"""
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PLOTS = RES / "plots"
K5_CONFIG = "128x128x32_w2x4_s3"

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3})
C_CUBLAS, C_ORACLE, C_HEUR, C_FIXED = "#444444", "#1f77b4", "#ff7f0e", "#2ca02c"


def load(name):
    p = RES / name
    if not p.exists():
        print(f"(skipping: {p} not found)")
        return None
    return pd.read_csv(p)


def geomean(x):
    x = np.asarray([v for v in x if v > 0 and np.isfinite(v)])
    return float(np.exp(np.log(x).mean())) if len(x) else float("nan")


def shape_label(r):
    return f"{r.M}x{r.N}x{r.K}"


def save(fig, name):
    fig.savefig(PLOTS / name)
    plt.close(fig)
    print(f"wrote {PLOTS / name}")


md = []  # summary.md lines


def table(df, floatfmt="{:.2f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
def device_section(dev):
    d = dev.iloc[0]
    md.append("## Device\n")
    md.append(
        f"{d['name']} (sm_{d['cc']}), {d.sms} SMs, L2 {d.l2_bytes // 1024} KB, "
        f"measured copy bandwidth {d.copy_bw_gbs:.0f} GB/s, cuBLAS 4096^3: "
        f"FP16->FP32 {d.cublas_fp16_tflops:.2f} TFLOPS, FP32 {d.cublas_fp32_tflops:.2f} TFLOPS.\n"
    )


def ladder_plots(lad):
    s = lad[(lad.M == 4096) & (lad.N == 4096) & (lad.K == 4096)]
    if s.empty:
        return
    order = ["k0", "k1", "k2", "k3", "k4", "k5", "k6_heur"]
    fig, ax = plt.subplots(figsize=(8, 4))
    rows = [s[(s.kernel == k)].iloc[0] for k in order if (s.kernel == k).any()]
    xs = np.arange(len(rows))
    colors = ["#9ecae1" if r.family == "fp32" else "#3182bd" for r in rows]
    ax.bar(xs, [r.tflops for r in rows], color=colors)
    for x, r in zip(xs, rows):
        ax.text(x, r.tflops, f"{r.tflops:.1f}\n{r.pct_cublas:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs, [f"{r.kernel}\n{r.family}" for r in rows])
    for fam, ls in (("fp32", ":"), ("fp16", "--")):
        c = s[(s.kernel == "cublas") & (s.family == fam)]
        if not c.empty:
            ax.axhline(c.iloc[0].tflops, color=C_CUBLAS, ls=ls, lw=1, label=f"cuBLAS {fam}")
    ax.set_ylabel("TFLOPS")
    ax.set_title("The kernel ladder at 4096³ (labels: TFLOPS, % of same-dtype cuBLAS)")
    ax.legend(loc="upper left")
    save(fig, "ladder_4096.png")

    md.append("## Kernel ladder, 4096³\n")
    t = s[s.kernel != "cublas"][["kernel", "family", "config", "ms", "tflops", "pct_cublas"]]
    md.append(table(t) + "\n")

    sq = lad[lad.suite == "square"]
    if not sq.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        for k in ["k2", "k3", "k4", "k5", "k6_heur"]:
            d = sq[sq.kernel == k].sort_values("M")
            if not d.empty:
                ax.plot(d.M, d.pct_cublas, marker="o", label=f"{k} ({d.family.iloc[0]})")
        ax.axhline(100, color=C_CUBLAS, lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("M = N = K")
        ax.set_ylabel("% of cuBLAS (same dtype)")
        ax.set_title("Square shapes")
        ax.legend()
        save(fig, "square_pct.png")


def sawtooth_plot(tune, cfgs, dev):
    s = tune[tune.suite == "sawtooth"]
    if s.empty:
        return
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for kernel, color, label in (("cublas", C_CUBLAS, "cuBLAS"), ("oracle", C_ORACLE, "ours, autotuned"),
                                 ("heuristic", C_HEUR, "ours, heuristic")):
        d = s[s.kernel == kernel].sort_values("N")
        ax.plot(d.N, d.tflops, color=color, label=label, lw=1.5)
    fixed = s[(s.kernel == "tc") & (s.config == K5_CONFIG) & (s.splitk == 1)].sort_values("N")
    ax.plot(fixed.N, fixed.tflops, color=C_FIXED, label=f"ours, fixed {K5_CONFIG}", lw=1)
    ax.set_ylabel("TFLOPS")
    ax.set_title("M = K = 2048, sweeping N: tile and wave quantization")
    ax.legend(fontsize=8)

    # Model: fraction of launched tensor-core work that is useful, for the fixed 128x128 config.
    if cfgs is not None and dev is not None and (cfgs.config == K5_CONFIG).any():
        bps = int(cfgs[cfgs.config == K5_CONFIG].blocks_per_sm.iloc[0])
        slots = int(dev.sms.iloc[0]) * bps
        N = np.arange(1024, 1537, 8)
        tiles = math.ceil(2048 / 128) * np.ceil(N / 128)
        tile_eff = (2048 * N) / (tiles * 128 * 128)
        wave_eff = tiles / (np.ceil(tiles / slots) * slots)
        ax2.plot(N, tile_eff, label="tile efficiency (useful / computed)", color="#9467bd")
        ax2.plot(N, wave_eff, label=f"wave efficiency ({slots} slots)", color="#8c564b")
        ax2.plot(N, tile_eff * wave_eff, label="product", color="black", lw=1.5)
        ax2.set_ylim(0, 1.05)
        ax2.legend(fontsize=8)
    ax2.set_xlabel("N")
    ax2.set_ylabel(f"model, {K5_CONFIG.split('_')[0]}")
    save(fig, "sawtooth.png")

    o = s[s.kernel == "oracle"]
    f = fixed
    md.append("## Sawtooth (M=K=2048, N=1024..1536)\n")
    md.append(
        f"Autotuned: {o.pct_cublas.min():.0f}%–{o.pct_cublas.max():.0f}% of cuBLAS "
        f"(geomean {geomean(o.pct_cublas):.0f}%). Fixed {K5_CONFIG}: {f.pct_cublas.min():.0f}%–"
        f"{f.pct_cublas.max():.0f}% (geomean {geomean(f.pct_cublas):.0f}%).\n"
    )


def llm_plot(tune, dev):
    s = tune[tune.suite == "llm"]
    if s.empty:
        return
    groups = s.groupby(["N", "K"])
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), squeeze=False)
    bw = float(dev.copy_bw_gbs.iloc[0]) if dev is not None else None
    peak = float(dev.cublas_fp16_tflops.iloc[0]) if dev is not None else None
    for ax, ((N, K), d) in zip(axes[0], groups):
        for kernel, color, label in (("cublas", C_CUBLAS, "cuBLAS"), ("oracle", C_ORACLE, "ours, autotuned"),
                                     ("heuristic", C_HEUR, "ours, heuristic")):
            dd = d[d.kernel == kernel].sort_values("M")
            ax.plot(dd.M, dd.tflops, marker="o", color=color, label=label)
        if bw:
            M = np.unique(d.M)
            bytes_ = 2.0 * (M * K + K * N) + 4.0 * M * N
            roof = np.minimum(peak or np.inf, bw * 1e9 * (2.0 * M * N * K / bytes_) / 1e12)
            ax.plot(M, roof, color="red", ls="--", lw=1, label="roofline (copy BW, cuBLAS peak)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"N={N}, K={K}")
        ax.set_xlabel("M (tokens)")
    axes[0][0].set_ylabel("TFLOPS")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("LLM-style projections: skinny M is bandwidth-bound")
    save(fig, "llm.png")


def splitk_plot(tune):
    s = tune[(tune.suite == "skinny") & (tune.kernel == "tc")]
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = plt.get_cmap("viridis")
    shapes = s[["M", "N", "K"]].drop_duplicates().sort_values(["M", "K"]).values
    for i, (M, N, K) in enumerate(shapes):
        d = s[(s.M == M) & (s.N == N) & (s.K == K)]
        color = cmap(i / max(1, len(shapes) - 1))
        for mode, ls in (("workspace", "-"), ("atomic", ":")):
            dd = d[(d.splitmode == mode) | (d.splitk == 1)]
            best = dd.groupby("splitk").tflops.max()
            ax.plot(best.index, best.values, ls=ls, marker="o", ms=3, color=color,
                    label=f"{M}x{N}x{K}" if mode == "workspace" else None)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("split-K factor (best config at each split)")
    ax.set_ylabel("TFLOPS")
    ax.set_title("Split-K on small-output, long-K shapes (solid: workspace+reduce, dotted: atomics)")
    ax.legend(fontsize=7, ncol=2)
    save(fig, "splitk.png")


def heatmap_and_gap(tune):
    tc = tune[tune.kernel == "tc"].copy()
    orc = tune[tune.kernel == "oracle"].set_index(["suite", "M", "N", "K"])
    if tc.empty or orc.empty:
        return
    # Best over split-K for each (shape, config), relative to the oracle for that shape.
    best = tc.groupby(["suite", "M", "N", "K", "config"]).tflops.max().reset_index()
    best["rel"] = best.apply(lambda r: r.tflops / orc.loc[(r.suite, r.M, r.N, r.K)].tflops, axis=1)
    hm = best[best.suite != "sawtooth"].copy()
    hm["shape"] = hm.suite + " " + hm.apply(shape_label, axis=1)
    piv = hm.pivot_table(index="shape", columns="config", values="rel")
    order_rows = hm.drop_duplicates("shape")[["suite", "M", "N", "K", "shape"]].sort_values(["suite", "M", "N", "K"])
    piv = piv.reindex(order_rows["shape"])
    fig, ax = plt.subplots(figsize=(0.45 * piv.shape[1] + 3, 0.28 * piv.shape[0] + 2))
    im = ax.imshow(piv.values, aspect="auto", cmap="magma", vmin=0.3, vmax=1.0)
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=70, ha="right", fontsize=7)
    ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=7)
    for i in range(piv.shape[0]):
        j = np.nanargmax(piv.values[i])
        ax.text(j, i, "*", ha="center", va="center", color="cyan", fontsize=9)
    fig.colorbar(im, ax=ax, label="TFLOPS / best-for-this-shape")
    ax.set_title("No config wins everywhere (* = winner per shape)")
    ax.grid(False)
    save(fig, "config_heatmap.png")

    # Oracle vs heuristic vs the single best fixed config (split 1, chosen by geomean over all shapes).
    s1 = tc[(tc.splitk == 1)].copy()
    s1["pct"] = s1.pct_cublas
    fixed_scores = s1.groupby("config").pct.apply(geomean)
    n_shapes = len(tune[tune.kernel == "oracle"])
    complete = s1.groupby("config").size() == n_shapes
    fixed_scores = fixed_scores[complete[complete].index]
    fixed_best = fixed_scores.idxmax()
    rows = []
    for suite, d in tune.groupby("suite"):
        rows.append({
            "suite": suite,
            "shapes": int((d.kernel == "oracle").sum()),
            f"fixed ({fixed_best})": geomean(d[(d.kernel == "tc") & (d.config == fixed_best) & (d.splitk == 1)].pct_cublas),
            "heuristic": geomean(d[d.kernel == "heuristic"].pct_cublas),
            "oracle": geomean(d[d.kernel == "oracle"].pct_cublas),
        })
    gap = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(len(gap))
    w = 0.27
    for i, (col, color) in enumerate(zip(gap.columns[2:], (C_FIXED, C_HEUR, C_ORACLE))):
        ax.bar(xs + (i - 1) * w, gap[col], w, label=col, color=color)
    ax.axhline(100, color=C_CUBLAS, lw=1)
    ax.set_xticks(xs, gap.suite)
    ax.set_ylabel("geomean % of cuBLAS")
    ax.set_title("Same kernels, different selection: fixed vs heuristic vs exhaustive search")
    ax.legend(fontsize=8)
    save(fig, "autotune_gap.png")
    md.append("## Selection matters: geomean % of cuBLAS per suite\n")
    md.append(table(gap, "{:.1f}") + "\n")

    wins = tune[tune.kernel == "oracle"].groupby("config").size().sort_values(ascending=False)
    md.append("## How often each config is the oracle's pick\n")
    md.append(table(wins.rename("wins").reset_index()) + "\n")


def awkward_table(tune):
    s = tune[tune.suite == "awkward"]
    if s.empty:
        return
    rows = []
    for (M, N, K), d in s.groupby(["M", "N", "K"], sort=False):
        o = d[d.kernel == "oracle"].iloc[0]
        h = d[d.kernel == "heuristic"].iloc[0]
        rows.append({"shape": f"{M}x{N}x{K}", "vec path": "yes" if K % 8 == 0 and N % 8 == 0 else "no",
                     "cuBLAS TF": o.cublas_tflops, "oracle TF": o.tflops, "oracle %": o.pct_cublas,
                     "heuristic %": h.pct_cublas, "oracle config": f"{o.config} k={o.splitk}"})
    md.append("## Awkward shapes\n")
    md.append(table(pd.DataFrame(rows), "{:.1f}") + "\n")


def occupancy_plot(tune, cfgs):
    s = tune[(tune.suite == "square") & (tune.M == 4096) & (tune.kernel == "tc") & (tune.splitk == 1)]
    if s.empty or cfgs is None:
        return
    m = cfgs.merge(s[["config", "tflops", "pct_cublas"]], on="config")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for _, r in m.iterrows():
        color = "red" if r.local_bytes > 0 else ("#9467bd" if r.minb > 1 else C_ORACLE)
        ax.scatter(r.occupancy_pct, r.tflops, color=color)
        ax.annotate(f"{r.config}\n{r.regs} regs" + (f", {r.local_bytes}B local" if r.local_bytes else ""),
                    (r.occupancy_pct, r.tflops), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("theoretical occupancy (% of max warps/SM)")
    ax.set_ylabel("TFLOPS at 4096³")
    ax.set_title("Occupancy is not the goal (red = spills to local memory)")
    save(fig, "occupancy.png")
    md.append("## Configs: registers, occupancy, 4096³ performance\n")
    cols = ["config", "threads", "regs", "local_bytes", "smem_bytes", "blocks_per_sm", "occupancy_pct", "tflops",
            "pct_cublas"]
    md.append(table(m[cols].sort_values("tflops", ascending=False), "{:.1f}") + "\n")


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    lad, tune, cfgs, dev = load("ladder.csv"), load("tune.csv"), load("configs.csv"), load("device.csv")
    for df in (lad, tune):
        if df is not None and (df.ok == "FAIL").any():
            print("WARNING: some rows FAILED correctness; they are excluded from the plots")
    if lad is not None:
        lad = lad[lad.ok != "FAIL"]
    if tune is not None:
        tune = tune[tune.ok != "FAIL"]

    md.append("# Results summary (generated by scripts/plot.py)\n")
    if dev is not None:
        device_section(dev)
    if lad is not None:
        ladder_plots(lad)
    if tune is not None:
        sawtooth_plot(tune, cfgs, dev)
        llm_plot(tune, dev)
        splitk_plot(tune)
        heatmap_and_gap(tune)
        awkward_table(tune)
        occupancy_plot(tune, cfgs)
    (RES / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {RES / 'summary.md'}")


if __name__ == "__main__":
    main()
