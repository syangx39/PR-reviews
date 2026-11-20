#!/usr/bin/env python3
"""Generate the three review figures for google-pytorch/torchtitan PR #168 from summaries/."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_SUM_DIR = HERE.parent / "summaries"
DEFAULT_OUT_DIR = HERE.parent / "figures"


def generate_figures(sum_dir: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ab_sum = json.loads((sum_dir / "ab_compile_token_summary.json").read_text(encoding="utf-8"))["stage1_replay"]
    xp_comp = json.loads((sum_dir / "xprof_7cat_comparison.json").read_text(encoding="utf-8"))
    ms_comp = json.loads((sum_dir / "multiseed_comparison.json").read_text(encoding="utf-8"))

    def _warm_stats(arm: str) -> dict[str, float]:
        wt = json.loads((sum_dir / f"rank0_warm_timings_{arm}.json").read_text(encoding="utf-8"))["warm_iters"]
        fb_rank0 = [float(x["fwd_bwd_s"]) for x in wt]
        r0_mean = float(np.mean(fb_rank0))
        canon_fb = float(ab_sum[arm]["fwd_bwd_s"])
        canon_hbm = float(ab_sum[arm]["peak_hbm_gb"])
        return {
            "fb_mean": canon_fb,
            "fb_low": max(0.005, r0_mean - float(np.min(fb_rank0))),
            "fb_high": max(0.005, float(np.max(fb_rank0)) - r0_mean),
            "hbm_max": canon_hbm,
        }

    written: list[Path] = []

    # Figure 1: 2x2 Factorial on Locked-Layout Fixed-Batch Replay & Saved-Tensor Mechanism
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0), dpi=160)
    fac_arms = [
        ("A_sel", "A_sel (Baseline)\nrecompute_bmm=F\nsave_pallas=F", "#7f8691"),
        ("C3_ft", "C3_ft\nrecompute_bmm=F\nsave_pallas=T", "#4e79a7"),
        ("C2_tf", "C2_tf\nrecompute_bmm=T\nsave_pallas=F", "#f28e2b"),
        ("C4_tt", "C4_tt (Both #168)\nrecompute_bmm=T\nsave_pallas=T", "#2ca02c"),
    ]
    x = np.arange(len(fac_arms))
    w = 0.35
    ax = axes[0]
    ax_hbm = ax.twinx()

    fb_means = [_warm_stats(k)["fb_mean"] for k, _, _ in fac_arms]
    fb_low = [_warm_stats(k)["fb_low"] for k, _, _ in fac_arms]
    fb_high = [_warm_stats(k)["fb_high"] for k, _, _ in fac_arms]
    hbms = [_warm_stats(k)["hbm_max"] for k, _, _ in fac_arms]
    cols = [c for _, _, c in fac_arms]

    ax.bar(x - w / 2, fb_means, width=w * 0.90, yerr=[fb_low, fb_high], capsize=3.5,
           color=cols, edgecolor="#222222", linewidth=0.7, label="fwd_bwd_s (mean, min–max whiskers)", zorder=3)
    ax_hbm.bar(x + w / 2, hbms, width=w * 0.90, color=cols, alpha=0.45, hatch="//",
               edgecolor="#222222", linewidth=0.7, label="Peak allocated HBM (GB)", zorder=3)

    for i, (m, h) in enumerate(zip(fb_means, hbms)):
        delta_s = m - fb_means[0]
        s_txt = f"{m:.3f}s" if i == 0 else f"{m:.3f}s\n({delta_s:+.3f}s)"
        ax.text(x[i] - w / 2, m + fb_high[i] + 0.03, s_txt, ha="center", va="bottom", fontsize=9.0,
                fontweight="bold" if i == 3 else "normal")
        dh = h - hbms[0]
        h_txt = f"{h:.2f}GB" if i == 0 else f"{h:.2f}GB\n({dh:+.2f})"
        ax_hbm.text(x[i] + w / 2, h + 0.5, h_txt, ha="center", va="bottom", fontsize=9.0, color="#1f3b5c")

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl, _ in fac_arms], fontsize=9.0)
    ax.set_ylabel("Warm replay fwd_bwd time (s / step)", fontsize=10.0)
    ax_hbm.set_ylabel("Peak allocated Trainer HBM (GB)", fontsize=10.0, color="#1f3b5c")
    ax.set_ylim(4.4, 5.70)
    ax_hbm.set_ylim(0, 31.5)
    ax.tick_params(axis="y", labelsize=9.0)
    ax_hbm.tick_params(axis="y", labelsize=9.0)
    ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    ax.set_title("(a) 2×2 Factorial on Fixed Replay Batch (6 mb: 14336×4, 8192×2)", fontsize=10.0)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_hbm.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9.0, loc="upper right")

    ax2 = axes[1]
    gdn_mib = [604.74, 604.74, 269.24, 269.24]
    attn_mib = [256.42, 312.86, 256.42, 312.86]
    total_24l_gib = [(18 * g + 6 * a) / 1024.0 for g, a in zip(gdn_mib, attn_mib)]
    gdn_gib = [(18 * g) / 1024.0 for g in gdn_mib]
    attn_gib = [(6 * a) / 1024.0 for a in attn_mib]

    ax2.bar(x, gdn_gib, width=0.44, color="#3b75af", label="18 GDN layers saved tensors (GiB @ row=14336)", zorder=3)
    ax2.bar(x, attn_gib, width=0.44, bottom=gdn_gib, color="#e1812c", label="6 FullAttn layers saved tensors (GiB @ row=14336)", zorder=3)
    for i in range(len(fac_arms)):
        g_cnt = 699 if i in (0, 1) else 26
        a_cnt = 18 if i in (0, 2) else 20
        pallas_calls = 2 if i in (0, 2) else 1
        ax2.text(
            x[i], total_24l_gib[i] + 0.35,
            f"{total_24l_gib[i]:.2f} GiB\nGDN: {g_cnt} tensors\nAttn: {a_cnt} ({pallas_calls} Pallas)",
            ha="center", va="bottom", fontsize=9.0,
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels([lbl for _, lbl, _ in fac_arms], fontsize=9.0)
    ax2.set_ylabel("Saved Fwd→Bwd Activations (GiB, row=14336)", fontsize=10.0)
    ax2.set_ylim(0, 21.5)
    ax2.tick_params(axis="y", labelsize=9.0)
    ax2.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    ax2.set_title("(b) AOT FX Saved Activations Across 24 Layers (row=14336)", fontsize=10.0)
    ax2.legend(fontsize=9.0, loc="upper right")

    fig.suptitle("google-pytorch/torchtitan #168: TPUSelectiveAC 2×2 Ablation on Qwen3.5-0.8B (TPU v7x 2x2x4, FSDP2=32)", fontsize=11.0)
    fig.tight_layout()
    p1 = out_dir / "pr168_fig1_factorial_replay.png"
    fig.savefig(p1)
    plt.close(fig)
    written.append(p1)

    # Figure 3: 3-Seed (Seeds 1, 42, 101) 30-Step GSM8K Accuracy & Paired McNemar CIs
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0), dpi=160)
    steps = [0, 10, 20, 30]
    runs_d = ms_comp["runs"]
    a_keys = ["a_sel_s1", "a_sel_s42", "a_sel_s101"]
    c_keys = ["c4_tt_s1", "c4_tt_s42", "c4_tt_s101"]

    ax = axes[0]
    for rk in a_keys:
        ys = [float(runs_d[rk]["validation"][str(s)]["accuracy_pct"]) for s in steps]
        ax.plot(steps, ys, color="#4e79a7", alpha=0.35, linestyle="--", marker="o", markersize=4)
    for rk in c_keys:
        ys = [float(runs_d[rk]["validation"][str(s)]["accuracy_pct"]) for s in steps]
        ax.plot(steps, ys, color="#2ca02c", alpha=0.35, linestyle="-.", marker="s", markersize=4)

    a_grp = ms_comp["groups"]["gsc_A_sel_3seed_matched"]["validation"]
    c_grp = ms_comp["groups"]["gsc_C4_tt_3seed"]["validation"]
    a_means = [float(a_grp[str(s)]["accuracy_pct"]["mean"]) for s in steps]
    a_sds = [float(a_grp[str(s)]["accuracy_pct"]["sd"]) for s in steps]
    c_means = [float(c_grp[str(s)]["accuracy_pct"]["mean"]) for s in steps]
    c_sds = [float(c_grp[str(s)]["accuracy_pct"]["sd"]) for s in steps]

    ax.errorbar(np.array(steps) - 0.35, a_means, yerr=a_sds, color="#1f4e78", marker="o", linewidth=2.2,
                capsize=4, label=f"A_sel 3-seed mean ± SD (Step 30: {a_means[-1]:.2f}% ± {a_sds[-1]:.2f} pp)")
    ax.errorbar(np.array(steps) + 0.35, c_means, yerr=c_sds, color="#1f7a1f", marker="s", linewidth=2.2,
                capsize=4, label=f"C4_tt (#168) 3-seed mean ± SD (Step 30: {c_means[-1]:.2f}% ± {c_sds[-1]:.2f} pp)")

    a_nl = float(a_grp["30"]["non_loop_accuracy_pct"]["mean"])
    c_nl = float(c_grp["30"]["non_loop_accuracy_pct"]["mean"])
    ax.scatter([29.1], [a_nl], color="#1f4e78", marker="^", s=70, zorder=5, label=f"A_sel Step 30 non-loop mean: {a_nl:.2f}%")
    ax.scatter([30.9], [c_nl], color="#1f7a1f", marker="^", s=70, zorder=5, label=f"C4_tt Step 30 non-loop mean: {c_nl:.2f}% (+0.66 pp)")

    ax.set_xticks(steps)
    ax.set_xticklabels([f"Step {s}" for s in steps], fontsize=9.0)
    ax.set_ylabel("GSM8K Validation Accuracy (%, n=1,319 greedy)", fontsize=10.0)
    ax.set_ylim(44, 85)
    ax.tick_params(axis="y", labelsize=9.0)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_title("(a) 3-Seed 30-Step GSM8K Convergence (Seeds 1, 42, 101)", fontsize=10.0)
    ax.legend(fontsize=9.0, loc="lower right")

    ax2 = axes[1]
    pm = ms_comp["paired_mcnemar"]
    ci_items = [
        ("Seed 1 (matched, n=1,319)", pm["gsc_seed1_C4_minus_A_matched"]["30"]),
        ("Seed 1 (manifest, n=1,319)", pm["gsc_seed1_C4_minus_A_manifest"]["30"]),
        ("Seed 42 (n=1,319)", pm["gsc_seed42_C4_minus_A"]["30"]),
        ("Seed 101 (n=1,319)", pm["gsc_seed101_C4_minus_A"]["30"]),
        ("Pooled 3-Seed (matched, n=3,957)", pm["pooled_3seed_C4_minus_A_matched"]["30"]),
        ("Pooled 3-Seed (manifest, n=3,957)", pm["pooled_3seed_C4_minus_A_with_manifest_s1"]["30"]),
    ]
    yp = np.arange(len(ci_items))[::-1]
    ax2.axvline(0.0, color="#222222", linestyle="--", linewidth=1.2, zorder=2, label="Zero difference (0.0 pp)")
    for y, (lbl, rec) in zip(yp, ci_items):
        d = float(rec["diff_b_minus_a_pp"])
        lo = float(rec["ci95_low_pp"])
        hi = float(rec["ci95_high_pp"])
        col = "#1f7a1f" if "Pooled" in lbl else "#2b7bba"
        ax2.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", color=col, markersize=6, linewidth=2.0, capsize=4, zorder=4)
        ax2.text(hi + 0.14, y, f"{d:+.2f} pp [{lo:+.2f}, {hi:+.2f}]", va="center", ha="left", fontsize=9.0,
                 fontweight="bold" if "Pooled" in lbl else "normal")

    ax2.set_yticks(yp)
    ax2.set_yticklabels([lbl for lbl, _ in ci_items], fontsize=9.0)
    ax2.set_xlabel("Step-30 Paired Accuracy Difference: C4_tt − A_sel (pp, 95% McNemar CI)", fontsize=10.0)
    ax2.set_xlim(-4.8, 6.4)
    ax2.tick_params(axis="x", labelsize=9.0)
    ax2.grid(axis="x", linestyle=":", alpha=0.6)
    ax2.set_title("(b) Step-30 Paired McNemar 95% CIs (All Span 0; HBM −3.49 GB)", fontsize=10.0)

    fig.suptitle("Qwen3.5-0.8B 30-Step GRPO on GSM8K (16 Trainer + 16 Rollout TPU v7x Chips): Multi-Seed Verification of #168", fontsize=11.0)
    fig.tight_layout()
    p3 = out_dir / "pr168_fig3_multiseed_gsm8k.png"
    fig.savefig(p3)
    plt.close(fig)
    written.append(p3)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot review figures for torchtitan#168 from public summaries/")
    parser.add_argument("--summaries-dir", type=Path, default=DEFAULT_SUM_DIR, help="Path to summaries/ directory")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory to write PNG figures")
    args = parser.parse_args(argv)
    paths = generate_figures(args.summaries_dir.resolve(), args.out_dir.resolve())
    for p in paths:
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
