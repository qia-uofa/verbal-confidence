"""Build experiments.ipynb from cell definitions."""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

cells = []

def md(src): cells.append(new_markdown_cell(src))
def code(src): cells.append(new_code_cell(src))

# ─────────────────────────────────────────────────────────────────────────────
md("# Verbal Confidence — Experiments Notebook\n"
   "Run and visualise all experiments from *How do LLMs Compute Verbal Confidence?* (Kumaran et al., ICML 2026).\n\n"
   "**Usage:** Run cells top-to-bottom. Set `VISUALIZE_ONLY = True` to skip model loading and only plot cached results.")

# ── 1. Environment setup ─────────────────────────────────────────────────────
md("## 1 · Environment Setup\n"
   "Must run before any other cell — sets `HF_HOME` before HuggingFace is imported.")

code("""\
import os
import sys
from pathlib import Path

# ── Locate project root and load .env ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parents[1] if "__file__" in dir() else Path.cwd().parent
if not (PROJECT_ROOT / ".env").exists():
    # fallback: walk up to find .env
    for p in Path.cwd().parents:
        if (p / ".env").exists():
            PROJECT_ROOT = p
            break

_dotenv = PROJECT_ROOT / ".env"
if _dotenv.exists():
    for _line in _dotenv.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            if _k.strip() not in os.environ:
                os.environ[_k.strip()] = _v.strip()
    print(f"Loaded .env from {_dotenv}")
else:
    print(f"WARNING: .env not found at {_dotenv}")

# Set HF_HOME BEFORE importing transformers
if "EPHEMERAL_ROOT" in os.environ:
    os.environ.setdefault("HF_HOME", os.environ["EPHEMERAL_ROOT"] + "/hf_cache")

sys.path.insert(0, str(PROJECT_ROOT / "src"))

print(f"PROJECT_ROOT = {PROJECT_ROOT}")
print(f"HF_HOME      = {os.environ.get('HF_HOME', '<not set>')}")
print(f"EPHEMERAL_ROOT = {os.environ.get('EPHEMERAL_ROOT', '<not set>')}")
print(f"PERMANENT_ROOT = {os.environ.get('PERMANENT_ROOT', '<not set>')}")
""")

# ── 2. Config ────────────────────────────────────────────────────────────────
md("## 2 · Configuration")

code("""\
from verbal_confidence.config import load_config

cfg = load_config()

# ── User settings ─────────────────────────────────────────────────────────
# Set to the run_name of an existing run to load cached results,
# or leave as None to use cfg default ("default").
RUN_NAME = None          # e.g. "1715813"
VISUALIZE_ONLY = False   # True = skip model loading, only plot cached results
# ──────────────────────────────────────────────────────────────────────────

if RUN_NAME:
    cfg.paths.run_name = RUN_NAME

print(f"active_model   = {cfg.active_model}")
print(f"active_dataset = {cfg.active_dataset}")
print(f"n_questions    = {cfg.n_questions}")
print(f"seed           = {cfg.seed}")
print(f"run_name       = {cfg.paths.run_name}")
print(f"results_root   = {cfg.paths.results_root}")
print(f"hf_home        = {cfg.paths.hf_home}")
""")

# ── 3. Helpers ───────────────────────────────────────────────────────────────
md("## 3 · Helper imports")

code("""\
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

from verbal_confidence.utils.io import load_results
from verbal_confidence.config import results_dir

RESULTS = results_dir(cfg)
CONFIDENCE_CLASSES = cfg.confidence_classes
N_CLASSES = len(CONFIDENCE_CLASSES)
CLASS_LABELS = [c.replace(" ", "\\n") for c in CONFIDENCE_CLASSES]

def load(filename):
    results, meta = load_results(RESULTS / filename)
    if results is None:
        print(f"  Not found: {RESULTS / filename}")
    else:
        print(f"  Loaded {len(results)} records from {filename}")
    return results, meta

print(f"Results directory: {RESULTS}")
""")

# ── 4. Model loading ─────────────────────────────────────────────────────────
md("## 4 · Model Loading\n"
   "Skip this cell if `VISUALIZE_ONLY = True`.")

code("""\
if not VISUALIZE_ONLY:
    from verbal_confidence.models.loader import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)
    print("Model loaded.")
else:
    model, tokenizer = None, None
    print("VISUALIZE_ONLY mode — skipping model load.")
""")

# ── 5. Phase 0 ───────────────────────────────────────────────────────────────
md("## 5 · Phase 0 — Answer Generation")

code("""\
from verbal_confidence.experiments.phase0 import run_phase0

if not VISUALIZE_ONLY:
    p0 = run_phase0(cfg, model, tokenizer)
else:
    p0, _ = load(cfg.phase0.output_file)

if p0:
    print(f"\\n{len(p0)} questions answered.")
    print("\\nSample:")
    for r in p0[:3]:
        print(f"  Q: {r['question']}")
        print(f"  A: {r['model_answer'][:80].strip()}...")
        print()
""")

# ── 6. Phase 1 ───────────────────────────────────────────────────────────────
md("## 6 · Phase 1 — Confidence Elicitation")

code("""\
from verbal_confidence.experiments.phase1 import run_phase1

if not VISUALIZE_ONLY:
    p1 = run_phase1(cfg, model, tokenizer, p0)
else:
    p1, _ = load(cfg.phase1.output_file)

if p1:
    confidences  = [r["probs_cls"][r["pred_class"]] for r in p1]
    pred_classes = [r["pred_class"] for r in p1]
    is_correct   = [r["is_correct"] for r in p1]

    print(f"Mean confidence : {np.mean(confidences):.3f}")
    print(f"Mean accuracy   : {np.mean(is_correct):.3f}")
    print(f"Overconfident   : {np.mean(confidences) > np.mean(is_correct)}")
""")

code("""\
if p1:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # --- Class distribution ---
    ax = axes[0]
    counts = np.bincount(pred_classes, minlength=N_CLASSES)
    bars = ax.bar(range(N_CLASSES), counts, color="steelblue", edgecolor="white")
    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(CONFIDENCE_CLASSES, rotation=45, ha="right", fontsize=7)
    ax.set_title("Predicted Confidence Class Distribution")
    ax.set_ylabel("Count")

    # --- Calibration curve ---
    ax = axes[1]
    bin_edges = np.linspace(0, 1, 11)
    bin_mids  = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_acc, bin_conf, bin_n = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = [(lo <= c < hi) for c in confidences]
        if sum(mask) == 0:
            continue
        bin_acc.append(np.mean([is_correct[i] for i, m in enumerate(mask) if m]))
        bin_conf.append(np.mean([confidences[i] for i, m in enumerate(mask) if m]))
        bin_n.append(sum(mask))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.bar(bin_conf, bin_acc, width=0.08, alpha=0.5, color="steelblue", label="Accuracy")
    ax.plot(bin_conf, bin_acc, "o-", color="steelblue")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Calibration Curve")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # --- Confidence by correctness ---
    ax = axes[2]
    conf_correct   = [confidences[i] for i, c in enumerate(is_correct) if c]
    conf_incorrect = [confidences[i] for i, c in enumerate(is_correct) if not c]
    ax.hist(conf_correct,   bins=20, alpha=0.6, label=f"Correct (n={len(conf_correct)})",   color="green")
    ax.hist(conf_incorrect, bins=20, alpha=0.6, label=f"Incorrect (n={len(conf_incorrect)})", color="red")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence by Correctness")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(RESULTS / "phase1_summary.png", dpi=150, bbox_inches="tight")
    plt.show()
""")

# ── 7. Steering ──────────────────────────────────────────────────────────────
md("## 7 · Experiment 1 — Activation Steering")

code("""\
from verbal_confidence.experiments.steering import run_steering

if not VISUALIZE_ONLY:
    steer = run_steering(cfg, model, tokenizer, p1)
else:
    steer, _ = load(cfg.steering.output_file)

if steer:
    import pandas as pd
    df_s = pd.DataFrame(steer)
    print(df_s.head())
    print(f"\\nMean delta_class: {df_s['delta_class'].mean():.3f}")
""")

code("""\
if steer:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # --- Heatmap: mean delta_class by layer × alpha ---
    ax = axes[0]
    layers = sorted(df_s["layer"].unique())
    alphas = sorted(df_s["alpha"].unique())
    heat   = np.array([[df_s[(df_s.layer==l) & (df_s.alpha==a)]["delta_class"].mean()
                        for a in alphas] for l in layers])
    im = ax.imshow(heat, aspect="auto", cmap="RdBu_r",
                   vmin=-3, vmax=3, origin="lower")
    ax.set_xticks(range(len(alphas))); ax.set_xticklabels(alphas)
    ax.set_yticks(range(len(layers))); ax.set_yticklabels(layers)
    ax.set_xlabel("Alpha"); ax.set_ylabel("Layer")
    ax.set_title("Mean ΔClass (Steering: high→)")
    plt.colorbar(im, ax=ax, label="ΔClass")

    # --- Distribution of delta_class ---
    ax = axes[1]
    ax.hist(df_s["delta_class"], bins=range(-9, 11), edgecolor="white", color="steelblue")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("ΔClass"); ax.set_ylabel("Count")
    ax.set_title("Steering: Distribution of Confidence Change")

    plt.tight_layout()
    plt.savefig(RESULTS / "steering_summary.png", dpi=150, bbox_inches="tight")
    plt.show()
""")

# ── 8. Patching ──────────────────────────────────────────────────────────────
md("## 8 · Experiment 2 — Activation Patching")

code("""\
from verbal_confidence.experiments.patching import run_patching

if not VISUALIZE_ONLY:
    patch = run_patching(cfg, model, tokenizer, p1)
else:
    patch, _ = load(cfg.patching.output_file)

if patch:
    import pandas as pd
    df_p = pd.DataFrame(patch)
    print(df_p.groupby(["layer","position"])["delta_class"].mean().unstack())
""")

code("""\
if patch:
    positions = sorted(df_p["position"].unique())
    layers    = sorted(df_p["layer"].unique())

    fig, axes = plt.subplots(1, len(positions), figsize=(5*len(positions), 4), sharey=True)
    if len(positions) == 1:
        axes = [axes]

    for ax, pos in zip(axes, positions):
        means = [df_p[(df_p.layer==l) & (df_p.position==pos)]["delta_class"].mean()
                 for l in layers]
        stds  = [df_p[(df_p.layer==l) & (df_p.position==pos)]["delta_class"].std()
                 for l in layers]
        ax.bar(range(len(layers)), means, yerr=stds, capsize=4,
               color="steelblue", edgecolor="white")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f"L{l}" for l in layers])
        ax.set_title(f"Position: {pos.upper()}")
        ax.set_xlabel("Layer")
        if ax == axes[0]:
            ax.set_ylabel("Mean ΔClass")

    fig.suptitle("Activation Patching: ΔClass (high→low)", y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS / "patching_summary.png", dpi=150, bbox_inches="tight")
    plt.show()
""")

# ── 9. Noising ───────────────────────────────────────────────────────────────
md("## 9 · Experiment 3 — Activation Noising (Mean Ablation)")

code("""\
from verbal_confidence.experiments.noising import run_noising

if not VISUALIZE_ONLY:
    noise = run_noising(cfg, model, tokenizer, p1)
else:
    noise, _ = load(cfg.noising.output_file)

if noise:
    import pandas as pd
    df_n = pd.DataFrame(noise)
    print(df_n.groupby(["layer","position"])["delta_class"].mean().unstack())
""")

code("""\
if noise:
    positions = sorted(df_n["position"].unique())
    layers    = sorted(df_n["layer"].unique())

    fig, axes = plt.subplots(1, len(positions), figsize=(5*len(positions), 4), sharey=True)
    if len(positions) == 1:
        axes = [axes]

    for ax, pos in zip(axes, positions):
        vals = [df_n[(df_n.layer==l) & (df_n.position==pos)]["delta_class"].mean()
                for l in layers]
        ax.bar(range(len(layers)), vals, color="coral", edgecolor="white")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f"L{l}" for l in layers])
        ax.set_title(f"Position: {pos.upper()}")
        ax.set_xlabel("Layer")
        if ax == axes[0]:
            ax.set_ylabel("Mean ΔClass")

    fig.suptitle("Mean Ablation: ΔClass", y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS / "noising_summary.png", dpi=150, bbox_inches="tight")
    plt.show()
""")

# ── 10. Swap ──────────────────────────────────────────────────────────────────
md("## 10 · Experiment 4 — Activation Swap")

code("""\
from verbal_confidence.experiments.swap import run_swap

if not VISUALIZE_ONLY:
    swap = run_swap(cfg, model, tokenizer, p1)
else:
    swap, _ = load(cfg.swap.output_file)

if swap:
    import pandas as pd
    df_sw = pd.DataFrame(swap)
    print(df_sw.head())
""")

code("""\
if swap:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Delta class distribution
    ax = axes[0]
    ax.hist(df_sw["delta_class_A"], bins=range(-9,11), alpha=0.6,
            label="A (was high)", color="steelblue", edgecolor="white")
    ax.hist(df_sw["delta_class_B"], bins=range(-9,11), alpha=0.6,
            label="B (was low)",  color="coral",     edgecolor="white")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("ΔClass"); ax.set_ylabel("Count")
    ax.set_title("Swap: Confidence Change After Swapping")
    ax.legend()

    # By layer
    ax = axes[1]
    layers = sorted(df_sw["layer"].unique())
    means_a = [df_sw[df_sw.layer==l]["delta_class_A"].mean() for l in layers]
    means_b = [df_sw[df_sw.layer==l]["delta_class_B"].mean() for l in layers]
    x = np.arange(len(layers)); w = 0.35
    ax.bar(x - w/2, means_a, w, label="A (was high)", color="steelblue")
    ax.bar(x + w/2, means_b, w, label="B (was low)",  color="coral")
    ax.set_xticks(x); ax.set_xticklabels([f"L{l}" for l in layers])
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean ΔClass")
    ax.set_title("Swap: By Layer")
    ax.legend()

    plt.tight_layout()
    plt.savefig(RESULTS / "swap_summary.png", dpi=150, bbox_inches="tight")
    plt.show()
""")

# ── 11. Probing ───────────────────────────────────────────────────────────────
md("## 11 · Experiment 5 — Linear Probing")

code("""\
from verbal_confidence.experiments.probing import run_probing

if not VISUALIZE_ONLY:
    probe = run_probing(cfg, model, tokenizer, p1)
else:
    probe, probe_meta = load(cfg.probing.output_file)

if probe:
    import pandas as pd
    df_pr = pd.DataFrame(probe)
    metric_col = "r2" if "r2" in df_pr.columns else "auroc"
    print(f"Metric: {metric_col}")
    print(df_pr.groupby("position")[metric_col].max())
""")

code("""\
if probe:
    positions = sorted(df_pr["position"].unique())
    colors    = plt.cm.tab10(np.linspace(0, 0.5, len(positions)))

    fig, ax = plt.subplots(figsize=(12, 5))
    for pos, color in zip(positions, colors):
        sub = df_pr[df_pr.position == pos].sort_values("layer")
        ax.plot(sub["layer"], sub[metric_col], "o-", label=pos.upper(),
                color=color, lw=2, ms=5)

    ax.set_xlabel("Layer")
    ax.set_ylabel(metric_col.upper())
    ax.set_title(f"Linear Probing: {metric_col.upper()} by Layer and Position")
    ax.legend(title="Position")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS / "probing_summary.png", dpi=150, bbox_inches="tight")
    plt.show()
""")

# ── 12. Variance Partitioning ─────────────────────────────────────────────────
md("## 12 · Experiment 6 — Variance Partitioning")

code("""\
from verbal_confidence.experiments.variance_partitioning import run_variance_partitioning

if not VISUALIZE_ONLY:
    vp = run_variance_partitioning(cfg, model, tokenizer, p1)
else:
    vp, _ = load(cfg.variance_partitioning.output_file)

if vp:
    import pandas as pd
    df_vp = pd.DataFrame(vp)
    print(df_vp.head())
""")

code("""\
if vp:
    fig, ax = plt.subplots(figsize=(10, 4))

    positions  = sorted(df_vp["position"].unique()) if "position" in df_vp else ["overall"]
    baselines  = [c for c in df_vp.columns if c not in ("position","layer")]
    x          = np.arange(len(positions))
    width      = 0.8 / max(len(baselines), 1)
    colors     = plt.cm.Set2(np.linspace(0, 1, len(baselines)))

    for i, (bl, color) in enumerate(zip(baselines, colors)):
        if bl not in df_vp.columns:
            continue
        if "position" in df_vp.columns:
            vals = [df_vp[df_vp.position==p][bl].mean() for p in positions]
        else:
            vals = [df_vp[bl].mean()]
        ax.bar(x + i*width - width*len(baselines)/2, vals, width,
               label=bl, color=color, edgecolor="white")

    ax.set_xticks(x); ax.set_xticklabels(positions)
    ax.set_ylabel("R²")
    ax.set_title("Variance Partitioning: Explained Variance by Baseline")
    ax.legend(title="Baseline")
    ax.axhline(0, color="k", lw=0.5)

    plt.tight_layout()
    plt.savefig(RESULTS / "vp_summary.png", dpi=150, bbox_inches="tight")
    plt.show()
""")

# ── 13. Attention Blocking ────────────────────────────────────────────────────
md("## 13 · Experiment 7 — Attention Blocking")

code("""\
from verbal_confidence.experiments.attention_blocking import run_attention_blocking

if not VISUALIZE_ONLY:
    attn = run_attention_blocking(cfg, model, tokenizer, p1)
else:
    attn, _ = load(cfg.attention_blocking.output_file)

if attn:
    import pandas as pd
    df_at = pd.DataFrame(attn)
    print(df_at.head())
""")

code("""\
if attn:
    fig, ax = plt.subplots(figsize=(10, 4))
    patterns = sorted(df_at["pattern"].unique()) if "pattern" in df_at else ["overall"]

    if "delta_class" in df_at.columns and "pattern" in df_at.columns:
        means = [df_at[df_at.pattern==p]["delta_class"].mean() for p in patterns]
        stds  = [df_at[df_at.pattern==p]["delta_class"].std()  for p in patterns]
        colors = ["steelblue" if m < 0 else "coral" for m in means]
        ax.bar(range(len(patterns)), means, yerr=stds, capsize=5,
               color=colors, edgecolor="white")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xticks(range(len(patterns)))
        ax.set_xticklabels(patterns, rotation=15, ha="right")
        ax.set_ylabel("Mean ΔClass")
        ax.set_title("Attention Blocking: Effect on Predicted Confidence Class")
    else:
        ax.text(0.5, 0.5, "Check column names in attn results",
                ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(RESULTS / "attention_summary.png", dpi=150, bbox_inches="tight")
    plt.show()
""")

# ── 14. Generalization ────────────────────────────────────────────────────────
md("## 14 · Generalization")

code("""\
from verbal_confidence.experiments.generalization import run_generalization

if not VISUALIZE_ONLY:
    gen = run_generalization(cfg, model, tokenizer)
else:
    gen, _ = load(cfg.generalization.output_file)

if gen:
    import pandas as pd
    df_g = pd.DataFrame(gen)
    print(df_g.head())
""")

code("""\
if gen and "model" in df_g.columns and "mean_confidence" in df_g.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    groups = df_g.groupby(["model","prompt_variant"])["mean_confidence"].mean().unstack()
    groups.plot(kind="bar", ax=ax, edgecolor="white")
    ax.set_ylabel("Mean Confidence")
    ax.set_title("Generalization: Confidence by Model and Prompt Variant")
    ax.legend(title="Prompt Variant")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS / "generalization_summary.png", dpi=150, bbox_inches="tight")
    plt.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {
    "name": "python",
    "version": "3.10.0",
}

out = Path(__file__).parent / "experiments.ipynb"
with open(out, "w") as f:
    nbformat.write(nb, f)
print(f"Written: {out}")
