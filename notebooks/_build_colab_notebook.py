"""Build colab.ipynb — a self-contained Google Colab notebook."""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

cells = []

def md(src):  cells.append(new_markdown_cell(src))
def code(src): cells.append(new_code_cell(src))


# ── Title ────────────────────────────────────────────────────────────────────
md("""\
# Verbal Confidence — Google Colab
Reproduce experiments from *How do LLMs Compute Verbal Confidence?* (Kumaran et al., ICML 2026).

**Model:** Qwen2.5-1.5B-Instruct (≈3 GB, no gating, fits free T4)
**Runtime:** GPU → Runtime → Change runtime type → T4 GPU

Steps:
1. (Optional) Mount Google Drive to persist results
2. Run **Setup** cells once
3. Run experiments top-to-bottom; cached results are reused automatically
""")


# ── 0. GPU check ─────────────────────────────────────────────────────────────
md("## 0 · GPU Check")
code("""\
import subprocess, sys

try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    )
    print("GPU:", result.stdout.strip())
except Exception:
    print("WARNING: No GPU found. Experiments will be very slow on CPU.")
""")


# ── 1. Install ────────────────────────────────────────────────────────────────
md("## 1 · Install Dependencies")
code("""\
# Run once per Colab session (takes ~2 min on first run)
%pip install -q transformers>=4.40 accelerate>=0.27 bitsandbytes>=0.43 \\
    datasets>=2.18 pyyaml>=6.0 scikit-learn>=1.4 matplotlib>=3.8 nbformat
print("Installation complete.")
""")


# ── 2. Clone / locate repo ───────────────────────────────────────────────────
md("## 2 · Clone Repository")
code("""\
import os
from pathlib import Path

# ── Option A: clone from GitHub ──────────────────────────────────────────────
GITHUB_REPO = "https://github.com/YOUR_USERNAME/verbal-confidence.git"  # ← fill in

REPO_ROOT = Path("/content/verbal-confidence")
if not REPO_ROOT.exists():
    os.system(f"git clone {GITHUB_REPO} {REPO_ROOT}")
else:
    print("Repo already present, skipping clone.")

# ── Option B: if you uploaded a zip to Colab Files ───────────────────────────
# import zipfile
# with zipfile.ZipFile("/content/verbal-confidence.zip") as z:
#     z.extractall("/content")

# Install as editable package
os.system(f"pip install -q -e {REPO_ROOT}")
print("Package installed.")
""")


# ── 3. (Optional) Mount Drive ────────────────────────────────────────────────
md("## 3 · (Optional) Mount Google Drive\nSkip this cell to keep results in `/content` (lost when session ends).")
code("""\
USE_DRIVE = False   # ← set True to persist results to Google Drive

if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    PERMANENT_ROOT = "/content/drive/MyDrive/verbal-confidence"
else:
    PERMANENT_ROOT = "/content/results"

EPHEMERAL_ROOT = "/content"

import os
os.environ["EPHEMERAL_ROOT"] = EPHEMERAL_ROOT
os.environ["PERMANENT_ROOT"] = PERMANENT_ROOT
os.environ["HF_HOME"]        = EPHEMERAL_ROOT + "/hf_cache"

print(f"EPHEMERAL_ROOT = {EPHEMERAL_ROOT}")
print(f"PERMANENT_ROOT = {PERMANENT_ROOT}")
""")


# ── 4. HF Token (only needed for gated models) ───────────────────────────────
md("## 4 · HuggingFace Token\nNot required for Qwen2.5-1.5B-Instruct (ungated). Fill in only if switching to a gated model.")
code("""\
import os

HF_TOKEN = ""   # ← paste your token here if needed (e.g. for Gemma)
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    print("Token set.")
else:
    print("No token set — using Qwen2.5-1.5B-Instruct (no token required).")
""")


# ── 5. Config ─────────────────────────────────────────────────────────────────
md("## 5 · Load Config")
code("""\
import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from verbal_confidence.config import load_config

cfg = load_config(str(REPO_ROOT / "config" / "colab.yaml"))

# Override paths from env (in case Drive mount changed them)
cfg.paths.ephemeral_root = os.environ["EPHEMERAL_ROOT"]
cfg.paths.permanent_root = os.environ["PERMANENT_ROOT"]
cfg.paths.hf_home        = os.environ["HF_HOME"]
cfg.paths.hf_datasets    = os.environ["HF_HOME"] + "/datasets"
cfg.paths.model_cache    = os.environ["HF_HOME"] + "/hub"
cfg.paths.results_root   = os.environ["PERMANENT_ROOT"] + "/results/verbal-confidence"
cfg.paths.logs_root      = os.environ["PERMANENT_ROOT"] + "/logs/verbal-confidence"

import pathlib
pathlib.Path(cfg.paths.results_root).mkdir(parents=True, exist_ok=True)
pathlib.Path(cfg.paths.logs_root).mkdir(parents=True, exist_ok=True)

RUN_NAME = "colab_run"       # change to load a previous run's results
cfg.paths.run_name = RUN_NAME

VISUALIZE_ONLY = False       # set True to skip model loading, only plot cached results

print(f"Model  : {cfg.active_model}  ({cfg.models.small.name})")
print(f"Dataset: {cfg.active_dataset}  ({cfg.n_questions} questions)")
print(f"Run    : {cfg.paths.results_root}/{RUN_NAME}")
""")


# ── 6. Load model ────────────────────────────────────────────────────────────
md("## 6 · Load Model\nLoads Qwen2.5-1.5B-Instruct in float16 (~3 GB). Takes ~1–2 min on first run (downloads weights).")
code("""\
model, tokenizer = None, None

if not VISUALIZE_ONLY:
    from verbal_confidence.models.loader import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)
    print("Model ready.")
else:
    print("VISUALIZE_ONLY=True — skipping model load.")
""")


# ── helpers ───────────────────────────────────────────────────────────────────
md("## Helpers")
code("""\
import json, pathlib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["figure.dpi"] = 110

results_dir = pathlib.Path(cfg.paths.results_root) / RUN_NAME
results_dir.mkdir(parents=True, exist_ok=True)

def load(fname):
    p = results_dir / fname
    if p.exists():
        d = json.loads(p.read_text())
        return d.get("results", d), d.get("_meta", {})
    return None, {}

print(f"Results dir: {results_dir}")
""")


# ── Phase 0 ───────────────────────────────────────────────────────────────────
md("## Phase 0 · Answer Generation")
code("""\
from verbal_confidence.experiments.phase0 import run_phase0

p0, _ = load(cfg.phase0.output_file)
if p0 is None:
    if VISUALIZE_ONLY:
        raise RuntimeError("No phase0.json found. Run with VISUALIZE_ONLY=False first.")
    p0 = run_phase0(cfg, model, tokenizer)

print(f"Phase 0: {len(p0)} records")
correct = sum(1 for r in p0 if r.get("correct", False))
print(f"Accuracy: {correct}/{len(p0)} = {correct/len(p0):.1%}")
""")

code("""\
# Accuracy bar
fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(["Correct", "Incorrect"], [correct, len(p0) - correct], color=["#4CAF50", "#F44336"])
ax.set_ylabel("Count")
ax.set_title("Phase 0 — Model Accuracy")
plt.tight_layout()
plt.savefig(results_dir / "phase0_accuracy.png", dpi=150)
plt.show()
""")


# ── Phase 1 ───────────────────────────────────────────────────────────────────
md("## Phase 1 · Confidence Elicitation")
code("""\
from verbal_confidence.experiments.phase1 import run_phase1

p1, _ = load(cfg.phase1.output_file)
if p1 is None:
    if VISUALIZE_ONLY:
        raise RuntimeError("No phase1.json found.")
    p1 = run_phase1(cfg, model, tokenizer, p0)

print(f"Phase 1: {len(p1)} records")
""")

code("""\
import numpy as np

conf_classes = cfg.confidence_classes
n_cls = len(conf_classes)
idx_map = {c: i for i, c in enumerate(conf_classes)}

correct_dist  = [0] * n_cls
incorrect_dist = [0] * n_cls
for r in p1:
    idx = idx_map.get(r.get("confidence_class", ""), -1)
    if idx < 0: continue
    if r.get("correct"):
        correct_dist[idx] += 1
    else:
        incorrect_dist[idx] += 1

x = np.arange(n_cls)
w = 0.4
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x - w/2, correct_dist,   w, label="Correct",   color="#4CAF50", alpha=.8)
ax.bar(x + w/2, incorrect_dist, w, label="Incorrect", color="#F44336", alpha=.8)
ax.set_xticks(x)
ax.set_xticklabels(conf_classes, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Count")
ax.set_title("Phase 1 — Confidence Class Distribution")
ax.legend()
plt.tight_layout()
plt.savefig(results_dir / "phase1_confidence_dist.png", dpi=150)
plt.show()
""")

code("""\
# Calibration curve
numeric = np.linspace(0, 1, n_cls)
bins = np.linspace(0, 1, 6)
bin_labels, bin_acc = [], []
conf_vals = []
acc_vals  = []
for r in p1:
    idx = idx_map.get(r.get("confidence_class", ""), -1)
    if idx < 0: continue
    conf_vals.append(numeric[idx])
    acc_vals.append(float(r.get("correct", False)))

conf_arr = np.array(conf_vals)
acc_arr  = np.array(acc_vals)
bin_confs, bin_accs = [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (conf_arr >= lo) & (conf_arr < hi)
    if mask.sum() > 0:
        bin_confs.append(conf_arr[mask].mean())
        bin_accs.append(acc_arr[mask].mean())

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
ax.plot(bin_confs, bin_accs, "o-", color="#2196F3", label="Model")
ax.set_xlabel("Verbal confidence (normalised)")
ax.set_ylabel("Accuracy")
ax.set_title("Calibration Curve")
ax.legend()
plt.tight_layout()
plt.savefig(results_dir / "phase1_calibration.png", dpi=150)
plt.show()
""")


# ── Exp 1: Steering ──────────────────────────────────────────────────────────
md("## Experiment 1 · Activation Steering")
code("""\
from verbal_confidence.experiments.steering import run_steering

s_res, _ = load(cfg.steering.output_file)
if s_res is None:
    if VISUALIZE_ONLY: raise RuntimeError("No steering.json found.")
    s_res = run_steering(cfg, model, tokenizer, p1)
print(f"Steering: {len(s_res)} records")
""")

code("""\
import pandas as pd

df = pd.DataFrame(s_res)
layers = sorted(df["layer"].unique())
alphas = sorted(df["alpha"].unique())
pivot = df.groupby(["layer", "alpha"])["delta_conf"].mean().unstack("alpha")

fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(alphas)));   ax.set_xticklabels(alphas)
ax.set_yticks(range(len(layers)));  ax.set_yticklabels(layers)
ax.set_xlabel("Alpha"); ax.set_ylabel("Layer")
ax.set_title("Steering — Δ Confidence (mean)")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(results_dir / "steering_heatmap.png", dpi=150)
plt.show()
""")


# ── Exp 2: Patching ──────────────────────────────────────────────────────────
md("## Experiment 2 · Activation Patching")
code("""\
from verbal_confidence.experiments.patching import run_patching

pa_res, _ = load(cfg.patching.output_file)
if pa_res is None:
    if VISUALIZE_ONLY: raise RuntimeError("No patching.json found.")
    pa_res = run_patching(cfg, model, tokenizer, p1)
print(f"Patching: {len(pa_res)} records")
""")

code("""\
df = pd.DataFrame(pa_res)
effect = df.groupby(["layer", "position"])["delta_conf"].mean().reset_index()

fig, ax = plt.subplots(figsize=(6, 4))
for pos, grp in effect.groupby("position"):
    ax.plot(grp["layer"], grp["delta_conf"], marker="o", label=pos)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("Layer"); ax.set_ylabel("Δ Confidence")
ax.set_title("Activation Patching")
ax.legend()
plt.tight_layout()
plt.savefig(results_dir / "patching.png", dpi=150)
plt.show()
""")


# ── Exp 3: Noising ───────────────────────────────────────────────────────────
md("## Experiment 3 · Activation Noising (Mean Ablation)")
code("""\
from verbal_confidence.experiments.noising import run_noising

n_res, _ = load(cfg.noising.output_file)
if n_res is None:
    if VISUALIZE_ONLY: raise RuntimeError("No noising.json found.")
    n_res = run_noising(cfg, model, tokenizer, p1)
print(f"Noising: {len(n_res)} records")
""")

code("""\
df = pd.DataFrame(n_res)
effect = df.groupby(["layer", "position"])["delta_conf"].mean().reset_index()

fig, ax = plt.subplots(figsize=(6, 4))
for pos, grp in effect.groupby("position"):
    ax.plot(grp["layer"], grp["delta_conf"], marker="s", label=pos)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("Layer"); ax.set_ylabel("Δ Confidence")
ax.set_title("Activation Noising (Mean Ablation)")
ax.legend()
plt.tight_layout()
plt.savefig(results_dir / "noising.png", dpi=150)
plt.show()
""")


# ── Exp 4: Swap ──────────────────────────────────────────────────────────────
md("## Experiment 4 · Activation Swap")
code("""\
from verbal_confidence.experiments.swap import run_swap

sw_res, _ = load(cfg.swap.output_file)
if sw_res is None:
    if VISUALIZE_ONLY: raise RuntimeError("No swap.json found.")
    sw_res = run_swap(cfg, model, tokenizer, p1)
print(f"Swap: {len(sw_res)} records")
""")

code("""\
df = pd.DataFrame(sw_res)
effect = df.groupby(["layer", "position"])["delta_conf"].mean().reset_index()

fig, ax = plt.subplots(figsize=(6, 4))
for pos, grp in effect.groupby("position"):
    ax.plot(grp["layer"], grp["delta_conf"], marker="^", label=pos)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("Layer"); ax.set_ylabel("Δ Confidence swap")
ax.set_title("Activation Swap")
ax.legend()
plt.tight_layout()
plt.savefig(results_dir / "swap.png", dpi=150)
plt.show()
""")


# ── Exp 5: Probing ───────────────────────────────────────────────────────────
md("## Experiment 5 · Linear Probing")
code("""\
from verbal_confidence.experiments.probing import run_probing

pr_res, _ = load(cfg.probing.output_file)
if pr_res is None:
    if VISUALIZE_ONLY: raise RuntimeError("No probing.json found.")
    pr_res = run_probing(cfg, model, tokenizer, p1)
print(f"Probing: {len(pr_res)} layer records")
""")

code("""\
df = pd.DataFrame(pr_res)

fig, ax = plt.subplots(figsize=(8, 4))
for pos, grp in df.groupby("position"):
    ax.plot(grp["layer"], grp["r2"], marker="o", label=pos)
ax.set_xlabel("Layer"); ax.set_ylabel("R²")
ax.set_title("Linear Probing — Confidence Decodability by Layer")
ax.legend()
plt.tight_layout()
plt.savefig(results_dir / "probing_r2.png", dpi=150)
plt.show()
""")


# ── Exp 6: Variance Partitioning ─────────────────────────────────────────────
md("## Experiment 6 · Variance Partitioning")
code("""\
from verbal_confidence.experiments.variance_partitioning import run_variance_partitioning

vp_res, _ = load(cfg.variance_partitioning.output_file)
if vp_res is None:
    if VISUALIZE_ONLY: raise RuntimeError("No vp.json found.")
    vp_res = run_variance_partitioning(cfg, model, tokenizer, p1)
print(f"Variance Partitioning: {len(vp_res)} records")
""")

code("""\
df = pd.DataFrame(vp_res)
baselines = df["baseline"].unique()
means = df.groupby("baseline")["r2"].mean()

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(means.index, means.values, color="#2196F3", alpha=0.8)
ax.set_ylabel("R²"); ax.set_title("Variance Partitioning")
ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig(results_dir / "variance_partitioning.png", dpi=150)
plt.show()
""")


# ── Exp 7: Attention Blocking ─────────────────────────────────────────────────
md("## Experiment 7 · Attention Blocking")
code("""\
from verbal_confidence.experiments.attention_blocking import run_attention_blocking

ab_res, _ = load(cfg.attention_blocking.output_file)
if ab_res is None:
    if VISUALIZE_ONLY: raise RuntimeError("No attention.json found.")
    ab_res = run_attention_blocking(cfg, model, tokenizer, p1)
print(f"Attention Blocking: {len(ab_res)} records")
""")

code("""\
df = pd.DataFrame(ab_res)
effect = df.groupby(["pattern", "layer"])["delta_conf"].mean().reset_index()

fig, ax = plt.subplots(figsize=(8, 4))
for pat, grp in effect.groupby("pattern"):
    ax.plot(grp["layer"], grp["delta_conf"], marker="o", label=pat)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("Layer"); ax.set_ylabel("Δ Confidence")
ax.set_title("Attention Blocking")
ax.legend()
plt.tight_layout()
plt.savefig(results_dir / "attention_blocking.png", dpi=150)
plt.show()
""")


# ── Generalization ────────────────────────────────────────────────────────────
md("## Generalization\nTests across prompt variants. Extra models / datasets are skipped in Colab to save time.")
code("""\
from verbal_confidence.experiments.generalization import run_generalization

# Limit to prompt variants only (skip extra models to avoid downloading more weights)
import copy
gen_cfg = copy.deepcopy(cfg)
gen_cfg.generalization.extra_models   = []
gen_cfg.generalization.extra_datasets = []

gen_res, _ = load(cfg.generalization.output_file)
if gen_res is None:
    if VISUALIZE_ONLY: raise RuntimeError("No generalization.json found.")
    gen_res = run_generalization(gen_cfg, model, tokenizer)
print(f"Generalization: {len(gen_res)} records")
""")

code("""\
df = pd.DataFrame(gen_res)
acc = df.groupby("prompt_variant")["correct"].mean()

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(acc.index, acc.values, color="#9C27B0", alpha=0.8)
ax.set_ylabel("Accuracy"); ax.set_title("Generalization — Accuracy by Prompt Variant")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(results_dir / "generalization.png", dpi=150)
plt.show()
""")


# ── Summary ───────────────────────────────────────────────────────────────────
md("## Summary\nAll plots saved to the results directory.")
code("""\
import glob
pngs = sorted(glob.glob(str(results_dir / "*.png")))
print(f"Saved {len(pngs)} plots to {results_dir}:")
for p in pngs:
    print(" ", pathlib.Path(p).name)
""")


# ── Build notebook ───────────────────────────────────────────────────────────
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
nb.metadata["colab"] = {
    "provenance": [],
    "gpuType": "T4",
}
nb.metadata["accelerator"] = "GPU"

out = Path(__file__).parent / "colab.ipynb"
with open(out, "w") as f:
    nbformat.write(nb, f)
print(f"Written: {out}")
