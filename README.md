# Verbal Confidence

Reproducibility code for **"How do LLMs Compute Verbal Confidence?"** (Kumaran et al., ICML 2026, Google DeepMind).

Implements all 7 mechanistic interpretability experiments tailored for the **Goethe-HLR Frankfurt cluster** (AMD MI210 GPUs, SLURM).

---

## Repository Layout

```
verbal-confidence/
├── config/
│   └── default.yaml          # All paths, models, datasets, hyperparameters
├── src/verbal_confidence/
│   ├── config.py             # Config loader + HF cache env setup
│   ├── data/
│   │   ├── loader.py         # TriviaQA, BigMath, MMLU loading
│   │   └── prompts.py        # Phase 0/1 prompt templates
│   ├── models/
│   │   ├── loader.py         # Model + tokenizer loading, hot-swap
│   │   └── inference.py      # Generation, logits, ActCollector hooks
│   ├── experiments/
│   │   ├── phase0.py         # Answer generation
│   │   ├── phase1.py         # Confidence elicitation
│   │   ├── steering.py       # Exp 1: Activation Steering
│   │   ├── patching.py       # Exp 2: Activation Patching
│   │   ├── noising.py        # Exp 3: Activation Noising
│   │   ├── swap.py           # Exp 4: Activation Swap
│   │   ├── probing.py        # Exp 5: Linear Probing
│   │   ├── variance_partitioning.py  # Exp 6: Variance Partitioning
│   │   ├── attention_blocking.py     # Exp 7: Attention Blocking
│   │   └── generalization.py # Multi-model / multi-dataset sweep
│   └── utils/
│       ├── io.py             # JSON / NumPy I/O
│       ├── logging.py        # Logger setup
│       └── tokens.py         # Position finding, CLASS_TIDS
├── scripts/
│   ├── run_phase0.py
│   ├── run_phase1.py
│   └── run_all_experiments.py
├── slurm/
│   ├── env_setup.sh          # Sourced by all job scripts
│   ├── phase0.sh
│   ├── phase1.sh
│   ├── experiments.sh
│   ├── test_run.sh           # Quick smoke test (<2h, test partition)
│   └── submit_all.sh         # Submit full dependency chain
├── requirements.txt
└── setup.py
```

---

## Storage Layout (Goethe-HLR)

| Path | Size | Purpose |
|---|---|---|
| `/home/$USER/verbal-confidence` | ~30 GB quota | Source code only |
| `/scratch/$GROUP/$USER/hf_cache` | 5 TB Lustre | HF model weights + datasets |
| `/scratch/$GROUP/$USER/results` | 5 TB Lustre | Experiment outputs |
| `/local/$SLURM_JOB_ID` | 1.4 TB NVMe | Temp cache during job (deleted after) |

The config and SLURM scripts automatically route HF_HOME, HF_DATASETS_CACHE, and TRANSFORMERS_CACHE to `/scratch`.

---

## Setup

### 1. Clone to your home directory

```bash
# On HLR login node
cd $HOME
git clone https://github.com/YOUR_USER/verbal-confidence.git
```

### 2. Create a Python environment

```bash
module load Python/3.11   # or your available module
python -m venv $HOME/envs/verbal-confidence
source $HOME/envs/verbal-confidence/bin/activate
pip install -e .
```

### 3. Configure paths

Edit `config/default.yaml` — at minimum set `paths.scratch_root`:

```yaml
paths:
  scratch_root: /scratch/YOUR_GROUP/YOUR_USER
```

Or pass `GROUP` and `USER` as environment variables (already set in SLURM jobs).

### 4. HuggingFace token (for gated models)

Gemma 3 27B requires licence acceptance at huggingface.co and a read token:

```bash
huggingface-cli login   # paste your HF_TOKEN
```

Or set `HF_TOKEN` in your environment before submitting jobs.

---

## Running

### Quick smoke test (interactive / test partition)

```bash
# Interactive session on test partition
salloc --partition=test --nodes=1 --gres=gpu:2 --time=01:00:00

# Then run
source slurm/env_setup.sh
python scripts/run_all_experiments.py --skip generalization variance_partitioning
```

### Submit full pipeline (dependency chain)

```bash
bash slurm/submit_all.sh
# or with a different model:
bash slurm/submit_all.sh --model qwen
```

### Individual stages

```bash
sbatch slurm/phase0.sh
sbatch slurm/phase1.sh
sbatch slurm/experiments.sh
```

### Override config at runtime

```bash
# config/my_run.yaml (partial override)
n_questions: 500
active_model: qwen
steering:
  alphas: [-3.0, -1.0, 1.0, 3.0]
```

```bash
sbatch slurm/experiments.sh --config config/my_run.yaml
```

---

## Models

| Key | Model | Notes |
|---|---|---|
| `primary` | `google/gemma-3-27b-it` | Gated — accept licence on HF |
| `qwen` | `Qwen/Qwen2.5-7B-Instruct` | Open |
| `magistral` | `mistralai/Magistral-Small-2506` | Open |

Switch model: `--model qwen` or set `active_model: qwen` in config.

---

## Confidence Classes

The 10 verbal confidence expressions used in the paper (index 0 = least confident):

```
0: No chance          5: Likely
1: Really unlikely    6: Good chance
2: Chances are slight 7: Very good chance
3: Unlikely           8: Highly likely
4: About even         9: Almost certain
```

---

## Caching

Every experiment checks for a cached output file before running.
To rerun an experiment, delete its output file:

```bash
rm $SCRATCH/results/verbal-confidence/$RUN_NAME/steering.json
```

---

## Citation

```bibtex
@inproceedings{kumaran2026verbal,
  title     = {How do {LLMs} Compute Verbal Confidence?},
  author    = {Kumaran, Divyansh and others},
  booktitle = {ICML},
  year      = {2026},
}
```
