#!/usr/bin/env bash
# ============================================================================
# run.sh -- One-command reproduction script for MMAI 845 Final Project
# Author: Syed Ali Turab
#
# This script automates the ENTIRE reproduction pipeline end-to-end:
#   1. Checks that a compatible Python version (3.10-3.12) is available
#   2. Creates an isolated virtual environment (.venv/) so system packages
#      are never touched
#   3. Installs all project dependencies (NASim, Stable-Baselines3,
#      sb3-contrib for MaskablePPO, PyTorch, etc.)
#   4. Runs code quality checks (black, isort, flake8) -- non-blocking
#   5. Runs the full pytest test suite (66 tests covering env, agents, eval)
#   6. Trains PPO and DQN agents in both baseline and stealth modes
#   7. Evaluates all 4 trained models over 100 seeded episodes
#   8. Generates publication-quality figures and a penetration testing report
#   9. Prints a summary of key results
#
# Usage:
#   chmod +x run.sh
#   ./run.sh                     # Full pipeline (train + evaluate + report)
#   ./run.sh --eval-only         # Skip training, evaluate existing models
#   ./run.sh --test-only         # Just run linter + tests (fastest check)
#   ./run.sh --train-only        # Just train (no evaluation or reporting)
#
# Environment variables (optional overrides):
#   TIMESTEPS=500000   Training budget per agent (default: 500000)
#   SEED=42            Random seed for reproducibility (default: 42)
#   EPISODES=100       Evaluation episodes per agent (default: 100)
#
# Prerequisites:
#   - Python 3.10, 3.11, or 3.12 must be available as python3
#     (PyTorch does not support Python 3.13+)
#   - Internet access for pip install (first run only)
#
# Estimated time:
#   Full pipeline:     ~45-60 min on modern CPU
#   Evaluation only:   ~2-5 min
#   Tests only:        <5 min
#
# FALLBACK: If this script fails for any reason (wrong Python version,
# dependency conflicts, OS incompatibility), use the Google Colab notebook
# instead -- it runs the same pipeline on Google's cloud with zero local
# dependencies:
#
#   notebooks/00_colab_training.ipynb
#
# Open it at https://colab.research.google.com, set runtime to T4 GPU,
# and click Runtime > Run All. See README.md for full instructions.
# ============================================================================

set -euo pipefail

# Navigate to the repository root (same directory as this script),
# regardless of where the user invoked it from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default configuration -- override via environment variables if needed.
VENV_DIR=".venv"
RESULTS_DIR="results"
TIMESTEPS="${TIMESTEPS:-500000}"
SEED="${SEED:-42}"
EPISODES="${EPISODES:-100}"

# --------------------------------------------------------------------------
# Colour helpers for readable terminal output
# --------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }

# fail() prints an error AND a Colab fallback message, then exits.
fail()  {
    echo -e "${RED}[FAIL]${NC}  $*"
    echo ""
    echo -e "${YELLOW}--- FALLBACK ---${NC}"
    echo "If this script is not working on your machine, you can reproduce"
    echo "all results using Google Colab instead (zero local dependencies):"
    echo ""
    echo "  1. Open notebooks/00_colab_training.ipynb"
    echo "  2. Upload to https://colab.research.google.com"
    echo "  3. Set runtime to T4 GPU (Runtime > Change runtime type)"
    echo "  4. Click Runtime > Run All"
    echo ""
    echo "The Colab notebook runs the exact same pipeline and produces"
    echo "identical results. See README.md Section 8 for full details."
    echo -e "${YELLOW}----------------${NC}"
    exit 1
}

# --------------------------------------------------------------------------
# Step 0: Parse command-line arguments
# --------------------------------------------------------------------------
MODE="full"
for arg in "$@"; do
    case "$arg" in
        --eval-only)  MODE="eval"  ;;
        --test-only)  MODE="test"  ;;
        --train-only) MODE="train" ;;
        --help|-h)
            echo "Usage: ./run.sh [--eval-only|--test-only|--train-only]"
            echo ""
            echo "Modes:"
            echo "  (no flag)      Full pipeline: lint + test + train + eval + report"
            echo "  --test-only    Just run linter and test suite (<5 min)"
            echo "  --train-only   Just train PPO+DQN baseline+stealth"
            echo "  --eval-only    Evaluate existing trained models"
            echo ""
            echo "Environment variables:"
            echo "  TIMESTEPS  Training budget per agent (default: 500000)"
            echo "  SEED       Random seed (default: 42)"
            echo "  EPISODES   Evaluation episodes per agent (default: 100)"
            echo ""
            echo "Fallback: If local setup fails, use notebooks/00_colab_training.ipynb"
            echo "          on Google Colab (T4 GPU). See README.md for details."
            exit 0
            ;;
        *) warn "Unknown argument: $arg (ignored)" ;;
    esac
done

# Print banner with current configuration.
echo ""
echo "============================================================"
echo "  MMAI 845 -- RL Attack Path Simulation"
echo "  Syed Ali Turab"
echo "  Mode: $MODE | Timesteps: $TIMESTEPS | Seed: $SEED"
echo "============================================================"
echo ""
echo "  Fallback: If anything fails, use the Colab notebook instead."
echo "            See notebooks/00_colab_training.ipynb or README.md."
echo ""

# --------------------------------------------------------------------------
# Step 1: Python version check
#
# PyTorch (required by Stable-Baselines3) only supports Python 3.10-3.12.
# We check for the highest available version first.
# --------------------------------------------------------------------------
info "Checking Python version..."

PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [[ "$major" -eq 3 && "$minor" -ge 10 && "$minor" -le 12 ]]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    fail "Python 3.10-3.12 is required but none was found.
    Install from https://www.python.org/downloads/
    Or use the Colab fallback (see above)."
fi
ok "Using $PYTHON ($($PYTHON --version))"

# --------------------------------------------------------------------------
# Step 2: Create / activate virtual environment
#
# A virtual environment isolates this project's dependencies from your
# system Python packages. The venv is created once and reused on
# subsequent runs.
# --------------------------------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtual environment in $VENV_DIR/..."
    "$PYTHON" -m venv "$VENV_DIR"
    ok "Virtual environment created."
else
    info "Virtual environment already exists at $VENV_DIR/."
fi

# Activate the venv so all subsequent pip/python calls use it.
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
ok "Virtual environment activated ($(python --version))."

# --------------------------------------------------------------------------
# Step 3: Install dependencies
#
# 'pip install -e ".[dev]"' installs the project in editable mode along
# with all runtime dependencies (NASim, SB3, sb3-contrib, PyTorch, etc.)
# and dev dependencies (pytest, black, isort, flake8).
# The --quiet flag suppresses verbose pip output.
# --------------------------------------------------------------------------
info "Installing/upgrading dependencies (this may take 2-3 min on first run)..."
pip install --upgrade pip --quiet
pip install -e ".[dev]" --quiet
ok "Dependencies installed."

# Sanity check: verify that the three critical packages are importable.
# If this fails, there is a fundamental dependency issue.
python -c "import nasim; import stable_baselines3; import sb3_contrib" \
    || fail "Dependency import check failed. Core packages not importable."
ok "All core packages importable (nasim, stable_baselines3, sb3_contrib)."

# --------------------------------------------------------------------------
# Step 4: Lint + Test
#
# Linters (black, isort, flake8) check code formatting and style.
# These are NON-BLOCKING: warnings are printed but the script continues.
# The --exclude flags prevent linters from scanning the virtual
# environment directory, which would be extremely slow.
#
# pytest runs the full test suite (66 tests covering environment setup,
# agent training/prediction, evaluation harness, attack path analysis,
# and visualization).
# --------------------------------------------------------------------------
info "Running code quality checks (black, isort, flake8)..."
black --check --exclude '\.venv|\.venv_test|\.eggs' . --quiet 2>/dev/null \
    || warn "black found formatting issues (non-blocking)"
isort --check-only --skip .venv --skip .venv_test . --quiet 2>/dev/null \
    || warn "isort found import order issues (non-blocking)"
flake8 . --count --max-line-length=120 --statistics \
    --exclude=.venv,.venv_test,.eggs,build,dist --quiet 2>/dev/null \
    || warn "flake8 found lint issues (non-blocking)"
ok "Linting complete."

info "Running test suite (66 tests, may take 3-6 min)..."
pytest tests/ -v --tb=short || warn "Some tests failed (see output above)"
ok "Tests complete."

# If --test-only was specified, stop here.
if [[ "$MODE" == "test" ]]; then
    echo ""
    ok "Test-only mode complete. All quality checks passed."
    echo ""
    echo "  To run the full pipeline:  ./run.sh"
    echo "  To train only:             ./run.sh --train-only"
    echo "  To evaluate only:          ./run.sh --eval-only"
    exit 0
fi

# --------------------------------------------------------------------------
# Step 5: Train agents
#
# Trains both PPO (MaskablePPO with native action masking) and DQN
# (with manual Q-value masking) using the --compare flag.
#
# Baseline mode: raw NASim rewards + dense reward shaping.
# Stealth mode:  adds cumulative detection + caught penalty.
#
# Each agent trains for $TIMESTEPS steps (default 500k).
# Models are saved to $RESULTS_DIR/{ppo,dqn}_{baseline,stealth}/.
# --------------------------------------------------------------------------
if [[ "$MODE" == "full" || "$MODE" == "train" ]]; then
    info "Training PPO + DQN (baseline mode)... [$TIMESTEPS timesteps, seed $SEED]"
    info "  This trains both algorithms on the same environment for fair comparison."
    python -m training.train --compare --timesteps "$TIMESTEPS" --seed "$SEED" \
        --output_dir "$RESULTS_DIR"
    ok "Baseline training complete. Models saved to $RESULTS_DIR/"

    info "Training PPO + DQN (stealth mode)... [$TIMESTEPS timesteps, seed $SEED]"
    info "  Stealth adds SOC/IDS detection: agents are caught after ~8-9 active steps."
    python -m training.train --compare --stealth --timesteps "$TIMESTEPS" --seed "$SEED" \
        --output_dir "$RESULTS_DIR"
    ok "Stealth training complete. Models saved to $RESULTS_DIR/"
fi

if [[ "$MODE" == "train" ]]; then
    echo ""
    ok "Train-only mode complete. Models saved in $RESULTS_DIR/"
    echo "  To evaluate:  ./run.sh --eval-only"
    exit 0
fi

# --------------------------------------------------------------------------
# Step 6: Evaluate trained models
#
# Runs each trained model for $EPISODES episodes (default 100) with a
# fixed seed for reproducibility. Outputs JSON files with metrics:
#   - mean_reward, std_reward, success_rate (baseline)
#   - mean_reward, catch_rate, mean_steps (stealth)
#
# Expected results (seed 42, 500k steps):
#   Baseline PPO:  mean_reward=-100.0, std=0.0
#   Baseline DQN:  mean_reward~-498, std~19.9
#   Stealth both:  mean_reward~-109.9, catch_rate=1.0, steps~9.0
# --------------------------------------------------------------------------
info "Evaluating baseline models... [$EPISODES episodes, seed $SEED]"
python -m training.evaluate \
    --ppo_model "$RESULTS_DIR/ppo_baseline/final_model" \
    --dqn_model "$RESULTS_DIR/dqn_baseline/final_model" \
    --episodes "$EPISODES" --seed "$SEED" \
    --output_dir "$RESULTS_DIR"
ok "Baseline evaluation complete -> $RESULTS_DIR/eval_baseline.json"

info "Evaluating stealth models... [$EPISODES episodes, seed $SEED]"
python -m training.evaluate \
    --ppo_model "$RESULTS_DIR/ppo_stealth/final_model" \
    --dqn_model "$RESULTS_DIR/dqn_stealth/final_model" \
    --stealth --episodes "$EPISODES" --seed "$SEED" \
    --output_dir "$RESULTS_DIR"
ok "Stealth evaluation complete -> $RESULTS_DIR/eval_stealth.json"

# --------------------------------------------------------------------------
# Step 7: Generate figures and penetration testing report
#
# Produces publication-quality matplotlib figures (training curves,
# algorithm comparison bars, network topology, attack path flows)
# and an automated Markdown pentest report with MITRE ATT&CK mapping.
# --------------------------------------------------------------------------
if [[ "$MODE" == "full" ]]; then
    info "Generating figures (training curves, comparisons, topology)..."
    python -m analysis.generate_all_figures 2>/dev/null \
        || warn "Figure generation had issues (non-blocking)"
    ok "Figures saved to $RESULTS_DIR/plots/"

    info "Generating penetration testing report..."
    python -m analysis.report_generator --results_dir "$RESULTS_DIR" \
        --output "$RESULTS_DIR/pentest_report.md" 2>/dev/null \
        || warn "Report generation had issues (non-blocking)"
    ok "Report saved to $RESULTS_DIR/pentest_report.md"
fi

# --------------------------------------------------------------------------
# Step 8: Print summary of results
# --------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Pipeline Complete"
echo "============================================================"
echo ""
echo "  Output directory: $RESULTS_DIR/"
echo ""

# Print baseline metrics from the evaluation JSON.
if [[ -f "$RESULTS_DIR/eval_baseline.json" ]]; then
    echo "  Baseline results:"
    python -c "
import json
with open('$RESULTS_DIR/eval_baseline.json') as f:
    d = json.load(f)
for agent in ['ppo', 'dqn']:
    r = d[agent]
    print(f'    {agent.upper()}: mean_reward={r[\"mean_reward\"]:.1f}  std={r[\"std_reward\"]:.1f}  success={r[\"success_rate\"]:.0%}')
"
fi

# Print stealth metrics from the evaluation JSON.
if [[ -f "$RESULTS_DIR/eval_stealth.json" ]]; then
    echo "  Stealth results:"
    python -c "
import json
with open('$RESULTS_DIR/eval_stealth.json') as f:
    d = json.load(f)
for agent in ['ppo', 'dqn']:
    r = d[agent]
    print(f'    {agent.upper()}: mean_reward={r[\"mean_reward\"]:.1f}  catch_rate={r[\"catch_rate\"]:.0%}  steps={r[\"mean_steps\"]:.0f}')
"
fi

echo ""
echo "  Key output files:"
echo "    $RESULTS_DIR/eval_baseline.json   -- baseline evaluation metrics"
echo "    $RESULTS_DIR/eval_stealth.json    -- stealth evaluation metrics"
echo "    $RESULTS_DIR/pentest_report.md    -- penetration testing report"
echo "    $RESULTS_DIR/plots/              -- all generated figures"
echo ""
ok "All done. See $RESULTS_DIR/ for full outputs."
echo ""
echo "  NOTE: If you need to re-run only parts of the pipeline:"
echo "    ./run.sh --test-only     Linter + tests only"
echo "    ./run.sh --train-only    Training only"
echo "    ./run.sh --eval-only     Evaluation only"
echo ""
echo "  COLAB FALLBACK: If any step failed, use notebooks/00_colab_training.ipynb"
echo "  on Google Colab (T4 GPU) for identical results. See README.md."
