#!/usr/bin/env bash
# ============================================================================
# run.sh -- One-command reproduction script for MMAI 845 Final Project
# Syed Ali Turab
#
# Handles: venv creation, dependency installation, linting, tests,
#          training (PPO + DQN, baseline + stealth), evaluation, and
#          report/figure generation.
#
# Usage:
#   chmod +x run.sh
#   ./run.sh                     # Full pipeline (train + evaluate + report)
#   ./run.sh --eval-only         # Skip training, evaluate existing models
#   ./run.sh --test-only         # Just run linter + tests
#   ./run.sh --train-only        # Just train (no evaluation or reporting)
#
# Prerequisites:
#   - Python 3.10, 3.11, or 3.12 must be available as python3
#   - Internet access for pip install (first run only)
#
# Estimated time:
#   Full pipeline: 30-60 min on modern CPU
#   Evaluation only: 2-5 min
#   Tests only: <1 min
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
RESULTS_DIR="results"
TIMESTEPS="${TIMESTEPS:-500000}"
SEED="${SEED:-42}"
EPISODES="${EPISODES:-100}"

# --------------------------------------------------------------------------
# Colour helpers
# --------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

# --------------------------------------------------------------------------
# Step 0: Parse arguments
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
            echo "Environment variables:"
            echo "  TIMESTEPS  Training budget per agent (default: 500000)"
            echo "  SEED       Random seed (default: 42)"
            echo "  EPISODES   Evaluation episodes per agent (default: 100)"
            exit 0
            ;;
        *) warn "Unknown argument: $arg (ignored)" ;;
    esac
done

echo ""
echo "============================================================"
echo "  MMAI 845 -- RL Attack Path Simulation"
echo "  Syed Ali Turab"
echo "  Mode: $MODE | Timesteps: $TIMESTEPS | Seed: $SEED"
echo "============================================================"
echo ""

# --------------------------------------------------------------------------
# Step 1: Python version check
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
    fail "Python 3.10-3.12 is required. Found none. Install from https://www.python.org/"
fi
ok "Using $PYTHON ($($PYTHON --version))"

# --------------------------------------------------------------------------
# Step 2: Create / activate virtual environment
# --------------------------------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtual environment in $VENV_DIR..."
    "$PYTHON" -m venv "$VENV_DIR"
    ok "Virtual environment created."
else
    info "Virtual environment already exists."
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
ok "Virtual environment activated."

# --------------------------------------------------------------------------
# Step 3: Install dependencies
# --------------------------------------------------------------------------
info "Installing/upgrading dependencies..."
pip install --upgrade pip --quiet
pip install -e ".[dev]" --quiet
ok "Dependencies installed."

# Quick sanity check
python -c "import nasim; import stable_baselines3; import sb3_contrib" \
    || fail "Dependency import check failed."
ok "All core packages importable."

# --------------------------------------------------------------------------
# Step 4: Lint + Test
# --------------------------------------------------------------------------
info "Running linter (black, isort, flake8)..."
black --check . --quiet 2>/dev/null || warn "black found formatting issues (non-blocking)"
isort --check-only . --quiet 2>/dev/null || warn "isort found import order issues (non-blocking)"
flake8 . --count --max-line-length=120 --statistics --quiet 2>/dev/null || warn "flake8 found lint issues (non-blocking)"
ok "Linting complete."

info "Running test suite..."
pytest tests/ -v --tb=short || warn "Some tests failed (see output above)"
ok "Tests complete."

if [[ "$MODE" == "test" ]]; then
    echo ""
    ok "Test-only mode complete. Exiting."
    exit 0
fi

# --------------------------------------------------------------------------
# Step 5: Train agents
# --------------------------------------------------------------------------
if [[ "$MODE" == "full" || "$MODE" == "train" ]]; then
    info "Training PPO + DQN (baseline)... [$TIMESTEPS timesteps, seed $SEED]"
    python -m training.train --compare --timesteps "$TIMESTEPS" --seed "$SEED" \
        --output_dir "$RESULTS_DIR"
    ok "Baseline training complete."

    info "Training PPO + DQN (stealth)... [$TIMESTEPS timesteps, seed $SEED]"
    python -m training.train --compare --stealth --timesteps "$TIMESTEPS" --seed "$SEED" \
        --output_dir "$RESULTS_DIR"
    ok "Stealth training complete."
fi

if [[ "$MODE" == "train" ]]; then
    echo ""
    ok "Train-only mode complete. Models saved in $RESULTS_DIR/"
    exit 0
fi

# --------------------------------------------------------------------------
# Step 6: Evaluate trained models
# --------------------------------------------------------------------------
info "Evaluating baseline models... [$EPISODES episodes, seed $SEED]"
python -m training.evaluate \
    --ppo_model "$RESULTS_DIR/ppo_baseline/final_model" \
    --dqn_model "$RESULTS_DIR/dqn_baseline/final_model" \
    --episodes "$EPISODES" --seed "$SEED" \
    --output_dir "$RESULTS_DIR"
ok "Baseline evaluation complete."

info "Evaluating stealth models... [$EPISODES episodes, seed $SEED]"
python -m training.evaluate \
    --ppo_model "$RESULTS_DIR/ppo_stealth/final_model" \
    --dqn_model "$RESULTS_DIR/dqn_stealth/final_model" \
    --stealth --episodes "$EPISODES" --seed "$SEED" \
    --output_dir "$RESULTS_DIR"
ok "Stealth evaluation complete."

# --------------------------------------------------------------------------
# Step 7: Generate figures and report
# --------------------------------------------------------------------------
if [[ "$MODE" == "full" ]]; then
    info "Generating figures..."
    python -m analysis.generate_all_figures 2>/dev/null || warn "Figure generation had issues (non-blocking)"
    ok "Figures saved to $RESULTS_DIR/plots/"

    info "Generating penetration testing report..."
    python -m analysis.report_generator --results_dir "$RESULTS_DIR" \
        --output "$RESULTS_DIR/pentest_report.md" 2>/dev/null \
        || warn "Report generation had issues (non-blocking)"
    ok "Report saved to $RESULTS_DIR/pentest_report.md"
fi

# --------------------------------------------------------------------------
# Step 8: Summary
# --------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Pipeline Complete"
echo "============================================================"
echo ""
echo "  Output directory: $RESULTS_DIR/"
echo ""

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
ok "All done. See $RESULTS_DIR/ for full outputs."
