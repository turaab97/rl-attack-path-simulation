#!/usr/bin/env bash
# ============================================================================
# entrypoint.sh — Docker entrypoint for RL Attack-Path Simulation
# Author: Syed Ali Turab
# Course: MMAI 845 – Reinforcement Learning
#
# Dispatches to the appropriate Python module based on the first argument:
#   train     → python -m training.train (+ remaining args)
#   evaluate  → python -m training.evaluate (+ remaining args)
#   visualize → python -m analysis.visualize (+ remaining args)
#   report    → python -m analysis.report_generator (+ remaining args)
#   test      → pytest tests/ -v
#   *         → execute arbitrary command
# ============================================================================

set -euo pipefail

COMMAND="${1:-help}"
shift || true

case "$COMMAND" in
    train)
        echo "=== Training agents ==="
        exec python -m training.train "$@"
        ;;
    evaluate)
        echo "=== Evaluating agents ==="
        exec python -m training.evaluate "$@"
        ;;
    visualize)
        echo "=== Generating plots ==="
        exec python -m analysis.visualize "$@"
        ;;
    report)
        echo "=== Generating pentest report ==="
        exec python -m analysis.report_generator "$@"
        ;;
    test)
        echo "=== Running test suite ==="
        exec pytest tests/ -v --tb=short "$@"
        ;;
    help)
        echo "RL Attack-Path Simulation — Docker Entrypoint"
        echo ""
        echo "Usage:"
        echo "  docker compose run train      [--agent ppo|dqn] [--stealth] [--timesteps N] ..."
        echo "  docker compose run evaluate   --ppo_model PATH --dqn_model PATH [--stealth] ..."
        echo "  docker compose run visualize  [--results_dir DIR] [--output_dir DIR]"
        echo "  docker compose run report     [--results_dir DIR] [--output PATH]"
        echo "  docker compose run test"
        echo ""
        echo "See README.md for full documentation."
        ;;
    *)
        exec "$COMMAND" "$@"
        ;;
esac
