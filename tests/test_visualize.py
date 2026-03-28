"""
test_visualize.py
-----------------
Smoke tests for analysis/visualize.py plotting functions.

Author: Syed Ali Turab
Course: MMAI 845 – Reinforcement Learning
"""

import json
from pathlib import Path

import pytest


class TestPlotComparisonBar:
    def test_generates_png(self, tmp_path):
        from analysis.visualize import plot_comparison_bar

        eval_results = {
            "ppo": {
                "mean_reward": 150.0,
                "mean_steps": 45.0,
                "success_rate": 0.7,
                "catch_rate": 0.3,
            },
            "dqn": {
                "mean_reward": 120.0,
                "mean_steps": 50.0,
                "success_rate": 0.6,
                "catch_rate": 0.4,
            },
        }
        out = str(tmp_path / "bar.png")
        plot_comparison_bar(eval_results, output_path=out)
        assert Path(out).exists()

    def test_handles_all_zero_values(self, tmp_path):
        """Regression: should not crash when all metric values are 0."""
        from analysis.visualize import plot_comparison_bar

        eval_results = {
            "ppo": {"mean_reward": 0, "mean_steps": 0, "success_rate": 0, "catch_rate": 0},
            "dqn": {"mean_reward": 0, "mean_steps": 0, "success_rate": 0, "catch_rate": 0},
        }
        out = str(tmp_path / "bar_zero.png")
        plot_comparison_bar(eval_results, output_path=out)
        assert Path(out).exists()

    def test_handles_missing_keys(self, tmp_path):
        from analysis.visualize import plot_comparison_bar

        eval_results = {
            "ppo": {"mean_reward": 100.0},
            "dqn": {"mean_reward": 80.0},
        }
        out = str(tmp_path / "bar_partial.png")
        plot_comparison_bar(eval_results, output_path=out)
        assert Path(out).exists()


class TestPlotAttackPath:
    def test_generates_png(self, tmp_path):
        from analysis.visualize import plot_attack_path

        path = [0, 1, 2, 3, 1, 2]
        out = str(tmp_path / "attack_path.png")
        plot_attack_path(path, output_path=out)
        assert Path(out).exists()

    def test_with_action_meanings(self, tmp_path):
        from analysis.visualize import plot_attack_path

        path = [0, 1, 2]
        meanings = ["noop", "subnet_scan(1)", "e_ssh(1,0)"]
        out = str(tmp_path / "attack_path_meanings.png")
        plot_attack_path(path, action_meanings=meanings, output_path=out)
        assert Path(out).exists()


class TestPlotDetectionSensitivity:
    def test_generates_png(self, tmp_path):
        from analysis.visualize import plot_detection_sensitivity

        thresholds = [0.3, 0.5, 0.8, 1.0, 1.5]
        ppo_rates = [0.2, 0.4, 0.7, 0.85, 0.95]
        dqn_rates = [0.15, 0.35, 0.6, 0.8, 0.9]
        out = str(tmp_path / "sensitivity.png")
        plot_detection_sensitivity(thresholds, ppo_rates, dqn_rates, output_path=out)
        assert Path(out).exists()


class TestGenerateFullReport:
    def test_runs_without_data(self, tmp_path):
        """Should not crash even if no results files exist."""
        from analysis.visualize import generate_full_report

        generate_full_report(
            results_dir=str(tmp_path),
            output_dir=str(tmp_path / "plots"),
        )

    def test_generates_plots_from_eval_results(self, tmp_path):
        from analysis.visualize import generate_full_report

        eval_data = {
            "ppo": {
                "mean_reward": 200.0,
                "mean_steps": 30.0,
                "success_rate": 0.8,
                "catch_rate": 0.2,
            },
            "dqn": {
                "mean_reward": 180.0,
                "mean_steps": 35.0,
                "success_rate": 0.7,
                "catch_rate": 0.3,
            },
        }
        with open(tmp_path / "eval_results.json", "w") as f:
            json.dump(eval_data, f)

        generate_full_report(
            results_dir=str(tmp_path),
            output_dir=str(tmp_path / "plots"),
        )
        assert (tmp_path / "plots" / "comparison_bar.png").exists()
