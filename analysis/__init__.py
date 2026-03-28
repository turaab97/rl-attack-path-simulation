"""
Analysis and visualisation package.

Author: Syed Ali Turab
Course: MMAI 845 -- Reinforcement Learning
"""

from analysis.attack_path import (
    build_action_map,
    find_common_pivots,
    interpret_path,
    summarise_path,
)
from analysis.mitre_mapping import (
    generate_mitre_summary_table,
    get_mitre_for_action,
    map_path_to_mitre,
)
from analysis.report_generator import (
    generate_pentest_report,
    generate_report_from_results_dir,
)
from analysis.visualize import (
    generate_full_report,
    plot_attack_path,
    plot_comparison_bar,
    plot_detection_sensitivity,
    plot_training_curves,
)

__all__ = [
    "build_action_map",
    "interpret_path",
    "summarise_path",
    "find_common_pivots",
    "get_mitre_for_action",
    "map_path_to_mitre",
    "generate_mitre_summary_table",
    "generate_pentest_report",
    "generate_report_from_results_dir",
    "plot_training_curves",
    "plot_attack_path",
    "plot_detection_sensitivity",
    "plot_comparison_bar",
    "generate_full_report",
]
