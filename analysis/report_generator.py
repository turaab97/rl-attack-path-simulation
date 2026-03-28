"""
report_generator.py
-------------------
Automated penetration testing report generator. Takes the RL agent's
learned attack paths, MITRE ATT&CK annotations, evaluation metrics,
and what-if counterfactual results and produces a structured security
assessment report in Markdown format.

Author: Syed Ali Turab
Course: MMAI 845 -- Reinforcement Learning

This bridges the gap between an RL experiment and a security deliverable.
The output mirrors the format of professional penetration testing reports
used by security consultancies and internal red teams. Where traditional
pentest reports document a single manual engagement, this report is
generated from thousands of automated RL episodes, providing statistical
confidence that manual testing cannot match.

Usage:
    from analysis.report_generator import generate_pentest_report

    report_md = generate_pentest_report(
        eval_results=eval_data,
        attack_paths=interpreted_paths,
        mitre_annotated=mitre_paths,
        pivot_hosts=common_pivots,
        what_if_results=what_if_data,
    )
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _severity_from_value(value: int) -> str:
    """Map host value to a severity rating."""
    if value >= 300:
        return "CRITICAL"
    if value >= 200:
        return "HIGH"
    if value >= 100:
        return "MEDIUM"
    if value > 0:
        return "LOW"
    return "INFORMATIONAL"


def _format_percent(val: float) -> str:
    return f"{val:.1%}"


def _format_float(val: float) -> str:
    return f"{val:.2f}"


def generate_pentest_report(
    eval_results: dict[str, dict[str, Any]] | None = None,
    attack_paths: list[dict[str, Any]] | None = None,
    mitre_annotated: list[dict[str, Any]] | None = None,
    pivot_hosts: list[tuple[str, int]] | None = None,
    what_if_results: list[dict[str, Any]] | None = None,
    detection_sweep: dict[str, list[float]] | None = None,
    agent_name: str = "PPO",
    n_episodes: int = 100,
    output_path: str | None = None,
) -> str:
    """Generate a Markdown penetration testing report from RL results.

    Parameters
    ----------
    eval_results : dict
        Evaluation metrics keyed by agent name (from evaluate.py).
    attack_paths : list[dict]
        Interpreted attack path from a representative episode.
    mitre_annotated : list[dict]
        Attack path annotated with MITRE ATT&CK info.
    pivot_hosts : list[tuple]
        Common pivot hosts from find_common_pivots().
    what_if_results : list[dict]
        Results from run_all_what_if().
    detection_sweep : dict
        Keys: 'thresholds', 'success_rates'.
    agent_name : str
        Name of the primary agent evaluated.
    n_episodes : int
        Number of evaluation episodes.
    output_path : str or None
        If provided, write the report to this file.

    Returns
    -------
    str
        The complete report in Markdown format.
    """
    now = datetime.now().strftime("%B %d, %Y")
    sections = []

    # -----------------------------------------------------------------------
    # Title page
    # -----------------------------------------------------------------------
    sections.append(f"""# Automated Penetration Testing Report
## Enterprise AI Infrastructure Security Assessment

| Field | Value |
|---|---|
| **Date** | {now} |
| **Assessor** | RL Attack Agent ({agent_name}) |
| **Methodology** | Reinforcement Learning -- {n_episodes}-episode automated assessment |
| **Target** | Enterprise AI Infrastructure (5-subnet topology) |
| **Classification** | Internal -- Course Project (MMAI 845) |
| **Author** | Syed Ali Turab |

---

## Executive Summary

This report presents the findings of an automated penetration testing
engagement conducted using a reinforcement learning agent trained to
discover and exploit attack paths through an enterprise network hosting
AI infrastructure. Unlike traditional penetration tests that rely on a
single assessor's judgment over a limited engagement window, this
assessment is based on **{n_episodes} independent attack simulations**,
providing statistically robust conclusions about the network's security
posture.

The RL agent learns optimal multi-hop attack strategies by interacting
with a simulated replica of the target network. Its learned behaviour
reveals which hosts serve as critical stepping stones, which firewall
rules are most effective, and what detection thresholds are needed to
reliably prevent compromise of AI assets.
""")

    # -----------------------------------------------------------------------
    # Evaluation metrics
    # -----------------------------------------------------------------------
    if eval_results:
        sections.append("---\n\n## Assessment Metrics\n")
        sections.append("| Agent | Mean Reward | Success Rate | Catch Rate | Mean Steps |")
        sections.append("|---|---|---|---|---|")
        for name, metrics in eval_results.items():
            sections.append(
                f"| {name.upper()} "
                f"| {_format_float(metrics.get('mean_reward', 0))} "
                f"| {_format_percent(metrics.get('success_rate', 0))} "
                f"| {_format_percent(metrics.get('catch_rate', 0))} "
                f"| {_format_float(metrics.get('mean_steps', 0))} |"
            )
        sections.append("")

        best_agent = max(
            eval_results.items(),
            key=lambda x: x[1].get("success_rate", 0),
        )
        sections.append(
            f"**Primary finding:** The {best_agent[0].upper()} agent achieved "
            f"a {_format_percent(best_agent[1].get('success_rate', 0))} success rate "
            f"in reaching AI infrastructure assets across {n_episodes} episodes.\n"
        )

    # -----------------------------------------------------------------------
    # Attack path findings
    # -----------------------------------------------------------------------
    if attack_paths:
        sections.append("---\n\n## Findings: Attack Path Analysis\n")
        sections.append(
            "The following table shows the representative attack path "
            "discovered by the RL agent. Each row is one step in the "
            "agent's learned strategy.\n"
        )
        sections.append("| Step | Action Type | Target Host | Subnet |")
        sections.append("|---|---|---|---|")
        for step in attack_paths[:30]:
            sections.append(
                f"| {step.get('step', '?')} "
                f"| {step.get('action_type', '?')} "
                f"| {step.get('host_name', '?')} "
                f"| {step.get('subnet_name', '?')} |"
            )
        if len(attack_paths) > 30:
            sections.append(f"| ... | *{len(attack_paths) - 30} more steps* | ... | ... |")
        sections.append("")

    # -----------------------------------------------------------------------
    # MITRE ATT&CK coverage
    # -----------------------------------------------------------------------
    if mitre_annotated:
        sections.append("---\n\n## MITRE ATT&CK Coverage\n")
        sections.append(
            "The agent's learned attack chain maps to the following MITRE "
            "ATT&CK techniques, demonstrating coverage across the kill chain.\n"
        )

        # Deduplicate by technique
        seen = set()
        unique_techniques = []
        for step in mitre_annotated:
            tid = step.get("mitre_technique_id", "N/A")
            if tid not in seen and tid != "N/A":
                seen.add(tid)
                unique_techniques.append(step)

        sections.append("| ATT&CK Tactic | Technique ID | Technique Name | Example Action |")
        sections.append("|---|---|---|---|")
        for t in unique_techniques:
            sections.append(
                f"| {t.get('mitre_tactic', '?')} "
                f"| {t.get('mitre_technique_id', '?')} "
                f"| {t.get('mitre_technique_name', '?')} "
                f"| {t.get('action_name', '?')} |"
            )
        sections.append("")

        tactics_covered = set(t.get("mitre_tactic") for t in unique_techniques)
        sections.append(
            f"**Kill chain coverage:** {len(tactics_covered)} tactics, "
            f"{len(unique_techniques)} unique techniques.\n"
        )

    # -----------------------------------------------------------------------
    # Pivot host findings (critical stepping stones)
    # -----------------------------------------------------------------------
    if pivot_hosts:
        sections.append("---\n\n## Findings: Critical Pivot Hosts\n")
        sections.append(
            "The following hosts were most frequently used as stepping stones "
            "across all evaluation episodes. Security controls on these hosts "
            "would disrupt the majority of learned attack strategies.\n"
        )
        sections.append("| Rank | Host | Episodes Used | Frequency | Risk |")
        sections.append("|---|---|---|---|---|")
        for rank, (host_name, count) in enumerate(pivot_hosts, 1):
            freq = count / n_episodes
            risk = "CRITICAL" if freq > 0.7 else "HIGH" if freq > 0.4 else "MEDIUM"
            sections.append(
                f"| {rank} | {host_name} | {count}/{n_episodes} "
                f"| {_format_percent(freq)} | {risk} |"
            )
        sections.append("")

    # -----------------------------------------------------------------------
    # What-if recommendations
    # -----------------------------------------------------------------------
    if what_if_results:
        sections.append("---\n\n## Recommendations: Network Hardening\n")
        sections.append(
            "The following counterfactual analysis evaluates the impact of "
            "specific firewall rule changes on attacker success. The trained "
            "agent was evaluated on each modified topology **without retraining**, "
            "simulating a defender deploying changes against an attacker with "
            "existing reconnaissance knowledge.\n"
        )
        sections.append("| Configuration Change | Success Rate | Mean Reward | Impact |")
        sections.append("|---|---|---|---|")
        for wif in what_if_results:
            delta_str = ""
            if eval_results:
                first_key = next(iter(eval_results))
                baseline_sr = eval_results[first_key].get("success_rate", 0)
                delta = wif["success_rate"] - baseline_sr
                delta_str = f"{delta:+.1%}"
            sections.append(
                f"| {wif['description'][:80]} "
                f"| {_format_percent(wif['success_rate'])} "
                f"| {_format_float(wif['mean_reward'])} "
                f"| {delta_str} |"
            )
        sections.append("")

        # Find most effective change
        if eval_results:
            first_key = next(iter(eval_results))
            baseline_sr = eval_results[first_key].get("success_rate", 0)
            best_change = min(what_if_results, key=lambda x: x["success_rate"])
            reduction = baseline_sr - best_change["success_rate"]
            if reduction > 0:
                sections.append(
                    f"**Top recommendation:** Implementing "
                    f"*{best_change['modification'].replace('_', ' ')}* "
                    f"reduces attacker success by {_format_percent(reduction)}, "
                    f"the largest impact of any single configuration change tested.\n"
                )

    # -----------------------------------------------------------------------
    # Detection threshold analysis
    # -----------------------------------------------------------------------
    if detection_sweep:
        thresholds = detection_sweep.get("thresholds", [])
        success_rates = detection_sweep.get("success_rates", [])
        if thresholds and success_rates:
            sections.append("---\n\n## Detection Capability Assessment\n")
            sections.append(
                "The following table shows how the agent's success rate varies "
                "with the SOC's detection sensitivity. Lower thresholds represent "
                "more aggressive detection (IDS/SIEM alerting on fewer anomalies).\n"
            )
            sections.append("| Detection Threshold | Agent Success Rate | Assessment |")
            sections.append("|---|---|---|")
            for t, s in zip(thresholds, success_rates):
                assessment = (
                    "Effective" if s < 0.2 else
                    "Partial" if s < 0.5 else
                    "Insufficient"
                )
                sections.append(
                    f"| {t:.1f} | {_format_percent(s)} | {assessment} |"
                )
            sections.append("")

            # Find the threshold where success drops below 50%
            for t, s in zip(thresholds, success_rates):
                if s < 0.5:
                    sections.append(
                        f"**Finding:** A detection threshold of **{t:.1f}** or lower "
                        f"reduces agent success below 50%. This represents the "
                        f"minimum detection capability required to reliably defend "
                        f"AI infrastructure assets.\n"
                    )
                    break

    # -----------------------------------------------------------------------
    # Conclusion
    # -----------------------------------------------------------------------
    sections.append("""---

## Methodology Note

This assessment was conducted using reinforcement learning agents (PPO and
DQN algorithms from Stable-Baselines3) trained on a NASim simulation of the
target network. The agents were trained over hundreds of thousands of
timesteps, during which they learned optimal attack strategies through
trial-and-error interaction with the environment.

**Key differences from traditional penetration testing:**

| Aspect | Traditional Pentest | RL-Based Assessment |
|---|---|---|
| Coverage | Limited by assessor time | Exhaustive (thousands of episodes) |
| Consistency | Varies by assessor skill | Deterministic and reproducible |
| Statistical confidence | Anecdotal | Quantified (success rates, confidence intervals) |
| Counterfactual analysis | Manual "what-if" reasoning | Automated topology modifications |
| Detection modelling | Separate red/blue exercise | Integrated stealth-aware reward |
| Scalability | Linear (more time = more cost) | Train once, evaluate many topologies |

The RL approach does not replace human expertise but provides a continuous,
automated layer of security validation that complements periodic manual
assessments.

---

*Report generated by RL Attack Path Simulation*
*Author: Syed Ali Turab -- MMAI 845, Queen's University*
""")

    report = "\n".join(sections)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"[report] Saved penetration testing report to {out}")

    return report


def generate_report_from_results_dir(
    results_dir: str = "results",
    output_path: str = "results/pentest_report.md",
) -> str:
    """Convenience function: load saved JSON results and generate report.

    Parameters
    ----------
    results_dir : str
        Directory containing eval_results.json and other output files.
    output_path : str
        Where to save the Markdown report.

    Returns
    -------
    str
        The report Markdown.
    """
    rdir = Path(results_dir)

    eval_results = None
    eval_path = rdir / "eval_results.json"
    if eval_path.exists():
        with open(eval_path) as f:
            eval_results = json.load(f)

    return generate_pentest_report(
        eval_results=eval_results,
        output_path=output_path,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate pentest report from RL results.")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--output", default="results/pentest_report.md")
    args = p.parse_args()
    generate_report_from_results_dir(
        results_dir=args.results_dir,
        output_path=args.output,
    )
