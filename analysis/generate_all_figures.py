"""
generate_all_figures.py
-----------------------
Generates all publication-quality figures for the MMAI 845 final project
from the saved evaluation JSON files.

Author: Syed Ali Turab
Course: MMAI 845 -- Reinforcement Learning

Run:
    python -m analysis.generate_all_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = Path("results")
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

BLUE = "#2196F3"
ORANGE = "#FF5722"
PURPLE = "#9C27B0"
GREEN = "#4CAF50"
RED = "#EF5350"
GREY = "#78909C"
DARK = "#263238"
TEAL = "#009688"


def _load_json(name: str) -> dict:
    with open(RESULTS / name) as f:
        return json.load(f)


# -----------------------------------------------------------------------
# 1. Network Topology Diagram
# -----------------------------------------------------------------------


def fig_topology():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 12)
    ax.set_aspect("equal")
    ax.axis("off")

    subnet_cfg = [
        {
            "name": "INTERNET\n(Attacker Entry)",
            "y": 11,
            "x": 5,
            "w": 3,
            "h": 0.9,
            "color": "#B71C1C",
            "hosts": [],
        },
        {
            "name": "SUBNET 1 -- DMZ",
            "y": 8.5,
            "x": 5,
            "w": 9,
            "h": 1.5,
            "color": "#1565C0",
            "hosts": [
                "(1,0) Web Server\nLinux | HTTP,SSH",
                "(1,1) Email Gateway\nWindows | SMB,RDP",
                "(1,2) Proxy Server\nLinux | SSH,HTTP",
            ],
        },
        {
            "name": "SUBNET 2 -- Corporate LAN",
            "y": 5.5,
            "x": 5,
            "w": 9,
            "h": 1.5,
            "color": "#2E7D32",
            "hosts": [
                "(2,0) Workstation\nWin | SMB,RDP",
                "(2,1) App Server\nLinux | SSH,HTTP",
                "(2,2) File Server\nWin | SMB,RDP",
                "(2,3) AD Controller\nLinux | SSH",
            ],
        },
        {
            "name": "SUBNET 3 -- AI Infrastructure",
            "y": 2.5,
            "x": 5,
            "w": 9,
            "h": 1.5,
            "color": "#E65100",
            "hosts": [
                "(3,0) LLM API\nLinux | HTTP,SSH\nVALUE: 200",
                "(3,1) Vector DB\nLinux | HTTP\nVALUE: 200",
                "(3,2) Model Repo\nLinux | SSH\nVALUE: 150",
            ],
        },
        {
            "name": "SUBNET 4 -- Data Lake",
            "y": 0,
            "x": 5,
            "w": 5,
            "h": 1.0,
            "color": "#880E4F",
            "hosts": ["(4,0) Training Data Store\nLinux | SSH | VALUE: 300"],
        },
    ]

    for cfg in subnet_cfg:
        x0 = cfg["x"] - cfg["w"] / 2
        y0 = cfg["y"]
        rect = plt.Rectangle(
            (x0, y0),
            cfg["w"],
            cfg["h"],
            facecolor=cfg["color"],
            alpha=0.15,
            edgecolor=cfg["color"],
            linewidth=2,
            linestyle="--",
            zorder=1,
        )
        ax.add_patch(rect)
        ax.text(
            x0 + 0.15,
            y0 + cfg["h"] - 0.15,
            cfg["name"],
            fontsize=9,
            fontweight="bold",
            color=cfg["color"],
            va="top",
            ha="left",
            zorder=3,
        )

        n = len(cfg["hosts"])
        if n > 0:
            spacing = cfg["w"] / (n + 1)
            for i, host in enumerate(cfg["hosts"]):
                hx = x0 + spacing * (i + 1)
                hy = y0 + cfg["h"] / 2 - 0.1
                box = plt.Rectangle(
                    (hx - 0.9, hy - 0.35),
                    1.8,
                    0.7,
                    facecolor="white",
                    edgecolor=cfg["color"],
                    linewidth=1.5,
                    zorder=2,
                )
                ax.add_patch(box)
                ax.text(
                    hx, hy, host, fontsize=5.5, ha="center", va="center", zorder=3, linespacing=1.3
                )

    fw_labels = [
        (5, 10.9, 8.5 + 1.5, "Firewall: HTTP,SSH,SMB,RDP"),
        (5, 8.4, 5.5 + 1.5, "Firewall: SSH,SMB,RDP,HTTP"),
        (5, 5.4, 2.5 + 1.5, "Firewall: HTTP,SSH"),
        (5, 2.4, 0 + 1.0, "Firewall: SSH only"),
    ]
    for x, y_top, y_bot, label in fw_labels:
        ax.annotate(
            "",
            xy=(x, y_bot + 0.05),
            xytext=(x, y_top - 0.05),
            arrowprops=dict(arrowstyle="->", color=DARK, lw=1.8),
            zorder=2,
        )
        mid_y = (y_top + y_bot) / 2
        ax.text(
            x + 0.15,
            mid_y,
            label,
            fontsize=7,
            color=RED,
            fontweight="bold",
            va="center",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=RED, alpha=0.9),
            zorder=4,
        )

    ax.set_title(
        "Enterprise AI Infrastructure -- Network Topology", fontsize=14, fontweight="bold", pad=15
    )
    fig.tight_layout()
    path = PLOTS / "network_topology.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# -----------------------------------------------------------------------
# 2. PPO vs DQN -- Baseline Evaluation Bar Chart
# -----------------------------------------------------------------------


def fig_baseline_comparison():
    data = _load_json("eval_baseline.json")

    metrics = {
        "Mean Reward": (data["ppo"]["mean_reward"], data["dqn"]["mean_reward"]),
        "Std Reward": (data["ppo"]["std_reward"], data["dqn"]["std_reward"]),
        "Max Reward": (data["ppo"]["max_reward"], data["dqn"]["max_reward"]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (label, (ppo_v, dqn_v)) in zip(axes, metrics.items()):
        bars = ax.bar(
            ["PPO\n(MaskablePPO)", "DQN\n(Masked Q)"],
            [ppo_v, dqn_v],
            color=[BLUE, ORANGE],
            edgecolor="white",
            width=0.5,
        )
        for bar, val in zip(bars, [ppo_v, dqn_v]):
            offset = 5 if val >= 0 else -15
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{val:.1f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontweight="bold",
                fontsize=11,
            )
        ax.set_title(label, fontweight="bold", fontsize=12)
        ax.axhline(y=0, color="grey", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("Value")

    fig.suptitle(
        "Baseline Evaluation: PPO vs DQN (100 Episodes, No Detection)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    path = PLOTS / "baseline_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# -----------------------------------------------------------------------
# 3. Stealth vs Baseline -- Side-by-Side
# -----------------------------------------------------------------------


def fig_stealth_vs_baseline():
    base = _load_json("eval_baseline.json")
    stlth = _load_json("eval_stealth.json")

    labels = ["PPO Baseline", "PPO Stealth", "DQN Baseline", "DQN Stealth"]
    rewards = [
        base["ppo"]["mean_reward"],
        stlth["ppo"]["mean_reward"],
        base["dqn"]["mean_reward"],
        stlth["dqn"]["mean_reward"],
    ]
    steps = [
        base["ppo"]["mean_steps"],
        stlth["ppo"]["mean_steps"],
        base["dqn"]["mean_steps"],
        stlth["dqn"]["mean_steps"],
    ]
    catch = [
        base["ppo"]["catch_rate"],
        stlth["ppo"]["catch_rate"],
        base["dqn"]["catch_rate"],
        stlth["dqn"]["catch_rate"],
    ]
    colors = [BLUE, PURPLE, ORANGE, PURPLE]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, data_vals, title, ylabel in [
        (axes[0], rewards, "Mean Reward", "Reward"),
        (axes[1], steps, "Mean Steps", "Steps"),
        (axes[2], catch, "Catch Rate", "Rate"),
    ]:
        bars = ax.bar(labels, data_vals, color=colors, edgecolor="white", width=0.6)
        for bar, val in zip(bars, data_vals):
            y_pos = bar.get_height()
            offset = max(abs(v) for v in data_vals) * 0.02
            if val < 0:
                offset = -offset
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos + offset,
                f"{val:.1f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontweight="bold",
                fontsize=9,
            )
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.axhline(y=0, color="grey", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=25, labelsize=8)

    fig.suptitle(
        "Baseline vs Stealth Evaluation (100 Episodes Each)", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    path = PLOTS / "stealth_vs_baseline.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# -----------------------------------------------------------------------
# 4. Reward Decomposition
# -----------------------------------------------------------------------


def fig_reward_decomposition():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PPO Baseline: -100 = -500 (step cost) + 400 (2 hosts)
    ax = axes[0]
    components = [
        "Step Cost\n(500 x -1)",
        "Host (3,0)\nLLM API +200",
        "Host (3,1)\nVector DB +200",
        "Net Reward",
    ]
    values = [-500, 200, 200, -100]
    colors_bar = [RED, GREEN, GREEN, BLUE]
    bars = ax.bar(components, values, color=colors_bar, edgecolor="white", width=0.6)
    for bar, val in zip(bars, values):
        y = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y + (10 if val >= 0 else -20),
            f"{val:+d}",
            ha="center",
            fontweight="bold",
            fontsize=11,
        )
    ax.set_title("PPO Baseline Reward Decomposition", fontweight="bold")
    ax.axhline(y=0, color="grey", linewidth=0.8)
    ax.set_ylabel("Reward")
    ax.grid(axis="y", alpha=0.3)

    # Stealth: -109.9 = -9 (steps) + -0.9 (detection) + -100 (caught)
    ax = axes[1]
    components = [
        "Step Cost\n(9 x -1)",
        "Detection\n(9 x -0.1)",
        "Caught Penalty\n(-100)",
        "Net Reward",
    ]
    values = [-9, -0.9, -100, -109.9]
    colors_bar = [ORANGE, PURPLE, RED, GREY]
    bars = ax.bar(components, values, color=colors_bar, edgecolor="white", width=0.6)
    for bar, val in zip(bars, values):
        y = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y - 5,
            f"{val:.1f}",
            ha="center",
            va="top",
            fontweight="bold",
            fontsize=11,
            color="white",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")],
        )
    ax.set_title("Stealth Mode Reward Decomposition", fontweight="bold")
    ax.axhline(y=0, color="grey", linewidth=0.8)
    ax.set_ylabel("Reward")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Reward Decomposition Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = PLOTS / "reward_decomposition.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# -----------------------------------------------------------------------
# 5. Attack Path Flow (PPO Baseline)
# -----------------------------------------------------------------------


def fig_attack_path_flow():
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 5)
    ax.axis("off")

    stages = [
        {"x": 0, "y": 2.5, "label": "INTERNET\n(Attacker)", "color": "#B71C1C", "icon": "ATK"},
        {"x": 1.3, "y": 2.5, "label": "DMZ\nSubnet 1", "color": "#1565C0", "icon": "DMZ"},
        {"x": 2.6, "y": 2.5, "label": "Corp LAN\nSubnet 2", "color": "#2E7D32", "icon": "LAN"},
        {
            "x": 4.0,
            "y": 3.5,
            "label": "LLM API (3,0)\nVALUE: 200",
            "color": "#E65100",
            "icon": "AI",
        },
        {
            "x": 4.0,
            "y": 1.5,
            "label": "Vector DB (3,1)\nVALUE: 200",
            "color": "#E65100",
            "icon": "AI",
        },
        {"x": 5.5, "y": 3.5, "label": "Model Repo (3,2)\nVALUE: 150", "color": GREY, "icon": "X"},
        {"x": 5.5, "y": 1.5, "label": "Data Lake (4,0)\nVALUE: 300", "color": GREY, "icon": "X"},
    ]

    for s in stages:
        circ = plt.Circle(
            (s["x"], s["y"]),
            0.4,
            facecolor=s["color"],
            alpha=0.2,
            edgecolor=s["color"],
            linewidth=2,
            zorder=2,
        )
        ax.add_patch(circ)
        ax.text(
            s["x"],
            s["y"],
            s["icon"],
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=s["color"],
            zorder=3,
        )
        ax.text(
            s["x"],
            s["y"] - 0.6,
            s["label"],
            ha="center",
            va="top",
            fontsize=8,
            color=DARK,
            zorder=3,
            linespacing=1.3,
        )

    arrows = [
        (0, 1.3, 2.5, 2.5, "e_http\n(0.7)", GREEN),
        (1.3, 2.6, 2.5, 2.5, "e_ssh\n(0.8)", GREEN),
        (2.6, 4.0, 2.5, 3.5, "e_http\n(0.7)", GREEN),
        (2.6, 4.0, 2.5, 1.5, "e_http\n(0.7)", GREEN),
        (4.0, 5.5, 3.5, 3.5, "SSH\nFW block", RED),
        (4.0, 5.5, 1.5, 1.5, "SSH\nFW block", RED),
    ]

    for x1, x2, y1, y2, label, color in arrows:
        style = "->" if color == GREEN else "-|>"
        ax.annotate(
            "",
            xy=(x2 - 0.4, y2),
            xytext=(x1 + 0.4, y1),
            arrowprops=dict(
                arrowstyle=style,
                color=color,
                lw=2.5 if color == GREEN else 1.5,
                linestyle="-" if color == GREEN else "--",
            ),
            zorder=1,
        )
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2 + 0.25
        ax.text(
            mid_x,
            mid_y,
            label,
            fontsize=7,
            ha="center",
            va="bottom",
            color=color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=color, alpha=0.85),
            zorder=4,
        )

    legend_elements = [
        mpatches.Patch(facecolor=GREEN, alpha=0.5, label="Successful exploit path"),
        mpatches.Patch(facecolor=RED, alpha=0.5, label="Blocked by firewall"),
        mpatches.Patch(facecolor="#E65100", alpha=0.5, label="Compromised (AI Infra)"),
        mpatches.Patch(facecolor=GREY, alpha=0.5, label="Not reached"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)

    ax.set_title(
        "PPO Agent Learned Attack Path (Baseline, 500k Steps)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    fig.tight_layout()
    path = PLOTS / "attack_path_flow.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# -----------------------------------------------------------------------
# 6. Detection Sensitivity (Simulated Sweep)
# -----------------------------------------------------------------------


def fig_detection_sensitivity():
    """
    Simulate how different detection thresholds affect catch timing.
    With detection_cost = 0.1 per step, the agent is caught at step = threshold / 0.1.
    The PPO attack needs ~15 steps minimum to reach AI infra.
    """
    thresholds = np.arange(0.1, 2.1, 0.1)
    detection_cost = 0.1
    min_steps_to_ai = 15

    caught_step = thresholds / detection_cost
    ppo_reaches = caught_step >= min_steps_to_ai

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.fill_between(
        thresholds,
        0,
        caught_step,
        where=~ppo_reaches,
        color=GREEN,
        alpha=0.15,
        label="Attacker caught before reaching AI",
    )
    ax1.fill_between(
        thresholds,
        0,
        caught_step,
        where=ppo_reaches,
        color=RED,
        alpha=0.15,
        label="Attacker reaches AI before detection",
    )

    ax1.plot(
        thresholds,
        caught_step,
        "o-",
        color=BLUE,
        linewidth=2,
        markersize=5,
        label="Steps until caught",
    )
    ax1.axhline(
        y=min_steps_to_ai,
        color=RED,
        linestyle="--",
        linewidth=1.5,
        label=f"Min steps to AI infra ({min_steps_to_ai})",
    )

    current_thresh = 0.8
    current_steps = current_thresh / detection_cost
    ax1.axvline(x=current_thresh, color=PURPLE, linestyle=":", linewidth=1.5)
    ax1.annotate(
        f"Current: threshold={current_thresh}\ncaught at step {int(current_steps)}",
        xy=(current_thresh, current_steps),
        xytext=(current_thresh + 0.3, current_steps + 3),
        fontsize=9,
        fontweight="bold",
        color=PURPLE,
        arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=PURPLE, alpha=0.9),
    )

    ax1.set_xlabel("Detection Threshold", fontsize=11)
    ax1.set_ylabel("Steps Until Caught", fontsize=11)
    ax1.set_title(
        "Detection Sensitivity: How Monitoring Aggressiveness Affects Security",
        fontsize=12,
        fontweight="bold",
    )
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0.05, 2.05)
    ax1.set_ylim(0, 25)

    fig.tight_layout()
    path = PLOTS / "detection_sensitivity.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# -----------------------------------------------------------------------
# 7. MITRE ATT&CK Heat Map
# -----------------------------------------------------------------------


def fig_mitre_heatmap():
    tactics = [
        "Reconnaissance",
        "Initial Access",
        "Lateral Movement",
        "Privilege\nEscalation",
        "Collection",
    ]
    techniques = [
        "T1046 Network\nService Scanning",
        "T1082 System Info\nDiscovery",
        "T1190 Exploit\nPublic-Facing App",
        "T1021.004 SSH",
        "T1021.002 SMB",
        "T1021.001 RDP",
        "T1068 Exploitation\nfor Priv Esc",
    ]

    mapping = np.array(
        [
            # Recon  InitAcc  LatMov  PrivEsc  Collect
            [1.0, 0.0, 0.0, 0.0, 0.0],  # T1046
            [0.8, 0.0, 0.0, 0.0, 0.0],  # T1082
            [0.0, 0.9, 0.0, 0.0, 0.0],  # T1190
            [0.0, 0.0, 1.0, 0.0, 0.0],  # T1021.004
            [0.0, 0.0, 0.7, 0.0, 0.0],  # T1021.002
            [0.0, 0.0, 0.5, 0.0, 0.0],  # T1021.001
            [0.0, 0.0, 0.0, 0.8, 0.0],  # T1068
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(mapping, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(tactics)))
    ax.set_xticklabels(tactics, fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(techniques)))
    ax.set_yticklabels(techniques, fontsize=8)

    for i in range(len(techniques)):
        for j in range(len(tactics)):
            val = mapping[i, j]
            if val > 0:
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="white" if val > 0.5 else "black",
                )

    ax.set_title(
        "MITRE ATT&CK Coverage -- RL Agent Action Mapping", fontsize=12, fontweight="bold", pad=15
    )
    ax.set_xlabel("ATT&CK Tactic", fontsize=10)
    ax.set_ylabel("ATT&CK Technique", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Agent Utilization Frequency", fontsize=9)

    fig.tight_layout()
    path = PLOTS / "mitre_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# -----------------------------------------------------------------------
# 8. System Architecture Diagram
# -----------------------------------------------------------------------


def fig_architecture():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def draw_box(x, y, w, h, label, color, sublabel=None, fontsize=9):
        rect = plt.Rectangle(
            (x, y), w, h, facecolor=color, alpha=0.2, edgecolor=color, linewidth=2, zorder=2
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2 + (0.12 if sublabel else 0),
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=color,
            zorder=3,
        )
        if sublabel:
            ax.text(
                x + w / 2,
                y + h / 2 - 0.2,
                sublabel,
                ha="center",
                va="center",
                fontsize=7,
                color=DARK,
                zorder=3,
            )

    # Environment layer
    draw_box(
        0.5,
        0.3,
        5,
        2.5,
        "NASim 0.12 Environment",
        "#1565C0",
        "Gymnasium API | 288-dim obs | 110 actions",
    )

    # Wrapper stack
    draw_box(6, 0.3, 4, 0.7, "IntActionWrapper", TEAL, "np.int64 -> int")
    draw_box(6, 1.1, 4, 0.7, "ActionMaskWrapper", TEAL, "Valid action masking")
    draw_box(6, 1.9, 4, 0.7, "DenseRewardWrapper", TEAL, "+3.0 bonus on progress")

    # Agents
    draw_box(
        0.5,
        3.5,
        4.5,
        2,
        "MaskablePPO",
        BLUE,
        "sb3-contrib | [256,256] MLP\nent_coef=0.05 | lr=3e-4",
    )
    draw_box(5.5, 3.5, 4.5, 2, "Masked DQN", ORANGE, "SB3 DQN + Q-mask\n[256,256] MLP | lr=1e-4")

    # Training
    draw_box(0.5, 6.2, 4.5, 1.5, "Training Loop", GREEN, "500k steps | T4 GPU\nMaskedEvalCallback")
    draw_box(
        5.5,
        6.2,
        4.5,
        1.5,
        "Stealth Training",
        PURPLE,
        "StealthWrapper\ndet_threshold=0.8 | penalty=-100",
    )

    # Evaluation & Analysis
    draw_box(
        10.8,
        0.3,
        4.7,
        2.5,
        "Evaluation",
        "#B71C1C",
        "100 episodes | stochastic\nBaseline + Stealth modes",
    )
    draw_box(
        10.8,
        3.5,
        4.7,
        2,
        "Analysis Pipeline",
        "#E65100",
        "MITRE Mapping | What-If\nTopology Viz | Pentest Report",
    )
    draw_box(
        10.8,
        6.2,
        4.7,
        1.5,
        "Output Artifacts",
        DARK,
        "Plots | Reports | Models\neval_*.json | train_meta.json",
    )

    # Scenario
    draw_box(
        0.5,
        8.3,
        15,
        1.2,
        "Custom AI-Infrastructure Scenario",
        "#880E4F",
        "5 subnets | 11 hosts | 4 sensitive targets | 4 exploits | 2 priv-esc | Firewalls",
    )

    # Arrows (simplified connections)
    arrow_kw = dict(arrowstyle="->", color=DARK, lw=1.5)
    ax.annotate("", xy=(3, 3.5), xytext=(3, 2.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(7.75, 3.5), xytext=(7.75, 2.6), arrowprops=arrow_kw)
    ax.annotate("", xy=(3, 6.2), xytext=(3, 5.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(7.75, 6.2), xytext=(7.75, 5.5), arrowprops=arrow_kw)
    ax.annotate(
        "", xy=(8, 2.8), xytext=(5.5, 2.8), arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5)
    )
    ax.annotate(
        "", xy=(6, 1.5), xytext=(5.5, 1.5), arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5)
    )
    ax.annotate("", xy=(10.8, 1.5), xytext=(10, 1.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(10.8, 4.5), xytext=(10, 4.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(13.1, 6.2), xytext=(13.1, 5.5), arrowprops=arrow_kw)
    ax.annotate(
        "", xy=(8, 8.3), xytext=(8, 7.7), arrowprops=dict(arrowstyle="->", color="#880E4F", lw=1.5)
    )

    ax.set_title(
        "RL Attack Path Simulation -- System Architecture", fontsize=14, fontweight="bold", pad=20
    )

    fig.tight_layout()
    path = PLOTS / "system_architecture.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# -----------------------------------------------------------------------
# 9. Training Summary Dashboard
# -----------------------------------------------------------------------


def fig_training_summary():
    meta = {
        "PPO Baseline": _load_json("ppo_baseline/train_meta.json"),
        "DQN Baseline": _load_json("dqn_baseline/train_meta.json"),
        "PPO Stealth": _load_json("ppo_stealth/train_meta.json"),
        "DQN Stealth": _load_json("dqn_stealth/train_meta.json"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = list(meta.keys())
    times = [meta[n]["wall_time_seconds"] / 60.0 for n in names]
    colors_bar = [BLUE, ORANGE, PURPLE, PURPLE]

    ax = axes[0]
    bars = ax.bar(names, times, color=colors_bar, edgecolor="white", width=0.5)
    for bar, val in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f} min",
            ha="center",
            fontweight="bold",
            fontsize=10,
        )
    ax.set_title("Training Wall Time (T4 GPU)", fontweight="bold")
    ax.set_ylabel("Minutes")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)

    # Hyperparameter comparison table
    ax = axes[1]
    ax.axis("off")
    table_data = [
        ["Parameter", "PPO", "DQN"],
        ["Algorithm", "MaskablePPO", "DQN + Q-mask"],
        ["Learning Rate", "3e-4", "1e-4"],
        ["Batch Size", "64", "64"],
        ["Network", "[256, 256]", "[256, 256]"],
        ["Gamma", "0.99", "0.99"],
        ["Entropy / Exploration", "ent=0.05", "eps: 1.0->0.05"],
        ["Buffer", "On-policy (2048)", "100k replay"],
        ["Total Timesteps", "500,000", "500,000"],
    ]
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(DARK)
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 0:
            cell.set_text_props(fontweight="bold")
    ax.set_title("Hyperparameter Comparison", fontweight="bold", pad=20)

    fig.suptitle("Training Configuration Summary", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = PLOTS / "training_summary.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# -----------------------------------------------------------------------
# 10. Security Posture Scorecard
# -----------------------------------------------------------------------


def fig_security_scorecard():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")

    findings = [
        ["#", "Finding", "Severity", "Status", "Evidence"],
        [
            "1",
            "HTTP access from Corp LAN\nto AI Infrastructure",
            "CRITICAL",
            "EXPLOITED",
            "PPO reaches (3,0) & (3,1)\nvia HTTP in every episode",
        ],
        [
            "2",
            "DMZ hosts are pivot points\nfor all attack paths",
            "HIGH",
            "EXPLOITED",
            "100% of successful attacks\ntransit through Subnet 1",
        ],
        [
            "3",
            "SSH-only firewall between\nAI Infra and Data Lake",
            "LOW",
            "EFFECTIVE",
            "Agent never reaches (3,2)\nor (4,0) in 500k steps",
        ],
        [
            "4",
            "Detection threshold = 0.8\nwith cost = 0.1/step",
            "INFO",
            "EFFECTIVE",
            "100% catch rate in stealth\nmode (caught at step 9)",
        ],
        [
            "5",
            "Windows hosts in Corp LAN\nSMB/RDP exposed",
            "MEDIUM",
            "NOT TESTED",
            "DQN did not learn to\nexploit these paths",
        ],
    ]

    severity_colors = {
        "CRITICAL": "#B71C1C",
        "HIGH": "#E65100",
        "MEDIUM": "#F9A825",
        "LOW": GREEN,
        "INFO": BLUE,
    }
    status_colors = {
        "EXPLOITED": RED,
        "EFFECTIVE": GREEN,
        "NOT TESTED": GREY,
    }

    table = ax.table(cellText=findings, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#E0E0E0")
        if row == 0:
            cell.set_facecolor(DARK)
            cell.set_text_props(color="white", fontweight="bold", fontsize=9)
        else:
            if col == 2:
                sev = findings[row][2]
                cell.set_text_props(fontweight="bold", color=severity_colors.get(sev, DARK))
            elif col == 3:
                st = findings[row][3]
                cell.set_text_props(fontweight="bold", color=status_colors.get(st, DARK))

    ax.set_title(
        "Security Posture Scorecard -- RL Assessment Findings",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )

    fig.tight_layout()
    path = PLOTS / "security_scorecard.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def main():
    print("Generating all figures...\n")
    fig_topology()
    fig_baseline_comparison()
    fig_stealth_vs_baseline()
    fig_reward_decomposition()
    fig_attack_path_flow()
    fig_detection_sensitivity()
    fig_mitre_heatmap()
    fig_architecture()
    fig_training_summary()
    fig_security_scorecard()
    print(f"\nAll figures saved to {PLOTS}/")


if __name__ == "__main__":
    main()
