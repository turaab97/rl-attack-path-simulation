"""
topology_diagram.py
-------------------
Generates a visual network topology diagram of the AI-infrastructure
scenario using matplotlib. No external graph libraries required.

Author: Syed Ali Turab
Course: MMAI 845 -- Reinforcement Learning

The diagram shows:
- Subnet zones with colour coding (grey/blue/yellow/red/darkred)
- Individual hosts with their OS, services, and reward values
- Firewall-allowed connections as directed arrows
- AI infrastructure and Data Lake highlighted as high-value targets
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

# ---------------------------------------------------------------------------
# Layout data — derived from network_config.py
# ---------------------------------------------------------------------------

SUBNETS = {
    0: {"name": "Subnet 0: Internet", "color": "#E0E0E0", "y": 5.0, "hosts": []},
    1: {
        "name": "Subnet 1: DMZ",
        "color": "#BBDEFB",
        "y": 4.0,
        "hosts": [
            {"id": "(1,0)", "label": "Web Server\nLinux | HTTP, SSH\nValue: 0", "x": 1.5},
            {"id": "(1,1)", "label": "Email Gateway\nWindows | SMB, RDP\nValue: 0", "x": 3.5},
            {"id": "(1,2)", "label": "DNS Server\nLinux | SSH, HTTP\nValue: 0", "x": 5.5},
        ],
    },
    2: {
        "name": "Subnet 2: Corporate LAN",
        "color": "#FFF9C4",
        "y": 2.8,
        "hosts": [
            {"id": "(2,0)", "label": "Workstation-A\nWindows | SMB, RDP\nValue: 10", "x": 0.8},
            {"id": "(2,1)", "label": "Dev Server\nLinux | SSH, HTTP\nValue: 10", "x": 2.8},
            {"id": "(2,2)", "label": "Active Directory\nWindows | SMB, RDP\nValue: 20", "x": 4.8},
            {"id": "(2,3)", "label": "Internal Services\nLinux | SSH\nValue: 10", "x": 6.5},
        ],
    },
    3: {
        "name": "Subnet 3: AI Infrastructure",
        "color": "#FFCDD2",
        "y": 1.6,
        "hosts": [
            {"id": "(3,0)", "label": "LLM API Server\nLinux | HTTP, SSH\nValue: 200", "x": 2.0},
            {"id": "(3,1)", "label": "Vector Database\nLinux | HTTP\nValue: 200", "x": 4.2},
            {"id": "(3,2)", "label": "Model Repository\nLinux | SSH\nValue: 150", "x": 6.2},
        ],
    },
    4: {
        "name": "Subnet 4: Data Lake",
        "color": "#B71C1C",
        "y": 0.4,
        "hosts": [
            {"id": "(4,0)", "label": "Training Data Lake\nLinux | SSH\nValue: 300", "x": 3.5},
        ],
    },
}

# Firewall-allowed connections: (source_host, target_host, services)
CONNECTIONS = [
    ("(1,0)", "(2,0)", "SMB"),
    ("(1,1)", "(2,0)", "SMB, RDP"),
    ("(1,2)", "(2,1)", "SSH"),
    ("(2,0)", "(2,2)", "SMB"),
    ("(2,1)", "(2,3)", "SSH"),
    ("(2,1)", "(3,0)", "HTTP"),
    ("(2,3)", "(3,0)", "SSH"),
    ("(3,0)", "(3,1)", "HTTP"),
    ("(3,0)", "(3,2)", "SSH"),
    ("(3,0)", "(4,0)", "SSH"),
    ("(3,2)", "(4,0)", "SSH"),
]


def generate_topology_diagram(
    output_path: str = "results/plots/network_topology.png",
    figsize: tuple[float, float] = (14, 10),
    dpi: int = 150,
) -> None:
    """Generate and save the network topology diagram.

    Parameters
    ----------
    output_path : str
        Where to save the PNG.
    figsize : tuple
        Figure dimensions.
    dpi : int
        Resolution.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.2, 5.8)
    ax.set_aspect("equal")
    ax.axis("off")

    host_positions = {}

    # Draw subnet zones and hosts
    for subnet_id, subnet in SUBNETS.items():
        if not subnet["hosts"]:
            # Attacker entry point
            ax.annotate(
                "ATTACKER\n(Internet)",
                xy=(3.5, subnet["y"]),
                fontsize=11,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor=subnet["color"],
                    edgecolor="#424242",
                    linewidth=1.5,
                ),
            )
            host_positions["attacker"] = (3.5, subnet["y"])
            continue

        # Zone background
        xs = [h["x"] for h in subnet["hosts"]]
        zone_left = min(xs) - 0.9
        zone_right = max(xs) + 0.9
        zone_rect = FancyBboxPatch(
            (zone_left, subnet["y"] - 0.45),
            zone_right - zone_left,
            0.9,
            boxstyle="round,pad=0.1",
            facecolor=subnet["color"],
            edgecolor="#616161",
            linewidth=1.2,
            alpha=0.4,
            zorder=0,
        )
        ax.add_patch(zone_rect)
        ax.text(
            zone_left + 0.1,
            subnet["y"] + 0.35,
            subnet["name"],
            fontsize=8,
            fontweight="bold",
            color="#424242",
            va="top",
        )

        # Host boxes
        for host in subnet["hosts"]:
            is_high_value = subnet_id >= 3
            edge_color = "#B71C1C" if is_high_value else "#1565C0"
            face_color = "#FFF" if subnet_id < 4 else "#FFEBEE"
            lw = 2.0 if is_high_value else 1.0

            ax.annotate(
                host["label"],
                xy=(host["x"], subnet["y"]),
                fontsize=6.5,
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=face_color,
                    edgecolor=edge_color,
                    linewidth=lw,
                ),
            )
            host_positions[host["id"]] = (host["x"], subnet["y"])

    # Draw connections
    for src, tgt, services in CONNECTIONS:
        if src in host_positions and tgt in host_positions:
            src_pos = host_positions[src]
            tgt_pos = host_positions[tgt]
            ax.annotate(
                "",
                xy=tgt_pos,
                xytext=src_pos,
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#78909C",
                    lw=1.0,
                    connectionstyle="arc3,rad=0.1",
                ),
            )
            mid_x = (src_pos[0] + tgt_pos[0]) / 2
            mid_y = (src_pos[1] + tgt_pos[1]) / 2
            ax.text(
                mid_x,
                mid_y,
                services,
                fontsize=5,
                color="#546E7A",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8
                ),
            )

    # Draw attacker arrow to DMZ
    ax.annotate(
        "",
        xy=(3.5, 4.4),
        xytext=(3.5, 4.7),
        arrowprops=dict(arrowstyle="-|>", color="#D32F2F", lw=2.0),
    )

    # Title and legend
    ax.set_title(
        "Enterprise AI Infrastructure -- Network Topology\n" "RL Attack Path Simulation | MMAI 845",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    legend_items = [
        mpatches.Patch(facecolor="#BBDEFB", edgecolor="#616161", label="DMZ (Public-facing)"),
        mpatches.Patch(facecolor="#FFF9C4", edgecolor="#616161", label="Corporate LAN"),
        mpatches.Patch(
            facecolor="#FFCDD2", edgecolor="#616161", label="AI Infrastructure (High Value)"
        ),
        mpatches.Patch(facecolor="#B71C1C", edgecolor="#616161", label="Data Lake (Highest Value)"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
        edgecolor="#BDBDBD",
    )

    plt.tight_layout()
    plt.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[topology] Saved diagram to {out}")


if __name__ == "__main__":
    generate_topology_diagram()
