"""
mitre_mapping.py
----------------
Maps NASim action types to MITRE ATT&CK techniques, bridging the RL
simulation to the industry-standard adversary behaviour framework.

Author: Syed Ali Turab
Course: MMAI 845 -- Reinforcement Learning

This mapping enables security teams to interpret agent behaviour in terms
they already use for threat modelling and incident response. Each NASim
action category corresponds to one or more ATT&CK techniques across the
kill chain.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# MITRE ATT&CK mapping table
# ---------------------------------------------------------------------------
# Reference: https://attack.mitre.org/
#
# NASim actions are abstract (scan, exploit, priv_esc) but map cleanly to
# ATT&CK tactics and techniques. The mapping below uses the most common
# real-world technique for each abstract action.

MITRE_MAPPING = {
    "subnet_scan": {
        "tactic": "Discovery",
        "technique_id": "T1046",
        "technique_name": "Network Service Discovery",
        "description": (
            "The agent scans a subnet to discover reachable hosts and open "
            "ports, equivalent to running nmap against a network segment."
        ),
    },
    "os_scan": {
        "tactic": "Discovery",
        "technique_id": "T1082",
        "technique_name": "System Information Discovery",
        "description": (
            "The agent fingerprints the operating system of a discovered "
            "host to select the correct exploit."
        ),
    },
    "service_scan": {
        "tactic": "Discovery",
        "technique_id": "T1046",
        "technique_name": "Network Service Discovery",
        "description": (
            "The agent enumerates services running on a host to identify "
            "exploitable attack surface."
        ),
    },
    "e_ssh": {
        "tactic": "Lateral Movement",
        "technique_id": "T1021.004",
        "technique_name": "Remote Services: SSH",
        "description": (
            "The agent exploits the SSH service to gain user-level access, "
            "analogous to brute-force or credential-based SSH compromise."
        ),
    },
    "e_http": {
        "tactic": "Initial Access",
        "technique_id": "T1190",
        "technique_name": "Exploit Public-Facing Application",
        "description": (
            "The agent exploits a web application (Apache/HTTP) to gain "
            "initial foothold, similar to exploiting a web server vulnerability."
        ),
    },
    "e_smb": {
        "tactic": "Lateral Movement",
        "technique_id": "T1021.002",
        "technique_name": "Remote Services: SMB/Windows Admin Shares",
        "description": (
            "The agent exploits the SMB protocol to move laterally between "
            "Windows hosts, analogous to EternalBlue or pass-the-hash."
        ),
    },
    "e_rdp": {
        "tactic": "Lateral Movement",
        "technique_id": "T1021.001",
        "technique_name": "Remote Services: Remote Desktop Protocol",
        "description": (
            "The agent exploits RDP to gain access to a Windows host, "
            "similar to BlueKeep or credential-based RDP compromise."
        ),
    },
    "pe_linux": {
        "tactic": "Privilege Escalation",
        "technique_id": "T1068",
        "technique_name": "Exploitation for Privilege Escalation",
        "description": (
            "The agent escalates from user to root on a Linux host by "
            "exploiting a vulnerable process (Apache), analogous to a "
            "local kernel or service exploit."
        ),
    },
    "pe_windows": {
        "tactic": "Privilege Escalation",
        "technique_id": "T1068",
        "technique_name": "Exploitation for Privilege Escalation",
        "description": (
            "The agent escalates privileges on a Windows host by exploiting "
            "a vulnerable SMB daemon process."
        ),
    },
    "noop": {
        "tactic": "N/A",
        "technique_id": "N/A",
        "technique_name": "No Operation",
        "description": "The agent chooses to wait, taking no action this step.",
    },
}

# Ordered kill chain for display
KILL_CHAIN_ORDER = [
    "Reconnaissance",
    "Initial Access",
    "Discovery",
    "Lateral Movement",
    "Privilege Escalation",
    "Impact",
]


def get_mitre_for_action(action_name: str) -> dict[str, str]:
    """Look up the MITRE ATT&CK mapping for a NASim action name.

    Parameters
    ----------
    action_name : str
        The raw action name from NASim (e.g. 'e_ssh', 'subnet_scan').

    Returns
    -------
    dict with keys: tactic, technique_id, technique_name, description.
    Returns an 'Unknown' entry if the action is not in the mapping.
    """
    key = action_name.lower().strip()
    if key in MITRE_MAPPING:
        return MITRE_MAPPING[key]

    # Try partial matching for scan variants
    for map_key, value in MITRE_MAPPING.items():
        if map_key in key or key in map_key:
            return value

    return {
        "tactic": "Unknown",
        "technique_id": "N/A",
        "technique_name": action_name,
        "description": f"No MITRE ATT&CK mapping defined for '{action_name}'.",
    }


def map_path_to_mitre(
    interpreted_path: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Annotate each step of an interpreted attack path with MITRE ATT&CK info.

    Parameters
    ----------
    interpreted_path : list[dict]
        Output of attack_path.interpret_path().

    Returns
    -------
    list[dict]
        Each step dict is extended with 'mitre_tactic', 'mitre_technique_id',
        and 'mitre_technique_name' keys.
    """
    annotated = []
    for step in interpreted_path:
        mitre = get_mitre_for_action(step.get("action_name", ""))
        enriched = dict(step)
        enriched["mitre_tactic"] = mitre["tactic"]
        enriched["mitre_technique_id"] = mitre["technique_id"]
        enriched["mitre_technique_name"] = mitre["technique_name"]
        annotated.append(enriched)
    return annotated


def generate_mitre_summary_table() -> list[dict[str, str]]:
    """Return the full MITRE mapping as a list of dicts for display.

    Returns
    -------
    list[dict]
        One dict per NASim action with keys: nasim_action, tactic,
        technique_id, technique_name, description.
    """
    rows = []
    for action_name, mapping in MITRE_MAPPING.items():
        rows.append(
            {
                "nasim_action": action_name,
                "tactic": mapping["tactic"],
                "technique_id": mapping["technique_id"],
                "technique_name": mapping["technique_name"],
                "description": mapping["description"],
            }
        )
    return rows
