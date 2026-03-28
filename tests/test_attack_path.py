"""
test_attack_path.py
-------------------
Tests for analysis/attack_path.py — action mapping and path interpretation.

Author: Syed Ali Turab
Course: MMAI 845 – Reinforcement Learning
"""

import pytest

from analysis.attack_path import (
    _classify_action,
    build_action_map,
    find_common_pivots,
    interpret_path,
    summarise_path,
)


class TestClassifyAction:
    def test_noop(self):
        assert _classify_action("noop") == "noop"
        assert _classify_action("no-op") == "noop"

    def test_scan(self):
        assert _classify_action("subnet_scan") == "scan"
        assert _classify_action("os_scan") == "scan"
        assert _classify_action("service_scan") == "scan"

    def test_exploit(self):
        assert _classify_action("e_ssh") == "exploit"
        assert _classify_action("e_http") == "exploit"
        assert _classify_action("exploit_rdp") == "exploit"

    def test_priv_esc(self):
        assert _classify_action("pe_linux") == "privilege_escalation"
        assert _classify_action("pe_windows") == "privilege_escalation"
        assert _classify_action("priv_esc_kernel") == "privilege_escalation"

    def test_other(self):
        assert _classify_action("something_unknown") == "other"


class TestBuildActionMap:
    def test_returns_list(self):
        import nasim
        env = nasim.make_benchmark("small-linear")
        action_map = build_action_map(env)
        assert isinstance(action_map, list)
        assert len(action_map) > 0
        env.close()

    def test_entries_have_required_keys(self):
        import nasim
        env = nasim.make_benchmark("small-linear")
        action_map = build_action_map(env)
        for entry in action_map:
            assert "action_idx" in entry
            assert "action_type" in entry
            assert "action_name" in entry
        env.close()


class TestInterpretPath:
    def test_interpret_simple_path(self):
        action_map = [
            {"action_idx": 0, "action_type": "noop", "action_name": "noop",
             "host_name": "none", "subnet_name": "none", "subnet": 0, "host": 0},
            {"action_idx": 1, "action_type": "scan", "action_name": "subnet_scan",
             "host_name": "Web Server", "subnet_name": "DMZ", "subnet": 1, "host": 0},
            {"action_idx": 2, "action_type": "exploit", "action_name": "e_ssh",
             "host_name": "LLM API Server", "subnet_name": "AI Infrastructure",
             "subnet": 3, "host": 0},
        ]

        result = interpret_path([0, 1, 2, 1], action_map)
        assert len(result) == 4
        assert result[0]["step"] == 0
        assert result[0]["action_type"] == "noop"
        assert result[2]["action_type"] == "exploit"
        assert result[2]["host_name"] == "LLM API Server"

    def test_out_of_range_action(self):
        action_map = [
            {"action_idx": 0, "action_type": "noop", "action_name": "noop",
             "host_name": "none", "subnet_name": "none", "subnet": 0, "host": 0},
        ]
        result = interpret_path([0, 999], action_map)
        assert len(result) == 2
        assert result[1]["action_type"] == "unknown"


class TestSummarisePath:
    def test_summary_structure(self):
        interpreted = [
            {"step": 0, "action_type": "scan", "host_name": "Web Server",
             "subnet_name": "DMZ"},
            {"step": 1, "action_type": "exploit", "host_name": "Web Server",
             "subnet_name": "DMZ"},
            {"step": 2, "action_type": "exploit", "host_name": "LLM API Server",
             "subnet_name": "AI Infrastructure"},
        ]
        summary = summarise_path(interpreted)
        assert summary["total_steps"] == 3
        assert "scan" in summary["action_type_counts"]
        assert "exploit" in summary["action_type_counts"]
        assert "Web Server" in summary["hosts_targeted"]
        assert "DMZ" in summary["subnets_visited"]


class TestFindCommonPivots:
    def test_identifies_frequent_hosts(self):
        action_map = [
            {"action_idx": 0, "action_type": "noop", "action_name": "noop",
             "host_name": "none", "subnet_name": "none", "subnet": 0, "host": 0},
            {"action_idx": 1, "action_type": "exploit", "action_name": "e_ssh",
             "host_name": "Web Server", "subnet_name": "DMZ", "subnet": 1, "host": 0},
            {"action_idx": 2, "action_type": "exploit", "action_name": "e_http",
             "host_name": "LLM API Server", "subnet_name": "AI Infrastructure",
             "subnet": 3, "host": 0},
        ]
        paths = [
            [1, 2],
            [1, 2],
            [0, 1],
        ]
        pivots = find_common_pivots(paths, action_map, top_n=3)
        assert isinstance(pivots, list)
        # Web Server appears in all 3 paths (exploit action)
        host_names = [name for name, _ in pivots]
        assert "Web Server" in host_names
