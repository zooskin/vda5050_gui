#!/usr/bin/env python3
"""Generate sample VDA5050 JSONL logs for testing the GUI.

Creates a realistic sequence of order/state/connection messages
that simulate multiple AGVs navigating through a set of nodes.

Usage:
    python sample_log_generator.py [output.jsonl]
"""

import json
import math
import sys
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# Node Layout – a simple warehouse-like grid
# ---------------------------------------------------------------------------
# Layout (5m spacing):
#
#   N1 --- N2 --- N3 --- N4
#          |             |
#          N5 --- N6 --- N7
#                 |
#                 N8 --- N9

NODES = {
    "N1": (0.0, 10.0),
    "N2": (5.0, 10.0),
    "N3": (10.0, 10.0),
    "N4": (15.0, 10.0),
    "N5": (5.0, 5.0),
    "N6": (10.0, 5.0),
    "N7": (15.0, 5.0),
    "N8": (10.0, 0.0),
    "N9": (15.0, 0.0),
}

# Per-AGV settings
AGVS = {
    "AGV-001": {"manufacturer": "RobotCompany", "serial": "AGV-001"},
    "AGV-002": {"manufacturer": "RobotCompany", "serial": "AGV-002"},
}

VERSION = "2.0.0"
MAP_ID = "warehouse_floor1"

_header_counters = {}  # {(agv_id, topic): count}


def _header(agv_id: str, topic: str, ts: datetime) -> dict:
    key = (agv_id, topic)
    _header_counters[key] = _header_counters.get(key, 0) + 1
    agv = AGVS[agv_id]
    return {
        "headerId": _header_counters[key],
        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z",
        "version": VERSION,
        "manufacturer": agv["manufacturer"],
        "serialNumber": agv["serial"],
    }


def _node(node_id: str, seq_id: int, released: bool, actions=None) -> dict:
    x, y = NODES[node_id]
    return {
        "nodeId": node_id,
        "sequenceId": seq_id,
        "released": released,
        "nodePosition": {
            "x": x,
            "y": y,
            "mapId": MAP_ID,
            "allowedDeviationXY": 0.5,
        },
        "actions": actions or [],
    }


def _edge(edge_id: str, seq_id: int, start: str, end: str, released: bool) -> dict:
    return {
        "edgeId": edge_id,
        "sequenceId": seq_id,
        "released": released,
        "startNodeId": start,
        "endNodeId": end,
        "maxSpeed": 1.5,
        "actions": [],
    }


def _action(action_type: str, action_id: str, blocking: str = "HARD", params=None) -> dict:
    return {
        "actionType": action_type,
        "actionId": action_id,
        "blockingType": blocking,
        "actionParameters": params or [],
    }


def _action_state(action_type: str, action_id: str, status: str) -> dict:
    return {
        "actionId": action_id,
        "actionType": action_type,
        "actionStatus": status,
    }


def _node_state(node_id: str, seq_id: int, released: bool) -> dict:
    return {"nodeId": node_id, "sequenceId": seq_id, "released": released}


def _edge_state(edge_id: str, seq_id: int, released: bool) -> dict:
    return {"edgeId": edge_id, "sequenceId": seq_id, "released": released}


def _state_msg(
    agv_id: str,
    ts: datetime,
    order_id: str,
    order_update_id: int,
    last_node_id: str,
    last_node_seq: int,
    x: float,
    y: float,
    theta: float,
    driving: bool,
    node_states: list,
    edge_states: list,
    action_states: list = None,
    errors: list = None,
    battery: float = 85.0,
    new_base_request: bool = False,
) -> dict:
    h = _header(agv_id, "state", ts)
    return {
        **h,
        "orderId": order_id,
        "orderUpdateId": order_update_id,
        "lastNodeId": last_node_id,
        "lastNodeSequenceId": last_node_seq,
        "driving": driving,
        "newBaseRequest": new_base_request,
        "operatingMode": "AUTOMATIC",
        "paused": False,
        "nodeStates": node_states,
        "edgeStates": edge_states,
        "actionStates": action_states or [],
        "agvPosition": {
            "x": x,
            "y": y,
            "theta": theta,
            "mapId": MAP_ID,
            "positionInitialized": True,
            "localizationScore": 0.95,
        },
        "velocity": {"vx": 0.5 if driving else 0.0, "vy": 0.0, "omega": 0.0},
        "batteryState": {"batteryCharge": battery, "charging": False},
        "safetyState": {"eStop": "NONE", "fieldViolation": False},
        "distanceSinceLastNode": 0.0,
        "loads": [],
        "errors": errors or [],
        "information": [],
    }


def _connection_msg(agv_id: str, ts: datetime, state: str) -> dict:
    h = _header(agv_id, "connection", ts)
    return {**h, "connectionState": state}


def _order_msg(agv_id: str, ts: datetime, order_id: str, update_id: int, nodes: list, edges: list) -> dict:
    h = _header(agv_id, "order", ts)
    return {
        **h,
        "orderId": order_id,
        "orderUpdateId": update_id,
        "nodes": nodes,
        "edges": edges,
    }


def _lerp(a, b, t):
    return a + (b - a) * t


def _angle_to(x1, y1, x2, y2):
    return math.atan2(y2 - y1, x2 - x1)


def _write_entry(f, ts: datetime, topic: str, data: dict):
    line = json.dumps(
        {
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z",
            "topic": topic,
            "data": data,
        },
        ensure_ascii=False,
    )
    f.write(line + "\n")


# ---------------------------------------------------------------------------
# Driving helper: generate state messages while moving between two nodes
# ---------------------------------------------------------------------------
def _drive_segment(
    entries, agv_id, ts, order_id, order_update_id,
    from_node, to_node, from_seq, to_seq,
    remaining_node_states, remaining_edge_states,
    steps=5, battery_start=90.0, action_states=None,
    new_base_on_arrival=False,
):
    """Generate driving states from from_node to to_node. Returns updated ts."""
    theta = _angle_to(*NODES[from_node], *NODES[to_node])
    for i in range(1, steps + 1):
        frac = i / steps
        x = _lerp(NODES[from_node][0], NODES[to_node][0], frac)
        y = _lerp(NODES[from_node][1], NODES[to_node][1], frac)
        driving = i < steps
        last_node = from_node if i < steps else to_node
        last_seq = from_seq if i < steps else to_seq

        ns = list(remaining_node_states)
        es = list(remaining_edge_states)

        if i == steps:
            # Remove arrived node/edge from remaining states
            ns = [s for s in ns if s["sequenceId"] != to_seq]
            es = [s for s in es if s["sequenceId"] != (to_seq - 1)]

        entries.append((
            ts, "state",
            _state_msg(
                agv_id, ts, order_id, order_update_id,
                last_node, last_seq, x, y, theta,
                driving=driving,
                node_states=ns, edge_states=es,
                action_states=action_states,
                battery=battery_start - i * 0.2,
                new_base_request=(new_base_on_arrival and i == steps),
            ),
        ))
        ts += timedelta(seconds=1)
    return ts, theta


def _generate_agv001(t: datetime):
    """AGV-001 path: N1 → N2 → N3 → N4 → N7, then new order N7 → N6 → N8 → N9 (pick)."""
    agv = "AGV-001"
    entries = []
    steps = 5

    # Connection
    entries.append((t, "connection", _connection_msg(agv, t, "ONLINE")))
    t += timedelta(milliseconds=500)

    # Initial idle state at N1
    entries.append((
        t, "state",
        _state_msg(agv, t, "", 0, "", 0, *NODES["N1"], 0.0,
                   driving=False, node_states=[], edge_states=[], battery=92.0),
    ))
    t += timedelta(seconds=1)

    # --- Order 1: N1 → N2 → N3 (base), N4 (horizon) ---
    oid = "order_001"
    order_nodes = [
        _node("N1", 0, True), _node("N2", 2, True),
        _node("N3", 4, True), _node("N4", 6, False),
    ]
    order_edges = [
        _edge("E_N1_N2", 1, "N1", "N2", True),
        _edge("E_N2_N3", 3, "N2", "N3", True),
        _edge("E_N3_N4", 5, "N3", "N4", False),
    ]
    entries.append((t, "order", _order_msg(agv, t, oid, 0, order_nodes, order_edges)))
    t += timedelta(milliseconds=500)

    # State: received order, at N1
    theta = _angle_to(*NODES["N1"], *NODES["N2"])
    entries.append((
        t, "state",
        _state_msg(agv, t, oid, 0, "N1", 0, *NODES["N1"], theta,
                   driving=False,
                   node_states=[_node_state("N2", 2, True), _node_state("N3", 4, True), _node_state("N4", 6, False)],
                   edge_states=[_edge_state("E_N1_N2", 1, True), _edge_state("E_N2_N3", 3, True), _edge_state("E_N3_N4", 5, False)],
                   battery=91.5),
    ))
    t += timedelta(seconds=1)

    # Drive N1 → N2
    t, theta = _drive_segment(
        entries, agv, t, oid, 0, "N1", "N2", 0, 2,
        [_node_state("N2", 2, True), _node_state("N3", 4, True), _node_state("N4", 6, False)],
        [_edge_state("E_N1_N2", 1, True), _edge_state("E_N2_N3", 3, True), _edge_state("E_N3_N4", 5, False)],
        battery_start=91.5,
    )

    # Drive N2 → N3
    t, theta = _drive_segment(
        entries, agv, t, oid, 0, "N2", "N3", 2, 4,
        [_node_state("N3", 4, True), _node_state("N4", 6, False)],
        [_edge_state("E_N2_N3", 3, True), _edge_state("E_N3_N4", 5, False)],
        battery_start=90.5, new_base_on_arrival=True,
    )

    # --- Order update 1: release N4, add N7 horizon ---
    t += timedelta(seconds=1)
    entries.append((t, "order", _order_msg(agv, t, oid, 1,
        [_node("N3", 4, True), _node("N4", 6, True), _node("N7", 8, False)],
        [_edge("E_N3_N4", 5, "N3", "N4", True), _edge("E_N4_N7", 7, "N4", "N7", False)],
    )))
    t += timedelta(milliseconds=500)

    entries.append((
        t, "state",
        _state_msg(agv, t, oid, 1, "N3", 4, *NODES["N3"], theta,
                   driving=False,
                   node_states=[_node_state("N4", 6, True), _node_state("N7", 8, False)],
                   edge_states=[_edge_state("E_N3_N4", 5, True), _edge_state("E_N4_N7", 7, False)],
                   battery=89.5),
    ))
    t += timedelta(seconds=1)

    # Drive N3 → N4
    t, theta = _drive_segment(
        entries, agv, t, oid, 1, "N3", "N4", 4, 6,
        [_node_state("N4", 6, True), _node_state("N7", 8, False)],
        [_edge_state("E_N3_N4", 5, True), _edge_state("E_N4_N7", 7, False)],
        battery_start=89.0, new_base_on_arrival=True,
    )

    # --- Order update 2: release N7 ---
    t += timedelta(seconds=1)
    entries.append((t, "order", _order_msg(agv, t, oid, 2,
        [_node("N4", 6, True), _node("N7", 8, True)],
        [_edge("E_N4_N7", 7, "N4", "N7", True)],
    )))
    t += timedelta(milliseconds=500)

    # Drive N4 → N7
    t, theta = _drive_segment(
        entries, agv, t, oid, 2, "N4", "N7", 6, 8,
        [_node_state("N7", 8, True)],
        [_edge_state("E_N4_N7", 7, True)],
        battery_start=88.0,
    )

    # --- Order 2: N7 → N6 → N8 → N9 with pick at N9 ---
    t += timedelta(seconds=2)
    oid2 = "order_002"
    pick = _action("pick", "act_pick_001", "HARD")
    entries.append((t, "order", _order_msg(agv, t, oid2, 0,
        [_node("N7", 0, True), _node("N6", 2, True), _node("N8", 4, True), _node("N9", 6, True, actions=[pick])],
        [_edge("E_N7_N6_2", 1, "N7", "N6", True), _edge("E_N6_N8_2", 3, "N6", "N8", True), _edge("E_N8_N9_2", 5, "N8", "N9", True)],
    )))
    t += timedelta(milliseconds=500)

    acts = [_action_state("pick", "act_pick_001", "WAITING")]

    # Drive N7 → N6
    t, theta = _drive_segment(
        entries, agv, t, oid2, 0, "N7", "N6", 0, 2,
        [_node_state("N6", 2, True), _node_state("N8", 4, True), _node_state("N9", 6, True)],
        [_edge_state("E_N7_N6_2", 1, True), _edge_state("E_N6_N8_2", 3, True), _edge_state("E_N8_N9_2", 5, True)],
        battery_start=86.0, action_states=acts,
    )

    # Drive N6 → N8
    t, theta = _drive_segment(
        entries, agv, t, oid2, 0, "N6", "N8", 2, 4,
        [_node_state("N8", 4, True), _node_state("N9", 6, True)],
        [_edge_state("E_N6_N8_2", 3, True), _edge_state("E_N8_N9_2", 5, True)],
        battery_start=85.0, action_states=acts,
    )

    # Drive N8 → N9
    t, theta = _drive_segment(
        entries, agv, t, oid2, 0, "N8", "N9", 4, 6,
        [_node_state("N9", 6, True)],
        [_edge_state("E_N8_N9_2", 5, True)],
        battery_start=84.0, action_states=acts,
    )

    # Pick action execution at N9
    for status in ["INITIALIZING", "RUNNING", "RUNNING", "FINISHED"]:
        entries.append((
            t, "state",
            _state_msg(agv, t, oid2, 0, "N9", 6, *NODES["N9"], theta,
                       driving=False, node_states=[], edge_states=[],
                       action_states=[_action_state("pick", "act_pick_001", status)],
                       battery=83.0),
        ))
        t += timedelta(seconds=2)

    # Connection drop & recovery
    entries.append((t, "connection", _connection_msg(agv, t, "CONNECTIONBROKEN")))
    t += timedelta(seconds=3)
    entries.append((t, "connection", _connection_msg(agv, t, "ONLINE")))
    t += timedelta(seconds=1)

    # Final idle
    entries.append((
        t, "state",
        _state_msg(agv, t, oid2, 0, "N9", 6, *NODES["N9"], theta,
                   driving=False, node_states=[], edge_states=[], battery=82.5),
    ))

    return entries


def _generate_agv002(t: datetime):
    """AGV-002 path: N5 → N2 → N3 → N4, with drop action at N4."""
    agv = "AGV-002"
    entries = []

    # Connection (slightly after AGV-001)
    t += timedelta(milliseconds=200)
    entries.append((t, "connection", _connection_msg(agv, t, "ONLINE")))
    t += timedelta(milliseconds=500)

    # Initial idle at N5
    entries.append((
        t, "state",
        _state_msg(agv, t, "", 0, "", 0, *NODES["N5"], 0.0,
                   driving=False, node_states=[], edge_states=[], battery=78.0),
    ))
    t += timedelta(seconds=2)

    # --- Order: N5 → N2 (base), N3 → N4 (horizon) ---
    oid = "order_101"
    entries.append((t, "order", _order_msg(agv, t, oid, 0,
        [_node("N5", 0, True), _node("N2", 2, True), _node("N3", 4, False), _node("N4", 6, False)],
        [_edge("E_N5_N2", 1, "N5", "N2", True), _edge("E_N2_N3", 3, "N2", "N3", False), _edge("E_N3_N4", 5, "N3", "N4", False)],
    )))
    t += timedelta(milliseconds=500)

    theta = _angle_to(*NODES["N5"], *NODES["N2"])
    entries.append((
        t, "state",
        _state_msg(agv, t, oid, 0, "N5", 0, *NODES["N5"], theta,
                   driving=False,
                   node_states=[_node_state("N2", 2, True), _node_state("N3", 4, False), _node_state("N4", 6, False)],
                   edge_states=[_edge_state("E_N5_N2", 1, True), _edge_state("E_N2_N3", 3, False), _edge_state("E_N3_N4", 5, False)],
                   battery=77.5),
    ))
    t += timedelta(seconds=1)

    # Drive N5 → N2
    t, theta = _drive_segment(
        entries, agv, t, oid, 0, "N5", "N2", 0, 2,
        [_node_state("N2", 2, True), _node_state("N3", 4, False), _node_state("N4", 6, False)],
        [_edge_state("E_N5_N2", 1, True), _edge_state("E_N2_N3", 3, False), _edge_state("E_N3_N4", 5, False)],
        battery_start=77.5, new_base_on_arrival=True,
    )

    # --- Order update: release N3, N4 with drop action ---
    t += timedelta(seconds=1)
    drop = _action("drop", "act_drop_101", "HARD")
    entries.append((t, "order", _order_msg(agv, t, oid, 1,
        [_node("N2", 2, True), _node("N3", 4, True), _node("N4", 6, True, actions=[drop])],
        [_edge("E_N2_N3", 3, "N2", "N3", True), _edge("E_N3_N4", 5, "N3", "N4", True)],
    )))
    t += timedelta(milliseconds=500)

    entries.append((
        t, "state",
        _state_msg(agv, t, oid, 1, "N2", 2, *NODES["N2"], theta,
                   driving=False,
                   node_states=[_node_state("N3", 4, True), _node_state("N4", 6, True)],
                   edge_states=[_edge_state("E_N2_N3", 3, True), _edge_state("E_N3_N4", 5, True)],
                   action_states=[_action_state("drop", "act_drop_101", "WAITING")],
                   battery=76.0),
    ))
    t += timedelta(seconds=1)

    acts = [_action_state("drop", "act_drop_101", "WAITING")]

    # Drive N2 → N3
    t, theta = _drive_segment(
        entries, agv, t, oid, 1, "N2", "N3", 2, 4,
        [_node_state("N3", 4, True), _node_state("N4", 6, True)],
        [_edge_state("E_N2_N3", 3, True), _edge_state("E_N3_N4", 5, True)],
        battery_start=76.0, action_states=acts,
    )

    # Drive N3 → N4
    t, theta = _drive_segment(
        entries, agv, t, oid, 1, "N3", "N4", 4, 6,
        [_node_state("N4", 6, True)],
        [_edge_state("E_N3_N4", 5, True)],
        battery_start=75.0, action_states=acts,
    )

    # Drop action at N4
    for status in ["INITIALIZING", "RUNNING", "FINISHED"]:
        entries.append((
            t, "state",
            _state_msg(agv, t, oid, 1, "N4", 6, *NODES["N4"], theta,
                       driving=False, node_states=[], edge_states=[],
                       action_states=[_action_state("drop", "act_drop_101", status)],
                       battery=74.0),
        ))
        t += timedelta(seconds=2)

    # Brief offline
    entries.append((t, "connection", _connection_msg(agv, t, "OFFLINE")))
    t += timedelta(seconds=5)
    entries.append((t, "connection", _connection_msg(agv, t, "ONLINE")))
    t += timedelta(seconds=1)

    # Final idle at N4
    entries.append((
        t, "state",
        _state_msg(agv, t, oid, 1, "N4", 6, *NODES["N4"], theta,
                   driving=False, node_states=[], edge_states=[], battery=73.5),
    ))

    return entries


def generate(output_path: str):
    t = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

    # Generate entries for both AGVs
    entries_1 = _generate_agv001(t)
    entries_2 = _generate_agv002(t)

    # Merge by timestamp to interleave realistically
    all_entries = entries_1 + entries_2
    all_entries.sort(key=lambda e: e[0])

    with open(output_path, "w", encoding="utf-8") as f:
        for ts, topic, data in all_entries:
            _write_entry(f, ts, topic, data)

    print(f"Generated {len(all_entries)} log entries ({len(entries_1)} AGV-001, {len(entries_2)} AGV-002) -> {output_path}")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "sample_log.jsonl"
    generate(output)
