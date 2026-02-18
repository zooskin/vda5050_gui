#!/usr/bin/env python3
"""VDA5050 Log Visualization GUI

Visualizes VDA5050 order/state/connection logs to help developers
inspect robot paths, order update timing, and base/horizon distinctions.
Supports multiple AGVs with per-robot color coding and filtering.

Usage:
    python vda5050_gui.py [logfile.jsonl]
"""

import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from PyQt5.QtCore import (
    QObject,
    QPointF,
    QRectF,
    Qt,
    QTimer,
    pyqtSignal,
    pyqtSlot,
)
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QTransform,
    QWheelEvent,
)
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
COL_HORIZON_NODE = QColor("#E0E0E0")
COL_HORIZON_EDGE = QColor("#BDBDBD")
COL_CONN_ONLINE = QColor("#4CAF50")
COL_CONN_OFFLINE = QColor("#FFC107")
COL_CONN_BROKEN = QColor("#F44336")
COL_ORDER_MARKER = QColor("#FF5722")
COL_BG = QColor("#FAFAFA")
COL_GRID = QColor("#E8E8E8")
COL_NODE_TEXT = QColor("#333333")

# Per-AGV color palette – each AGV gets a distinct theme color
AGV_PALETTE = [
    QColor("#E74C3C"),  # red
    QColor("#2196F3"),  # blue
    QColor("#4CAF50"),  # green
    QColor("#FF9800"),  # orange
    QColor("#9C27B0"),  # purple
    QColor("#00BCD4"),  # cyan
    QColor("#795548"),  # brown
    QColor("#607D8B"),  # blue-grey
    QColor("#E91E63"),  # pink
    QColor("#009688"),  # teal
]

NODE_RADIUS = 8
ROBOT_SIZE = 18
TRAIL_WIDTH = 2

# ---------------------------------------------------------------------------
# LogEntry – parsed JSONL line
# ---------------------------------------------------------------------------
@dataclass
class LogEntry:
    timestamp: str
    topic: str  # "order", "state", "connection"
    data: Dict[str, Any]
    dt: float = 0.0  # epoch seconds for sorting
    agv_id: str = ""

    def __post_init__(self):
        try:
            t = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
            self.dt = t.timestamp()
        except Exception:
            self.dt = 0.0
        if not self.agv_id:
            self.agv_id = self.data.get("serialNumber", "unknown")


# ---------------------------------------------------------------------------
# Snapshot – computed view at a point in time (per AGV)
# ---------------------------------------------------------------------------
@dataclass
class Snapshot:
    order: Optional[Dict[str, Any]] = None
    state: Optional[Dict[str, Any]] = None
    connection: Optional[Dict[str, Any]] = None
    # Derived helpers
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    agv_x: Optional[float] = None
    agv_y: Optional[float] = None
    agv_theta: Optional[float] = None
    last_node_id: Optional[str] = None
    last_node_seq_id: int = -1
    connection_state: str = "OFFLINE"
    order_id: str = ""
    order_update_id: int = 0
    driving: bool = False
    node_states: List[Dict[str, Any]] = field(default_factory=list)
    edge_states: List[Dict[str, Any]] = field(default_factory=list)
    action_states: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    battery_charge: float = 0.0


def _populate_snapshot(snap: Snapshot, order_data, state_data, conn_data):
    """Fill derived fields of a Snapshot from raw message data."""
    snap.order = order_data
    snap.state = state_data
    snap.connection = conn_data

    if order_data:
        snap.nodes = order_data.get("nodes", [])
        snap.edges = order_data.get("edges", [])
        snap.order_id = order_data.get("orderId", "")
        snap.order_update_id = order_data.get("orderUpdateId", 0)

    if state_data:
        pos = state_data.get("agvPosition", {})
        if pos:
            snap.agv_x = pos.get("x")
            snap.agv_y = pos.get("y")
            snap.agv_theta = pos.get("theta", 0.0)
        snap.last_node_id = state_data.get("lastNodeId", "")
        snap.last_node_seq_id = state_data.get("lastNodeSequenceId", -1)
        snap.driving = state_data.get("driving", False)
        snap.node_states = state_data.get("nodeStates", [])
        snap.edge_states = state_data.get("edgeStates", [])
        snap.action_states = state_data.get("actionStates", [])
        snap.errors = state_data.get("errors", [])
        battery = state_data.get("batteryState", {})
        snap.battery_charge = battery.get("batteryCharge", 0.0)

    if conn_data:
        snap.connection_state = conn_data.get("connectionState", "OFFLINE")


# ---------------------------------------------------------------------------
# LogStore – log list management + snapshot computation
# ---------------------------------------------------------------------------
class LogStore:
    def __init__(self):
        self.entries: List[LogEntry] = []
        self._order_change_indices: List[int] = []

    def clear(self):
        self.entries.clear()
        self._order_change_indices.clear()

    def load_file(self, path: str) -> int:
        self.clear()
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    entry = LogEntry(
                        timestamp=obj.get("timestamp", ""),
                        topic=obj.get("topic", ""),
                        data=obj.get("data", {}),
                    )
                    self.entries.append(entry)
                    count += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        self.entries.sort(key=lambda e: e.dt)
        self._compute_order_changes()
        return count

    def append_entry(self, entry: LogEntry):
        self.entries.append(entry)
        idx = len(self.entries) - 1
        if entry.topic == "order":
            self._order_change_indices.append(idx)

    def _compute_order_changes(self):
        self._order_change_indices.clear()
        prev: Dict[str, Tuple[str, int]] = {}  # agv_id -> (orderId, orderUpdateId)
        for i, e in enumerate(self.entries):
            if e.topic == "order":
                current = (e.data.get("orderId", ""), e.data.get("orderUpdateId", 0))
                if prev.get(e.agv_id) != current:
                    self._order_change_indices.append(i)
                    prev[e.agv_id] = current

    @property
    def order_change_indices(self) -> List[int]:
        return self._order_change_indices

    def get_agv_ids(self) -> List[str]:
        """Return sorted unique AGV IDs seen in all entries."""
        ids: Set[str] = set()
        for e in self.entries:
            if e.agv_id:
                ids.add(e.agv_id)
        return sorted(ids)

    def get_snapshots(self, index: int) -> Dict[str, Snapshot]:
        """Compute per-AGV snapshots at entry[index]."""
        if not self.entries or index < 0:
            return {}

        index = min(index, len(self.entries) - 1)

        last_order: Dict[str, Any] = {}
        last_state: Dict[str, Any] = {}
        last_conn: Dict[str, Any] = {}

        for i in range(index + 1):
            e = self.entries[i]
            aid = e.agv_id
            if e.topic == "order":
                last_order[aid] = e.data
            elif e.topic == "state":
                last_state[aid] = e.data
            elif e.topic == "connection":
                last_conn[aid] = e.data

        all_agvs = set(last_order.keys()) | set(last_state.keys()) | set(last_conn.keys())
        result: Dict[str, Snapshot] = {}
        for aid in all_agvs:
            snap = Snapshot()
            _populate_snapshot(snap, last_order.get(aid), last_state.get(aid), last_conn.get(aid))
            result[aid] = snap
        return result

    def get_trail(self, up_to_index: int, agv_id: str, max_points: int = 500) -> List[Tuple[float, float]]:
        """Collect AGV positions from state messages up to index for one AGV."""
        trail: List[Tuple[float, float]] = []
        start = max(0, up_to_index - max_points * 3)
        for i in range(start, min(up_to_index + 1, len(self.entries))):
            e = self.entries[i]
            if e.topic == "state" and e.agv_id == agv_id:
                pos = e.data.get("agvPosition", {})
                x = pos.get("x")
                y = pos.get("y")
                if x is not None and y is not None:
                    if not trail or (trail[-1][0] != x or trail[-1][1] != y):
                        trail.append((x, y))
        if len(trail) > max_points:
            trail = trail[-max_points:]
        return trail


# ---------------------------------------------------------------------------
# MqttLogClient – real-time MQTT listener with Qt signals
# ---------------------------------------------------------------------------
class MqttLogClient(QObject):
    log_received = pyqtSignal(object)  # LogEntry
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._client = None
        self._connected = False
        self._topics: List[Tuple[str, int]] = []

    def connect_to_broker(
        self,
        host: str,
        port: int,
        interface_name: str,
        version: str,
        manufacturer: str,
    ):
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            self.error_occurred.emit("paho-mqtt not installed. Run: pip install paho-mqtt")
            return

        # Use '+' wildcard for serialNumber to receive all AGVs under this manufacturer
        prefix = f"{interface_name}/{version}/{manufacturer}/+"
        self._topics = [
            (f"{prefix}/order", 0),
            (f"{prefix}/state", 0),
            (f"{prefix}/connection", 1),
        ]

        self._client = mqtt.Client()
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

        try:
            self._client.connect_async(host, port, keepalive=60)
            self._client.loop_start()
        except Exception as exc:
            self.error_occurred.emit(f"MQTT connect error: {exc}")

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            for topic, qos in self._topics:
                client.subscribe(topic, qos)
            self.connected.emit()
        else:
            self.error_occurred.emit(f"MQTT connect failed (rc={rc})")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        self.disconnected.emit()

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        parts = msg.topic.rsplit("/", 1)
        topic_type = parts[-1] if parts else "unknown"

        ts = payload.get("timestamp", datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z"))
        entry = LogEntry(timestamp=ts, topic=topic_type, data=payload)
        self.log_received.emit(entry)

    def stop(self):
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected


# ---------------------------------------------------------------------------
# MqttConnectionDialog
# ---------------------------------------------------------------------------
class MqttConnectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MQTT Connection")
        self.setMinimumWidth(400)

        layout = QFormLayout(self)

        self.host_edit = QLineEdit("localhost")
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(1883)

        self.interface_edit = QLineEdit("uagv")
        self.version_edit = QLineEdit("v2")
        self.manufacturer_edit = QLineEdit("RobotCompany")

        layout.addRow("Broker Host:", self.host_edit)
        layout.addRow("Broker Port:", self.port_spin)
        layout.addRow("Interface Name:", self.interface_edit)
        layout.addRow("Version:", self.version_edit)
        layout.addRow("Manufacturer:", self.manufacturer_edit)

        self.preview_label = QLabel()
        self.preview_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addRow("Topic Pattern:", self.preview_label)

        hint_label = QLabel("All AGVs under this manufacturer will be discovered automatically.")
        hint_label.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
        hint_label.setWordWrap(True)
        layout.addRow("", hint_label)

        self._update_preview()
        for edit in (self.interface_edit, self.version_edit, self.manufacturer_edit):
            edit.textChanged.connect(self._update_preview)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _update_preview(self):
        prefix = (
            f"{self.interface_edit.text()}/{self.version_edit.text()}/"
            f"{self.manufacturer_edit.text()}/+")
        self.preview_label.setText(f"{prefix}/[order|state|connection]")

    def get_params(self) -> dict:
        return {
            "host": self.host_edit.text(),
            "port": self.port_spin.value(),
            "interface_name": self.interface_edit.text(),
            "version": self.version_edit.text(),
            "manufacturer": self.manufacturer_edit.text(),
        }


# ---------------------------------------------------------------------------
# MapCanvas – 2D multi-AGV node/edge/robot visualization
# ---------------------------------------------------------------------------
class MapCanvas(QWidget):
    node_clicked = pyqtSignal(str, dict)  # nodeId, node data
    robot_clicked = pyqtSignal(str)  # agv_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        self._snapshots: Dict[str, Snapshot] = {}
        self._trails: Dict[str, List[Tuple[float, float]]] = {}
        self._visible_agvs: Set[str] = set()
        self._selected_agv: str = ""
        self._agv_color_cache: Dict[str, QColor] = {}

        # View transform
        self._scale = 40.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._panning = False
        self._pan_start = QPointF()
        self._pan_offset_start = (0.0, 0.0)

        # Screen position caches for click detection
        self._node_screen_positions: Dict[str, Tuple[float, float, Dict]] = {}
        self._robot_screen_positions: Dict[str, Tuple[float, float]] = {}

    def set_data(
        self,
        snapshots: Dict[str, Snapshot],
        trails: Dict[str, List[Tuple[float, float]]],
        visible_agvs: Set[str],
        selected_agv: str = "",
    ):
        self._snapshots = snapshots
        self._trails = trails
        self._visible_agvs = visible_agvs
        self._selected_agv = selected_agv
        self._node_screen_positions.clear()
        self._robot_screen_positions.clear()
        self._rebuild_color_cache()
        self.update()

    def _rebuild_color_cache(self):
        all_ids = sorted(self._snapshots.keys())
        self._agv_color_cache.clear()
        for i, aid in enumerate(all_ids):
            self._agv_color_cache[aid] = AGV_PALETTE[i % len(AGV_PALETTE)]

    def get_agv_color(self, agv_id: str) -> QColor:
        return self._agv_color_cache.get(agv_id, AGV_PALETTE[0])

    def fit_to_content(self):
        xs: List[float] = []
        ys: List[float] = []
        for aid in self._visible_agvs:
            snap = self._snapshots.get(aid)
            if not snap:
                continue
            for n in snap.nodes:
                pos = n.get("nodePosition", {})
                x, y = pos.get("x"), pos.get("y")
                if x is not None and y is not None:
                    xs.append(x)
                    ys.append(y)
            if snap.agv_x is not None:
                xs.append(snap.agv_x)
                ys.append(snap.agv_y)

        if not xs:
            return

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        w = max_x - min_x
        h = max_y - min_y

        margin = 60
        canvas_w = self.width() - margin * 2
        canvas_h = self.height() - margin * 2

        if w > 0 or h > 0:
            sx = canvas_w / w if w > 0 else canvas_w
            sy = canvas_h / h if h > 0 else canvas_h
            self._scale = min(sx, sy, 200.0)
            self._scale = max(self._scale, 5.0)
        else:
            self._scale = 40.0

        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        self._offset_x = self.width() / 2.0 - cx * self._scale
        self._offset_y = self.height() / 2.0 + cy * self._scale
        self.update()

    def _world_to_screen(self, wx: float, wy: float) -> Tuple[float, float]:
        sx = wx * self._scale + self._offset_x
        sy = -wy * self._scale + self._offset_y
        return sx, sy

    def _screen_to_world(self, sx: float, sy: float) -> Tuple[float, float]:
        wx = (sx - self._offset_x) / self._scale
        wy = -(sy - self._offset_y) / self._scale
        return wx, wy

    @staticmethod
    def _build_node_pos(snap: Snapshot) -> Dict[str, Tuple[float, float]]:
        node_pos: Dict[str, Tuple[float, float]] = {}
        for n in snap.nodes:
            nid = n.get("nodeId", "")
            pos = n.get("nodePosition", {})
            x, y = pos.get("x"), pos.get("y")
            if x is not None and y is not None:
                node_pos[nid] = (x, y)
        return node_pos

    # ---- Paint ----

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), COL_BG)
        self._draw_grid(painter)

        if not self._snapshots:
            painter.setPen(QPen(QColor("#999")))
            painter.setFont(QFont("sans-serif", 14))
            painter.drawText(self.rect(), Qt.AlignCenter, "No data loaded\n\nFile → Open Log (Ctrl+O)")
            painter.end()
            return

        visible_sorted = sorted(self._visible_agvs)
        self._node_screen_positions.clear()
        self._robot_screen_positions.clear()

        # Pass 1: edges
        for aid in visible_sorted:
            snap = self._snapshots.get(aid)
            if not snap or not snap.edges:
                continue
            color = self.get_agv_color(aid)
            node_pos = self._build_node_pos(snap)
            self._draw_edges(painter, snap, node_pos, color)

        # Pass 2: trails
        for aid in visible_sorted:
            trail = self._trails.get(aid, [])
            color = self.get_agv_color(aid)
            self._draw_trail(painter, trail, color)

        # Pass 3: nodes
        for aid in visible_sorted:
            snap = self._snapshots.get(aid)
            if not snap or not snap.nodes:
                continue
            color = self.get_agv_color(aid)
            node_pos = self._build_node_pos(snap)
            self._draw_nodes(painter, snap, node_pos, color)

        # Pass 4: robots (always on top)
        for aid in visible_sorted:
            snap = self._snapshots.get(aid)
            if not snap:
                continue
            color = self.get_agv_color(aid)
            is_selected = (aid == self._selected_agv)
            self._draw_robot(painter, snap, aid, color, is_selected)

        # Legend
        if len(self._agv_color_cache) > 1:
            self._draw_legend(painter)

        painter.end()

    def _draw_grid(self, painter: QPainter):
        pen = QPen(COL_GRID, 1, Qt.DotLine)
        painter.setPen(pen)

        grid_world = 1.0
        if self._scale < 15:
            grid_world = 5.0
        elif self._scale < 8:
            grid_world = 10.0

        w, h = self.width(), self.height()
        wl, wt = self._screen_to_world(0, 0)
        wr, wb = self._screen_to_world(w, h)

        min_wx, max_wx = min(wl, wr), max(wl, wr)
        min_wy, max_wy = min(wt, wb), max(wt, wb)

        x = math.floor(min_wx / grid_world) * grid_world
        while x <= max_wx:
            sx, _ = self._world_to_screen(x, 0)
            painter.drawLine(int(sx), 0, int(sx), h)
            x += grid_world

        y = math.floor(min_wy / grid_world) * grid_world
        while y <= max_wy:
            _, sy = self._world_to_screen(0, y)
            painter.drawLine(0, int(sy), w, int(sy))
            y += grid_world

    def _draw_edges(self, painter: QPainter, snap: Snapshot, node_pos: Dict, agv_color: QColor):
        remaining_edge_ids: Set[str] = set()
        for es in snap.edge_states:
            remaining_edge_ids.add(es.get("edgeId", ""))
        last_seq = snap.last_node_seq_id

        edge_color = agv_color.darker(120)
        visited_color = QColor(agv_color)
        visited_color.setAlpha(80)

        for edge in snap.edges:
            eid = edge.get("edgeId", "")
            start_id = edge.get("startNodeId", "")
            end_id = edge.get("endNodeId", "")
            released = edge.get("released", False)
            seq = edge.get("sequenceId", -1)

            if start_id not in node_pos or end_id not in node_pos:
                continue

            sx, sy = self._world_to_screen(*node_pos[start_id])
            ex, ey = self._world_to_screen(*node_pos[end_id])

            is_visited = eid not in remaining_edge_ids and last_seq >= 0 and seq < last_seq

            if is_visited:
                pen = QPen(visited_color, 1.5, Qt.SolidLine)
            elif released:
                pen = QPen(edge_color, 2, Qt.SolidLine)
            else:
                pen = QPen(COL_HORIZON_EDGE, 1, Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(QPointF(sx, sy), QPointF(ex, ey))

            arrow_c = visited_color if is_visited else (edge_color if released else COL_HORIZON_EDGE)
            self._draw_arrow_head(painter, sx, sy, ex, ey, arrow_c)

    def _draw_arrow_head(self, painter, sx, sy, ex, ey, color):
        dx = ex - sx
        dy = ey - sy
        length = math.hypot(dx, dy)
        if length < 1:
            return
        dx /= length
        dy /= length

        mx = sx + dx * length * 0.7
        my = sy + dy * length * 0.7
        al, aw = 8, 4

        p1 = QPointF(mx + dx * al, my + dy * al)
        p2 = QPointF(mx - dy * aw, my + dx * aw)
        p3 = QPointF(mx + dy * aw, my - dx * aw)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawPolygon(QPolygonF([p1, p2, p3]))

    def _draw_trail(self, painter: QPainter, trail: List[Tuple[float, float]], agv_color: QColor):
        if len(trail) < 2:
            return
        trail_color = QColor(agv_color)
        trail_color.setAlpha(100)
        pen = QPen(trail_color, TRAIL_WIDTH, Qt.SolidLine)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        path = QPainterPath()
        sx, sy = self._world_to_screen(*trail[0])
        path.moveTo(sx, sy)
        for wx, wy in trail[1:]:
            sx, sy = self._world_to_screen(wx, wy)
            path.lineTo(sx, sy)
        painter.drawPath(path)

    def _draw_nodes(self, painter: QPainter, snap: Snapshot, node_pos: Dict, agv_color: QColor):
        font = QFont("sans-serif", 8)
        painter.setFont(font)

        remaining_ids: Set[str] = set()
        for ns in snap.node_states:
            remaining_ids.add(ns.get("nodeId", ""))

        last_nid = snap.last_node_id or ""
        last_seq = snap.last_node_seq_id

        visited_fill = QColor(agv_color.red(), agv_color.green(), agv_color.blue(), 50)
        visited_pen_c = QColor(agv_color.red(), agv_color.green(), agv_color.blue(), 100)

        for node in snap.nodes:
            nid = node.get("nodeId", "")
            released = node.get("released", False)
            seq = node.get("sequenceId", -1)
            pos = node.get("nodePosition", {})
            x, y = pos.get("x"), pos.get("y")
            if x is None or y is None:
                continue

            sx, sy = self._world_to_screen(x, y)
            # Only store if not already stored (avoid overwrite from another AGV)
            if nid not in self._node_screen_positions:
                self._node_screen_positions[nid] = (sx, sy, node)

            r = NODE_RADIUS
            is_current = (nid == last_nid) and last_nid != ""
            is_visited = (not is_current and nid not in remaining_ids
                          and last_seq >= 0 and seq <= last_seq)

            if is_current:
                painter.setPen(QPen(agv_color, 3))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPointF(sx, sy), r + 4, r + 4)
            elif is_visited:
                painter.setPen(QPen(visited_pen_c, 1.5))
                painter.setBrush(QBrush(visited_fill))
                painter.drawEllipse(QPointF(sx, sy), r, r)
            elif released:
                painter.setPen(QPen(agv_color.darker(120), 1.5))
                painter.setBrush(QBrush(agv_color))
                painter.drawEllipse(QPointF(sx, sy), r, r)
            else:
                painter.setPen(QPen(COL_HORIZON_NODE.darker(130), 1.5, Qt.DashLine))
                painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
                painter.drawEllipse(QPointF(sx, sy), r, r)

            painter.setPen(QPen(COL_NODE_TEXT))
            painter.drawText(QRectF(sx - 40, sy - r - 16, 80, 14), Qt.AlignCenter, nid)

    def _draw_robot(self, painter: QPainter, snap: Snapshot, agv_id: str, agv_color: QColor, selected: bool):
        if snap.agv_x is None or snap.agv_y is None:
            return

        sx, sy = self._world_to_screen(snap.agv_x, snap.agv_y)
        self._robot_screen_positions[agv_id] = (sx, sy)
        theta = snap.agv_theta or 0.0
        s = ROBOT_SIZE

        painter.save()
        painter.translate(sx, sy)
        painter.rotate(-math.degrees(theta))

        tri = QPolygonF([
            QPointF(s, 0),
            QPointF(-s * 0.6, s * 0.7),
            QPointF(-s * 0.6, -s * 0.7),
        ])

        # Selection ring
        if selected:
            painter.setPen(QPen(QColor(255, 255, 255), 6))
            painter.setBrush(Qt.NoBrush)
            painter.drawPolygon(tri)
            painter.setPen(QPen(agv_color, 3))
            painter.setBrush(Qt.NoBrush)
            painter.drawPolygon(tri)
        else:
            # White halo
            painter.setPen(QPen(QColor(255, 255, 255), 4))
            painter.setBrush(Qt.NoBrush)
            painter.drawPolygon(tri)

        painter.setPen(QPen(agv_color.darker(130), 1.5))
        painter.setBrush(QBrush(agv_color))
        painter.drawPolygon(tri)
        painter.restore()

        # AGV ID label below robot
        painter.setPen(QPen(agv_color.darker(150)))
        label_font = QFont("sans-serif", 7, QFont.Bold)
        painter.setFont(label_font)
        painter.drawText(QRectF(sx - 50, sy + s + 2, 100, 14), Qt.AlignCenter, agv_id)

    def _draw_legend(self, painter: QPainter):
        """Draw small color legend in top-left corner."""
        visible_ids = sorted(self._visible_agvs)
        if not visible_ids:
            return

        x0, y0 = 10, 10
        line_h = 16
        painter.setFont(QFont("sans-serif", 8))

        for i, aid in enumerate(visible_ids):
            color = self.get_agv_color(aid)
            y = y0 + i * line_h

            # Color swatch
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawRect(int(x0), int(y + 2), 10, 10)

            # Label
            painter.setPen(QPen(COL_NODE_TEXT))
            painter.drawText(int(x0 + 14), int(y + 12), aid)

    # ---- Mouse events ----

    def wheelEvent(self, event: QWheelEvent):
        pos = event.pos()
        old_wx, old_wy = self._screen_to_world(pos.x(), pos.y())
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15
        self._scale = max(1.0, min(self._scale * factor, 500.0))
        new_sx, new_sy = self._world_to_screen(old_wx, old_wy)
        self._offset_x += pos.x() - new_sx
        self._offset_y += pos.y() - new_sy
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton and event.modifiers() & Qt.ShiftModifier
        ):
            self._panning = True
            self._pan_start = event.pos()
            self._pan_offset_start = (self._offset_x, self._offset_y)
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            # Check robot click first
            for aid, (rx, ry) in self._robot_screen_positions.items():
                dx = event.pos().x() - rx
                dy = event.pos().y() - ry
                if math.hypot(dx, dy) <= ROBOT_SIZE + 4:
                    self.robot_clicked.emit(aid)
                    return
            # Then node click
            self._check_node_click(event.pos())

    def mouseMoveEvent(self, event):
        if self._panning:
            dx = event.pos().x() - self._pan_start.x()
            dy = event.pos().y() - self._pan_start.y()
            self._offset_x = self._pan_offset_start[0] + dx
            self._offset_y = self._pan_offset_start[1] + dy
            self.update()

    def mouseReleaseEvent(self, event):
        if self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)

    def _check_node_click(self, pos):
        threshold = NODE_RADIUS + 6
        for nid, (sx, sy, node_data) in self._node_screen_positions.items():
            dx = pos.x() - sx
            dy = pos.y() - sy
            if math.hypot(dx, dy) <= threshold:
                self.node_clicked.emit(nid, node_data)
                return


# ---------------------------------------------------------------------------
# InfoPanel – state information display with AGV selector
# ---------------------------------------------------------------------------
class InfoPanel(QWidget):
    agv_selected = pyqtSignal(str)  # agv_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        self.setMaximumWidth(400)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # AGV selector row
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("AGV:"))
        self._agv_combo = QComboBox()
        self._agv_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._agv_combo.currentTextChanged.connect(self._on_agv_combo_changed)
        selector_layout.addWidget(self._agv_combo)
        layout.addLayout(selector_layout)

        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setFont(QFont("Consolas, Courier", 10))
        self._text.setStyleSheet(
            "QTextEdit { background: #FFFFFF; border: 1px solid #DDD; }"
        )
        layout.addWidget(self._text)

        self._current_agvs: List[str] = []

    def update_agv_list(self, agv_ids: List[str], selected: str = ""):
        """Update the AGV combo box items."""
        if agv_ids == self._current_agvs:
            # Just update selection if list is same
            if selected and self._agv_combo.currentText() != selected:
                idx = self._agv_combo.findText(selected)
                if idx >= 0:
                    self._agv_combo.setCurrentIndex(idx)
            return

        self._current_agvs = list(agv_ids)
        self._agv_combo.blockSignals(True)
        self._agv_combo.clear()
        for aid in agv_ids:
            self._agv_combo.addItem(aid)
        if selected and selected in agv_ids:
            self._agv_combo.setCurrentText(selected)
        elif agv_ids:
            self._agv_combo.setCurrentIndex(0)
        self._agv_combo.blockSignals(False)

    def set_snapshot(self, snap: Snapshot, agv_id: str = "", timestamp: str = "", agv_color: QColor = None):
        parts = []

        # Header
        color_hex = agv_color.name() if agv_color else "#333"
        if agv_id:
            parts.append(f'<b style="color:{color_hex}">AGV: {agv_id}</b>')
        if timestamp:
            parts.append(f"<b>Time:</b> {timestamp}")

        # Connection
        cs = snap.connection_state
        color_map = {"ONLINE": "#4CAF50", "OFFLINE": "#FFC107", "CONNECTIONBROKEN": "#F44336"}
        c_color = color_map.get(cs, "#999")
        parts.append(f'<b>Connection:</b> <span style="color:{c_color};font-weight:bold">{cs}</span>')

        parts.append("<hr>")

        # Order info
        if snap.order:
            parts.append("<b>Order</b>")
            parts.append(f"  orderId: {snap.order_id}")
            parts.append(f"  orderUpdateId: {snap.order_update_id}")
            n_base = sum(1 for n in snap.nodes if n.get("released"))
            n_horizon = sum(1 for n in snap.nodes if not n.get("released"))
            parts.append(f"  nodes: {len(snap.nodes)} (base={n_base}, horizon={n_horizon})")
            parts.append(f"  edges: {len(snap.edges)}")
        else:
            parts.append("<b>Order:</b> <i>none</i>")

        parts.append("<hr>")

        # AGV state
        if snap.state:
            parts.append("<b>AGV State</b>")
            parts.append(f"  driving: {snap.driving}")
            parts.append(f"  lastNodeId: {snap.last_node_id}")
            if snap.agv_x is not None:
                parts.append(f"  position: ({snap.agv_x:.3f}, {snap.agv_y:.3f})")
                if snap.agv_theta is not None:
                    parts.append(f"  theta: {snap.agv_theta:.3f} rad ({math.degrees(snap.agv_theta):.1f}\u00b0)")
            parts.append(f"  battery: {snap.battery_charge:.1f}%")
            op_mode = snap.state.get("operatingMode", "?")
            parts.append(f"  operatingMode: {op_mode}")
            parts.append(f"  paused: {snap.state.get('paused', False)}")
            parts.append(f"  newBaseRequest: {snap.state.get('newBaseRequest', False)}")
        else:
            parts.append("<b>AGV State:</b> <i>none</i>")

        parts.append("<hr>")

        # Node States
        if snap.node_states:
            parts.append(f"<b>Remaining Nodes ({len(snap.node_states)})</b>")
            for ns in snap.node_states:
                rel = "base" if ns.get("released") else "horizon"
                parts.append(f"  {ns.get('nodeId','')} (seq={ns.get('sequenceId','')}, {rel})")

        # Edge States
        if snap.edge_states:
            parts.append(f"<b>Remaining Edges ({len(snap.edge_states)})</b>")
            for es in snap.edge_states:
                rel = "base" if es.get("released") else "horizon"
                parts.append(f"  {es.get('edgeId','')} (seq={es.get('sequenceId','')}, {rel})")

        parts.append("<hr>")

        # Action States
        if snap.action_states:
            parts.append(f"<b>Action States ({len(snap.action_states)})</b>")
            status_colors = {
                "WAITING": "#9E9E9E", "INITIALIZING": "#2196F3",
                "RUNNING": "#4CAF50", "PAUSED": "#FFC107",
                "FINISHED": "#8BC34A", "FAILED": "#F44336",
            }
            for a in snap.action_states:
                st = a.get("actionStatus", "?")
                sc = status_colors.get(st, "#999")
                parts.append(
                    f'  {a.get("actionType","?")} [{a.get("actionId","")}]: '
                    f'<span style="color:{sc}">{st}</span>'
                )

        # Errors
        if snap.errors:
            parts.append("<hr>")
            parts.append(f'<b style="color:#F44336">Errors ({len(snap.errors)})</b>')
            for err in snap.errors:
                lvl = err.get("errorLevel", "?")
                parts.append(f'  [{lvl}] {err.get("errorType","?")} - {err.get("errorDescription","")}')

        html = "<pre style='font-family: Consolas, monospace; font-size: 10pt; white-space: pre-wrap;'>"
        html += "\n".join(parts)
        html += "</pre>"
        self._text.setHtml(html)

    def _on_agv_combo_changed(self, text):
        if text:
            self.agv_selected.emit(text)


# ---------------------------------------------------------------------------
# OrderProgressPanel – node/edge sequence with progress indication
# ---------------------------------------------------------------------------
class OrderProgressPanel(QWidget):
    """Displays the current order's node-edge sequence with progress status."""

    # Status colors
    _COL_VISITED = "#9E9E9E"       # grey – already passed
    _COL_CURRENT = "#4CAF50"       # green – AGV is here now
    _COL_PENDING_BASE = "#2196F3"  # blue – released, not yet reached
    _COL_HORIZON = "#FF9800"       # orange – horizon (not released)
    _COL_ARROW = "#BDBDBD"         # light grey arrow between items

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Header row with toggle and copy buttons
        header_layout = QHBoxLayout()
        header_layout.setSpacing(4)
        header = QLabel("<b>Order Progress</b>")
        header.setStyleSheet("font-size: 11px; color: #555;")
        header_layout.addWidget(header)
        header_layout.addStretch()

        btn_style = (
            "QPushButton { background: #F5F5F5; border: 1px solid #CCC; "
            "padding: 2px 8px; font-size: 10px; border-radius: 3px; }"
            "QPushButton:hover { background: #E0E0E0; }"
            "QPushButton:checked { background: #2196F3; color: white; border: 1px solid #1976D2; }"
        )

        self._btn_toggle = QPushButton("Raw JSON")
        self._btn_toggle.setCheckable(True)
        self._btn_toggle.setStyleSheet(btn_style)
        self._btn_toggle.setToolTip("Toggle between progress view and raw order JSON")
        self._btn_toggle.clicked.connect(self._toggle_view)
        header_layout.addWidget(self._btn_toggle)

        self._btn_copy = QPushButton("Copy")
        self._btn_copy.setStyleSheet(btn_style)
        self._btn_copy.setToolTip("Copy raw order JSON to clipboard")
        self._btn_copy.clicked.connect(self._copy_order_json)
        header_layout.addWidget(self._btn_copy)

        layout.addLayout(header_layout)

        self._summary_label = QLabel("")
        self._summary_label.setStyleSheet("font-size: 10px; color: #888;")
        layout.addWidget(self._summary_label)

        # Progress view (HTML)
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setFont(QFont("Consolas, Courier", 9))
        self._text.setStyleSheet(
            "QTextEdit { background: #FFFFFF; border: 1px solid #DDD; }"
        )
        layout.addWidget(self._text)

        # Raw JSON view (plain text, hidden by default)
        self._raw_text = QTextEdit()
        self._raw_text.setReadOnly(True)
        self._raw_text.setFont(QFont("Consolas, Courier", 9))
        self._raw_text.setStyleSheet(
            "QTextEdit { background: #FFFDE7; border: 1px solid #DDD; }"
        )
        self._raw_text.setVisible(False)
        layout.addWidget(self._raw_text)

        self._raw_json_str = ""
        self._showing_raw = False

    def _toggle_view(self):
        self._showing_raw = self._btn_toggle.isChecked()
        self._text.setVisible(not self._showing_raw)
        self._raw_text.setVisible(self._showing_raw)
        self._btn_toggle.setText("Progress" if self._showing_raw else "Raw JSON")

    def _copy_order_json(self):
        if not self._raw_json_str:
            return
        clipboard = QApplication.clipboard()
        clipboard.setText(self._raw_json_str)
        self._btn_copy.setText("Copied!")
        QTimer.singleShot(1500, lambda: self._btn_copy.setText("Copy"))

    def set_snapshot(self, snap: Snapshot, agv_color: QColor = None):
        """Build and display the node-edge progress sequence."""
        # Update raw JSON
        if snap.order:
            self._raw_json_str = json.dumps(snap.order, indent=2, ensure_ascii=False)
            self._raw_text.setPlainText(self._raw_json_str)
        else:
            self._raw_json_str = ""
            self._raw_text.setPlainText("")

        if not snap.order or (not snap.nodes and not snap.edges):
            self._summary_label.setText("No active order")
            self._text.setHtml(
                "<pre style='color:#999; font-size:9pt;'>No order data available</pre>"
            )
            return

        # Build combined sequence sorted by sequenceId
        items = []  # list of (sequenceId, type, id, released, data_dict)
        for n in snap.nodes:
            items.append((
                n.get("sequenceId", -1),
                "node",
                n.get("nodeId", "?"),
                n.get("released", False),
                n,
            ))
        for e in snap.edges:
            items.append((
                e.get("sequenceId", -1),
                "edge",
                e.get("edgeId", "?"),
                e.get("released", False),
                e,
            ))
        items.sort(key=lambda x: x[0])

        # Determine which nodes/edges are still remaining (from state)
        remaining_node_ids: set = set()
        for ns in snap.node_states:
            remaining_node_ids.add(ns.get("nodeId", ""))
        remaining_edge_ids: set = set()
        for es in snap.edge_states:
            remaining_edge_ids.add(es.get("edgeId", ""))

        last_nid = snap.last_node_id or ""
        last_seq = snap.last_node_seq_id

        # Classify each item
        visited_count = 0
        current_item_index = -1
        scroll_anchor = ""
        lines = []

        for idx, (seq, item_type, item_id, released, data) in enumerate(items):
            if item_type == "node":
                is_current = (item_id == last_nid and last_nid != "")
                is_remaining = item_id in remaining_node_ids
                is_visited = (not is_current and not is_remaining
                              and last_seq >= 0 and seq <= last_seq)
            else:  # edge
                is_current = False
                is_remaining = item_id in remaining_edge_ids
                is_visited = (not is_remaining and last_seq >= 0 and seq < last_seq)

            # Determine status
            if is_current:
                status = "current"
                color = self._COL_CURRENT
                icon = "&#9658;"   # ►
                current_item_index = idx
                visited_count += 1
            elif is_visited:
                status = "visited"
                color = self._COL_VISITED
                icon = "&#10003;"  # ✓
                visited_count += 1
            elif released:
                status = "pending"
                color = self._COL_PENDING_BASE
                icon = "&#9675;"   # ○
            else:
                status = "horizon"
                color = self._COL_HORIZON
                icon = "&#9676;"   # ◌ (dashed circle)

            release_tag = "base" if released else "horizon"
            type_label = "N" if item_type == "node" else "E"

            # Node: show position info, Edge: show start→end
            detail = ""
            if item_type == "node":
                pos = data.get("nodePosition", {})
                x, y = pos.get("x"), pos.get("y")
                if x is not None and y is not None:
                    detail = f"  ({x:.2f}, {y:.2f})"
                actions = data.get("actions", [])
                if actions:
                    action_types = [a.get("actionType", "?") for a in actions]
                    detail += f"  [{', '.join(action_types)}]"
            else:
                start = data.get("startNodeId", "?")
                end = data.get("endNodeId", "?")
                detail = f"  {start} → {end}"

            # Anchor for scrolling to current
            anchor = ""
            if is_current:
                anchor = ' id="current_pos"'
                scroll_anchor = "current_pos"

            # Build the line
            if is_current:
                line = (
                    f'<div{anchor} style="background:#E8F5E9; padding:2px 4px; '
                    f'margin:1px 0; border-left:3px solid {color}; font-weight:bold;">'
                    f'<span style="color:{color}">{icon}</span> '
                    f'<span style="color:{color}">[{type_label}] {item_id}</span>'
                    f'<span style="color:#666; font-size:8pt;"> seq={seq} {release_tag}{detail}</span>'
                    f'</div>'
                )
            elif is_visited:
                line = (
                    f'<div style="padding:2px 4px; margin:1px 0; '
                    f'border-left:3px solid #E0E0E0; opacity:0.7;">'
                    f'<span style="color:{color}">{icon}</span> '
                    f'<span style="color:{color}; text-decoration:line-through;">'
                    f'[{type_label}] {item_id}</span>'
                    f'<span style="color:#AAA; font-size:8pt;"> seq={seq}{detail}</span>'
                    f'</div>'
                )
            else:
                border_color = color if released else "#FFE0B2"
                line = (
                    f'<div style="padding:2px 4px; margin:1px 0; '
                    f'border-left:3px solid {border_color};">'
                    f'<span style="color:{color}">{icon}</span> '
                    f'<span style="color:{color}">[{type_label}] {item_id}</span>'
                    f'<span style="color:#888; font-size:8pt;"> seq={seq} {release_tag}{detail}</span>'
                    f'</div>'
                )
            lines.append(line)

        # Summary
        total_items = len(items)
        n_nodes = sum(1 for i in items if i[1] == "node")
        n_edges = sum(1 for i in items if i[1] == "edge")
        n_base = sum(1 for i in items if i[3])
        self._summary_label.setText(
            f"Order: {snap.order_id} (update={snap.order_update_id})  |  "
            f"Progress: {visited_count}/{total_items}  |  "
            f"Nodes={n_nodes}, Edges={n_edges}  |  Base={n_base}, Horizon={total_items - n_base}"
        )

        # Legend
        legend = (
            '<div style="font-size:8pt; color:#888; margin-bottom:4px; padding:2px;">'
            f'<span style="color:{self._COL_VISITED}">&#10003; Visited</span> &nbsp; '
            f'<span style="color:{self._COL_CURRENT}">&#9658; Current</span> &nbsp; '
            f'<span style="color:{self._COL_PENDING_BASE}">&#9675; Pending(base)</span> &nbsp; '
            f'<span style="color:{self._COL_HORIZON}">&#9676; Horizon</span>'
            '</div><hr style="border:none; border-top:1px solid #EEE; margin:2px 0;">'
        )

        html = (
            '<div style="font-family: Consolas, monospace; font-size: 9pt;">'
            + legend
            + "\n".join(lines)
            + '</div>'
        )
        self._text.setHtml(html)

        # Auto-scroll to current position
        if scroll_anchor:
            self._text.scrollToAnchor(scroll_anchor)


# ---------------------------------------------------------------------------
# TimelineWidget – slider + playback controls
# ---------------------------------------------------------------------------
class TimelineWidget(QWidget):
    index_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(80)
        self._max_index = 0
        self._playing = False
        self._speed = 1.0
        self._order_markers: List[int] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider)

        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(8)

        self._btn_first = QPushButton("|◀")
        self._btn_prev = QPushButton("◀")
        self._btn_play = QPushButton("▶ Play")
        self._btn_next = QPushButton("▶")
        self._btn_last = QPushButton("▶|")

        for btn in (self._btn_first, self._btn_prev, self._btn_next, self._btn_last):
            btn.setFixedWidth(40)
            btn.setStyleSheet("font-size: 12px;")

        self._btn_play.setFixedWidth(80)
        self._btn_play.setStyleSheet("font-size: 12px; font-weight: bold;")

        self._btn_first.clicked.connect(self.go_first)
        self._btn_prev.clicked.connect(self.go_prev)
        self._btn_play.clicked.connect(self.toggle_play)
        self._btn_next.clicked.connect(self.go_next)
        self._btn_last.clicked.connect(self.go_last)

        ctrl_layout.addWidget(self._btn_first)
        ctrl_layout.addWidget(self._btn_prev)
        ctrl_layout.addWidget(self._btn_play)
        ctrl_layout.addWidget(self._btn_next)
        ctrl_layout.addWidget(self._btn_last)

        ctrl_layout.addSpacing(16)

        self._step_label = QLabel("0 / 0")
        self._step_label.setMinimumWidth(120)
        ctrl_layout.addWidget(self._step_label)

        self._time_label = QLabel("")
        self._time_label.setMinimumWidth(200)
        ctrl_layout.addWidget(self._time_label)

        ctrl_layout.addStretch()

        ctrl_layout.addWidget(QLabel("Speed:"))
        self._speed_combo = QComboBox()
        self._speed_combo.addItems(["0.25x", "0.5x", "1x", "2x", "4x", "10x"])
        self._speed_combo.setCurrentIndex(2)
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        ctrl_layout.addWidget(self._speed_combo)

        layout.addLayout(ctrl_layout)

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_step)
        self._play_interval_ms = 500

    def set_range(self, max_index: int):
        self._max_index = max_index
        self._slider.setMaximum(max_index)
        self._update_label()

    def set_order_markers(self, indices: List[int]):
        self._order_markers = indices

    def set_timestamp(self, ts: str):
        self._time_label.setText(ts)

    @property
    def current_index(self) -> int:
        return self._slider.value()

    def set_index(self, idx: int):
        self._slider.setValue(idx)

    def go_first(self):
        self._slider.setValue(0)

    def go_prev(self):
        self._slider.setValue(max(0, self._slider.value() - 1))

    def go_next(self):
        self._slider.setValue(min(self._max_index, self._slider.value() + 1))

    def go_last(self):
        self._slider.setValue(self._max_index)

    def toggle_play(self):
        self._playing = not self._playing
        if self._playing:
            self._btn_play.setText("⏸ Pause")
            self._play_timer.start(self._play_interval_ms)
        else:
            self._btn_play.setText("▶ Play")
            self._play_timer.stop()

    def stop_play(self):
        self._playing = False
        self._btn_play.setText("▶ Play")
        self._play_timer.stop()

    def _play_step(self):
        if self._slider.value() >= self._max_index:
            self.stop_play()
            return
        self._slider.setValue(self._slider.value() + 1)

    def _on_slider_changed(self, value):
        self._update_label()
        self.index_changed.emit(value)

    def _on_speed_changed(self, idx):
        speeds = [0.25, 0.5, 1.0, 2.0, 4.0, 10.0]
        if 0 <= idx < len(speeds):
            self._speed = speeds[idx]
            self._play_interval_ms = max(10, int(500 / self._speed))
            if self._playing:
                self._play_timer.start(self._play_interval_ms)

    def _update_label(self):
        self._step_label.setText(f"{self._slider.value()} / {self._max_index}")

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._order_markers or self._max_index <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        slider_geom = self._slider.geometry()
        margin = 8
        x_start = slider_geom.x() + margin
        x_end = slider_geom.x() + slider_geom.width() - margin
        y_top = slider_geom.y()

        total_range = x_end - x_start
        if total_range <= 0:
            return

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(COL_ORDER_MARKER))

        for idx in self._order_markers:
            frac = idx / self._max_index if self._max_index > 0 else 0
            x = x_start + frac * total_range
            tri = QPolygonF([
                QPointF(x, y_top),
                QPointF(x - 4, y_top - 8),
                QPointF(x + 4, y_top - 8),
            ])
            painter.drawPolygon(tri)

        painter.end()


# ---------------------------------------------------------------------------
# MainWindow – top-level layout + mode switching + signal wiring
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VDA5050 Log Visualizer")
        self.resize(1280, 800)

        self._store = LogStore()
        self._mqtt_client = MqttLogClient(self)
        self._mode = "offline"
        self._live_follow = True
        self._auto_fit_done = False

        # Multi-AGV state
        self._visible_agvs: Set[str] = set()
        self._selected_agv: str = ""
        self._known_agvs: List[str] = []
        self._live_discovered_agvs: Set[str] = set()  # AGVs discovered via connection messages in live mode

        # Recording state
        self._recording = False
        self._record_file = None
        self._record_path = ""
        self._record_count = 0

        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._connect_signals()

        self.statusBar().showMessage("Ready – File → Open Log (Ctrl+O) to load a JSONL file")

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        self._canvas = MapCanvas()

        # Right side: InfoPanel + OrderProgressPanel in vertical splitter
        right_splitter = QSplitter(Qt.Vertical)
        self._info_panel = InfoPanel()
        self._order_progress = OrderProgressPanel()
        right_splitter.addWidget(self._info_panel)
        right_splitter.addWidget(self._order_progress)
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 2)

        splitter.addWidget(self._canvas)
        splitter.addWidget(right_splitter)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter, stretch=1)

        self._timeline = TimelineWidget()
        main_layout.addWidget(self._timeline)

    def _setup_menu(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        open_act = QAction("Open Log...", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._open_file)
        file_menu.addAction(open_act)
        save_act = QAction("Save Recording...", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self._save_recording)
        file_menu.addAction(save_act)
        file_menu.addSeparator()
        quit_act = QAction("Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # Mode menu
        mode_menu = menubar.addMenu("Mode")
        self._offline_act = QAction("Offline (Log Review)", self)
        self._offline_act.setCheckable(True)
        self._offline_act.setChecked(True)
        self._offline_act.triggered.connect(lambda: self._set_mode("offline"))
        mode_menu.addAction(self._offline_act)
        self._live_act = QAction("Live (MQTT)", self)
        self._live_act.setCheckable(True)
        self._live_act.triggered.connect(lambda: self._set_mode("live"))
        mode_menu.addAction(self._live_act)
        mode_menu.addSeparator()
        self._mqtt_connect_act = QAction("Connect MQTT...", self)
        self._mqtt_connect_act.triggered.connect(self._connect_mqtt)
        mode_menu.addAction(self._mqtt_connect_act)
        self._mqtt_disconnect_act = QAction("Disconnect MQTT", self)
        self._mqtt_disconnect_act.setEnabled(False)
        self._mqtt_disconnect_act.triggered.connect(self._disconnect_mqtt)
        mode_menu.addAction(self._mqtt_disconnect_act)
        mode_menu.addSeparator()
        self._go_live_act = QAction("Go Live (Follow Latest)", self)
        self._go_live_act.setShortcut("Ctrl+L")
        self._go_live_act.setCheckable(True)
        self._go_live_act.setChecked(True)
        self._go_live_act.triggered.connect(self._toggle_go_live)
        mode_menu.addAction(self._go_live_act)

        # View menu
        view_menu = menubar.addMenu("View")
        fit_act = QAction("Fit to Content", self)
        fit_act.setShortcut("Ctrl+F")
        fit_act.triggered.connect(self._canvas.fit_to_content)
        view_menu.addAction(fit_act)

    def _setup_toolbar(self):
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setStyleSheet(
            "QToolBar { spacing: 6px; padding: 2px 4px; border-bottom: 1px solid #DDD; }"
            "QPushButton, QToolButton { padding: 4px 12px; font-size: 12px; border-radius: 3px; }"
        )

        # Record button
        self._btn_record = QPushButton("⏺ Record")
        self._btn_record.setCheckable(True)
        self._btn_record.setStyleSheet(
            "QPushButton { background: #F5F5F5; border: 1px solid #CCC; }"
            "QPushButton:checked { background: #F44336; color: white; border: 1px solid #D32F2F; }"
        )
        self._btn_record.setToolTip("Start/stop recording MQTT messages to JSONL file")
        self._btn_record.clicked.connect(self._toggle_recording)
        toolbar.addWidget(self._btn_record)

        self._record_label = QLabel("")
        self._record_label.setStyleSheet("color: #666; font-size: 11px; margin-left: 4px;")
        toolbar.addWidget(self._record_label)

        # Clear button (live mode)
        self._btn_clear = QPushButton("Clear")
        self._btn_clear.setStyleSheet(
            "QPushButton { background: #F5F5F5; border: 1px solid #CCC; }"
            "QPushButton:hover { background: #FFEBEE; border: 1px solid #E57373; }"
        )
        self._btn_clear.setToolTip("Clear all accumulated log data (live mode)")
        self._btn_clear.clicked.connect(self._clear_live_data)
        toolbar.addWidget(self._btn_clear)

        toolbar.addSeparator()

        # AGV filter button
        self._agv_filter_btn = QToolButton()
        self._agv_filter_btn.setText("AGV Filter: All")
        self._agv_filter_btn.setPopupMode(QToolButton.InstantPopup)
        self._agv_filter_btn.setStyleSheet(
            "QToolButton { background: #F5F5F5; border: 1px solid #CCC; padding: 4px 10px; }"
            "QToolButton::menu-indicator { image: none; }"
        )
        self._agv_filter_menu = QMenu(self._agv_filter_btn)
        self._agv_filter_btn.setMenu(self._agv_filter_menu)
        toolbar.addWidget(self._agv_filter_btn)

    def _connect_signals(self):
        self._timeline.index_changed.connect(self._on_index_changed)
        self._canvas.node_clicked.connect(self._on_node_clicked)
        self._canvas.robot_clicked.connect(self._on_robot_clicked)
        self._info_panel.agv_selected.connect(self._on_agv_selected_from_panel)
        self._mqtt_client.log_received.connect(self._on_mqtt_log)
        self._mqtt_client.connected.connect(self._on_mqtt_connected)
        self._mqtt_client.disconnected.connect(self._on_mqtt_disconnected)
        self._mqtt_client.error_occurred.connect(self._on_mqtt_error)

    # -- AGV filter management --

    def _refresh_agv_filter(self, agv_ids: List[str]):
        """Rebuild the AGV filter menu when the set of known AGVs changes."""
        if agv_ids == self._known_agvs:
            return
        self._known_agvs = list(agv_ids)

        self._agv_filter_menu.clear()

        # Select All / Deselect All
        select_all_act = QAction("Select All", self)
        select_all_act.triggered.connect(self._filter_select_all)
        self._agv_filter_menu.addAction(select_all_act)
        deselect_all_act = QAction("Deselect All", self)
        deselect_all_act.triggered.connect(self._filter_deselect_all)
        self._agv_filter_menu.addAction(deselect_all_act)
        self._agv_filter_menu.addSeparator()

        # Per-AGV checkable actions
        for aid in agv_ids:
            act = QAction(aid, self)
            act.setCheckable(True)
            act.setChecked(aid in self._visible_agvs)
            act.toggled.connect(lambda checked, a=aid: self._on_agv_filter_toggled(a, checked))
            self._agv_filter_menu.addAction(act)

        # Auto-add new AGVs to visible set
        for aid in agv_ids:
            if aid not in self._visible_agvs:
                self._visible_agvs.add(aid)

        # If selected AGV not in list anymore, pick first
        if self._selected_agv not in agv_ids and agv_ids:
            self._selected_agv = agv_ids[0]

        self._update_filter_button_text()

    def _on_agv_filter_toggled(self, agv_id: str, checked: bool):
        if checked:
            self._visible_agvs.add(agv_id)
        else:
            self._visible_agvs.discard(agv_id)
            if self._selected_agv == agv_id:
                remaining = sorted(self._visible_agvs)
                self._selected_agv = remaining[0] if remaining else ""

        self._update_filter_button_text()
        self._on_index_changed(self._timeline.current_index)

    def _filter_select_all(self):
        self._visible_agvs = set(self._known_agvs)
        for act in self._agv_filter_menu.actions():
            if act.isCheckable():
                act.setChecked(True)
        self._update_filter_button_text()
        self._on_index_changed(self._timeline.current_index)

    def _filter_deselect_all(self):
        self._visible_agvs.clear()
        for act in self._agv_filter_menu.actions():
            if act.isCheckable():
                act.setChecked(False)
        self._update_filter_button_text()
        self._on_index_changed(self._timeline.current_index)

    def _update_filter_button_text(self):
        n_visible = len(self._visible_agvs)
        n_total = len(self._known_agvs)
        if n_visible == n_total:
            self._agv_filter_btn.setText(f"AGV Filter: All ({n_total})")
        elif n_visible == 0:
            self._agv_filter_btn.setText("AGV Filter: None")
        else:
            self._agv_filter_btn.setText(f"AGV Filter: {n_visible}/{n_total}")

    # -- File operations --

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open JSONL Log", "", "JSONL files (*.jsonl);;All files (*)"
        )
        if not path:
            return

        try:
            count = self._store.load_file(path)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{exc}")
            return

        if count == 0:
            QMessageBox.warning(self, "Warning", "No valid log entries found in file.")
            return

        self._set_mode("offline")
        agv_ids = self._store.get_agv_ids()
        self._visible_agvs = set(agv_ids)
        self._selected_agv = agv_ids[0] if agv_ids else ""
        self._refresh_agv_filter(agv_ids)

        self._timeline.set_range(count - 1)
        self._timeline.set_order_markers(self._store.order_change_indices)
        self._timeline.set_index(0)
        self._auto_fit_done = False
        self._on_index_changed(0)
        self.statusBar().showMessage(f"Loaded {count} entries from {path} ({len(agv_ids)} AGV(s))")

    def _save_recording(self):
        if not self._store.entries:
            QMessageBox.information(self, "Info", "No data to save.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Recording", "recording.jsonl", "JSONL files (*.jsonl)"
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                for e in self._store.entries:
                    line = json.dumps(
                        {"timestamp": e.timestamp, "topic": e.topic, "data": e.data},
                        ensure_ascii=False,
                    )
                    f.write(line + "\n")
            self.statusBar().showMessage(f"Saved {len(self._store.entries)} entries to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{exc}")

    # -- Recording --

    def _toggle_recording(self):
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        default_name = f"vda5050_log_{ts}.jsonl"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Recording As", default_name, "JSONL files (*.jsonl)"
        )
        if not path:
            self._btn_record.setChecked(False)
            return

        try:
            self._record_file = open(path, "w", encoding="utf-8")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Cannot open file:\n{exc}")
            self._btn_record.setChecked(False)
            return

        self._recording = True
        self._record_path = path
        self._record_count = 0
        self._btn_record.setText("⏹ Stop")
        self._record_label.setText(f"Recording → {path}")
        self.statusBar().showMessage(f"Recording to {path}")

    def _stop_recording(self):
        if self._record_file:
            self._record_file.close()
            self._record_file = None
        self._recording = False
        self._btn_record.setChecked(False)
        self._btn_record.setText("⏺ Record")
        msg = f"Saved {self._record_count} entries → {self._record_path}"
        self._record_label.setText(msg)
        self.statusBar().showMessage(msg)

    def _record_entry(self, entry: LogEntry):
        if not self._recording or not self._record_file:
            return
        try:
            line = json.dumps(
                {"timestamp": entry.timestamp, "topic": entry.topic, "data": entry.data},
                ensure_ascii=False,
            )
            self._record_file.write(line + "\n")
            self._record_file.flush()
            self._record_count += 1
            self._record_label.setText(f"⏺ {self._record_count} entries → {self._record_path}")
        except Exception as exc:
            self.statusBar().showMessage(f"Recording write error: {exc}")
            self._stop_recording()

    # -- Mode switching --

    def _set_mode(self, mode: str):
        self._mode = mode
        self._offline_act.setChecked(mode == "offline")
        self._live_act.setChecked(mode == "live")
        if mode == "offline":
            self._timeline.stop_play()
            self.statusBar().showMessage("Offline mode – use timeline to navigate")
        else:
            self.statusBar().showMessage("Live mode – waiting for MQTT data")

    # -- Clear live data --

    def _clear_live_data(self):
        """Clear all accumulated log data while keeping MQTT connection alive."""
        if self._mode != "live":
            self.statusBar().showMessage("Clear is only available in live mode")
            return

        self._store.clear()
        self._visible_agvs.clear()
        self._selected_agv = ""
        self._known_agvs.clear()
        self._live_discovered_agvs.clear()
        self._agv_filter_menu.clear()
        self._update_filter_button_text()
        self._timeline.set_range(0)
        self._timeline.set_order_markers([])
        self._timeline.set_timestamp("")
        self._auto_fit_done = False
        self._canvas.set_data({}, {}, set())
        self._info_panel.update_agv_list([])
        self._info_panel._text.clear()
        self._order_progress.set_snapshot(Snapshot())
        self.statusBar().showMessage("Live data cleared – waiting for new messages")

    # -- MQTT operations --

    def _connect_mqtt(self):
        dlg = MqttConnectionDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return

        params = dlg.get_params()
        self._store.clear()
        self._visible_agvs.clear()
        self._selected_agv = ""
        self._known_agvs.clear()
        self._live_discovered_agvs.clear()
        self._agv_filter_menu.clear()
        self._timeline.set_range(0)
        self._auto_fit_done = False
        self._mqtt_client.connect_to_broker(**params)
        self._set_mode("live")
        mfr = params["manufacturer"]
        self.statusBar().showMessage(
            f"Connecting to {params['host']}:{params['port']} – subscribing to all {mfr} AGVs..."
        )

    def _disconnect_mqtt(self):
        self._mqtt_client.stop()
        self._mqtt_disconnect_act.setEnabled(False)
        self._mqtt_connect_act.setEnabled(True)
        self.statusBar().showMessage("MQTT disconnected")

    def _toggle_go_live(self):
        self._live_follow = self._go_live_act.isChecked()

    @pyqtSlot(object)
    def _on_mqtt_log(self, entry: LogEntry):
        self._store.append_entry(entry)
        self._record_entry(entry)

        # Discover new AGVs from connection messages
        if entry.topic == "connection" and entry.agv_id:
            if entry.agv_id not in self._live_discovered_agvs:
                self._live_discovered_agvs.add(entry.agv_id)
                agv_ids = sorted(self._live_discovered_agvs)
                self._refresh_agv_filter(agv_ids)
                conn_state = entry.data.get("connectionState", "?")
                self.statusBar().showMessage(
                    f"Discovered AGV: {entry.agv_id} ({conn_state}) – {len(agv_ids)} robot(s) total"
                )

        count = len(self._store.entries)
        self._timeline.set_range(count - 1)
        self._timeline.set_order_markers(self._store.order_change_indices)

        if self._live_follow:
            self._timeline.set_index(count - 1)

    @pyqtSlot()
    def _on_mqtt_connected(self):
        self._mqtt_disconnect_act.setEnabled(True)
        self._mqtt_connect_act.setEnabled(False)
        self.statusBar().showMessage("MQTT connected – receiving data")

    @pyqtSlot()
    def _on_mqtt_disconnected(self):
        self._mqtt_disconnect_act.setEnabled(False)
        self._mqtt_connect_act.setEnabled(True)
        self.statusBar().showMessage("MQTT disconnected")

    @pyqtSlot(str)
    def _on_mqtt_error(self, msg: str):
        self.statusBar().showMessage(f"MQTT Error: {msg}")
        QMessageBox.warning(self, "MQTT Error", msg)

    # -- Display update --

    @pyqtSlot(int)
    def _on_index_changed(self, index: int):
        if not self._store.entries:
            return

        snapshots = self._store.get_snapshots(index)
        trails: Dict[str, List[Tuple[float, float]]] = {}
        for aid in self._visible_agvs:
            trails[aid] = self._store.get_trail(index, aid)

        timestamp = self._store.entries[index].timestamp if index < len(self._store.entries) else ""

        self._canvas.set_data(snapshots, trails, self._visible_agvs, self._selected_agv)
        self._timeline.set_timestamp(timestamp)

        # Update InfoPanel + OrderProgressPanel
        agv_ids = sorted(self._visible_agvs & set(snapshots.keys()))
        self._info_panel.update_agv_list(agv_ids, self._selected_agv)
        if self._selected_agv in snapshots:
            agv_color = self._canvas.get_agv_color(self._selected_agv)
            snap = snapshots[self._selected_agv]
            self._info_panel.set_snapshot(snap, self._selected_agv, timestamp, agv_color)
            self._order_progress.set_snapshot(snap, agv_color)
        elif agv_ids:
            self._selected_agv = agv_ids[0]
            agv_color = self._canvas.get_agv_color(self._selected_agv)
            snap = snapshots[self._selected_agv]
            self._info_panel.set_snapshot(snap, self._selected_agv, timestamp, agv_color)
            self._order_progress.set_snapshot(snap, agv_color)

        # Auto-fit on first data display
        if not self._auto_fit_done and snapshots:
            has_nodes = any(s.nodes for s in snapshots.values())
            if has_nodes:
                self._canvas.fit_to_content()
                self._auto_fit_done = True

    # -- Robot / Node click --

    @pyqtSlot(str)
    def _on_robot_clicked(self, agv_id: str):
        self._selected_agv = agv_id
        self._on_index_changed(self._timeline.current_index)

    @pyqtSlot(str)
    def _on_agv_selected_from_panel(self, agv_id: str):
        self._selected_agv = agv_id
        self._on_index_changed(self._timeline.current_index)

    @pyqtSlot(str, dict)
    def _on_node_clicked(self, node_id: str, node_data: dict):
        pos = node_data.get("nodePosition", {})
        released = node_data.get("released", False)
        actions = node_data.get("actions", [])
        seq = node_data.get("sequenceId", "?")

        info = (
            f"Node: {node_id}\n"
            f"Sequence ID: {seq}\n"
            f"Released: {released} ({'base' if released else 'horizon'})\n"
            f"Position: ({pos.get('x', '?')}, {pos.get('y', '?')})\n"
            f"Map: {pos.get('mapId', '?')}\n"
        )
        if actions:
            info += f"\nActions ({len(actions)}):\n"
            for a in actions:
                info += f"  - {a.get('actionType', '?')} [{a.get('actionId', '')}] ({a.get('blockingType', '?')})\n"

        QMessageBox.information(self, f"Node: {node_id}", info)

    def closeEvent(self, event):
        if self._recording:
            self._stop_recording()
        self._mqtt_client.stop()
        event.accept()


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QMainWindow { background: #F5F5F5; }
        QMenuBar { background: #FFFFFF; border-bottom: 1px solid #DDD; }
        QStatusBar { background: #FFFFFF; border-top: 1px solid #DDD; }
    """)

    window = MainWindow()

    if len(sys.argv) > 1:
        path = sys.argv[1]
        try:
            count = window._store.load_file(path)
            if count > 0:
                agv_ids = window._store.get_agv_ids()
                window._visible_agvs = set(agv_ids)
                window._selected_agv = agv_ids[0] if agv_ids else ""
                window._refresh_agv_filter(agv_ids)
                window._timeline.set_range(count - 1)
                window._timeline.set_order_markers(window._store.order_change_indices)
                window._timeline.set_index(0)
                window._on_index_changed(0)
                window.statusBar().showMessage(f"Loaded {count} entries from {path} ({len(agv_ids)} AGV(s))")
        except Exception as exc:
            print(f"Error loading {path}: {exc}", file=sys.stderr)

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
