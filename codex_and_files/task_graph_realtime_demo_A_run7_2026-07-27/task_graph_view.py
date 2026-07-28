from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from typing import Any


class TaskGraphView:
    """Compact, deterministic Task Graph renderer for the live demo."""

    IMMEDIATE_COLOR = "#B99CFF"
    MODEL_COLORS = {
        "m0": "#F5C451",
        "m3": "#3DDC97",
        "e2e": "#46A0FF",
    }
    MODEL_LABELS = {
        "m0": "M0",
        "m3": "M3",
        "e2e": "E2E",
    }

    def __init__(
        self,
        parent: tk.Widget,
        task_graph_path: Path,
        *,
        colors: dict[str, str],
    ) -> None:
        self.colors = colors
        graph = json.loads(task_graph_path.read_text(encoding="utf-8"))
        self.nodes = {
            int(node["node_idx"]): node
            for node in graph["nodes"]
        }
        self.atomic_sequences = [
            {
                "sequence_id": str(sequence["sequence_id"]),
                "nodes": [int(node_idx) for node_idx in sequence["nodes"]],
            }
            for sequence in graph.get("atomic_sequences", [])
            if len(sequence.get("nodes", [])) >= 2
        ]
        self.node_to_atomic_sequence = {
            node_idx: sequence
            for sequence in self.atomic_sequences
            for node_idx in sequence["nodes"]
        }
        self.edges = [
            (int(previous), int(node["node_idx"]))
            for node in graph["nodes"]
            for previous in node["execution_constraints"]["direct_must_previous_nodes"]
        ]
        self.immediate_edges = {
            (
                int(node["execution_constraints"]["must_immediately_previous_node"]),
                int(node["node_idx"]),
            )
            for node in graph["nodes"]
            if node["execution_constraints"]["must_immediately_previous_node"] is not None
        }

        self.completed_true_nodes: set[int] = set()
        self.completed_node_colors: dict[int, str] = {}
        self.current_true_node: int | None = None
        self.current_predictions: dict[str, int] = {}
        self.current_confidences: dict[str, float] = {}
        self.node_centers: dict[int, tuple[float, float]] = {}
        self.node_bounds: dict[int, tuple[float, float, float, float]] = {}
        self._redraw_job: str | None = None
        self._current_summary = "Waiting for the first completed action"

        self.frame = tk.Frame(
            parent,
            bg=colors["panel"],
            highlightbackground=colors["border"],
            highlightthickness=1,
            bd=0,
        )
        header = tk.Frame(self.frame, bg=colors["panel"])
        header.pack(fill="x", padx=12, pady=(8, 1))
        tk.Label(
            header,
            text="LIVE TASK GRAPH",
            bg=colors["panel"],
            fg=colors["muted"],
            font=("Segoe UI Semibold", 9),
        ).pack(side="left")
        tk.Label(
            header,
            text="[  ] = must execute consecutively",
            bg=colors["panel"],
            fg=self.IMMEDIATE_COLOR,
            font=("Segoe UI Semibold", 8),
        ).pack(side="left", padx=(14, 0))
        tk.Label(
            header,
            text="green = all correct   •   yellow = M3 correct/mixed   •   red = M3 wrong",
            bg=colors["panel"],
            fg=colors["muted"],
            font=("Segoe UI", 8),
        ).pack(side="right")

        self.canvas = tk.Canvas(
            self.frame,
            bg=colors["panel"],
            highlightthickness=0,
            bd=0,
            height=150,
        )
        self.canvas.pack(fill="both", expand=True, padx=7)
        self.detail = tk.Label(
            self.frame,
            text=self._current_summary,
            bg=colors["panel"],
            fg=colors["text"],
            font=("Segoe UI", 8),
            anchor="w",
        )
        self.detail.pack(fill="x", padx=12, pady=(0, 7))

        self.canvas.bind("<Configure>", self._schedule_redraw)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._on_leave)

    def grid(self, **kwargs: Any) -> None:
        self.frame.grid(**kwargs)

    def reset(self) -> None:
        self.completed_true_nodes.clear()
        self.completed_node_colors.clear()
        self.current_true_node = None
        self.current_predictions.clear()
        self.current_confidences.clear()
        self._current_summary = "Waiting for the first completed action"
        self.detail.configure(text=self._current_summary)
        self._draw()

    def clear_current_predictions(self, message: str) -> None:
        self.current_true_node = None
        self.current_predictions.clear()
        self.current_confidences.clear()
        self._current_summary = message
        self.detail.configure(text=message)
        self._draw()

    def show_result(self, result: dict[str, Any]) -> None:
        true_node = int(result["true_node_idx"])
        self.completed_true_nodes.add(true_node)
        if not bool(result["m3"]["correct"]):
            outcome_color = self.colors["red"]
        elif bool(result["m0"]["correct"]) and bool(result["e2e"]["correct"]):
            outcome_color = self.colors["green"]
        else:
            outcome_color = self.colors["amber"]
        self.completed_node_colors[true_node] = outcome_color
        self.current_true_node = true_node
        self.current_predictions = {
            key: int(result[key]["pred_node_idx"])
            for key in ("m0", "m3", "e2e")
        }
        self.current_confidences = {
            key: float(result[key]["confidence"])
            for key in ("m0", "m3", "e2e")
        }
        self._current_summary = (
            f"Action {result['annotation_row_index']}  •  GT N{true_node}  •  "
            f"M0 N{self.current_predictions['m0']}  •  "
            f"M3 N{self.current_predictions['m3']}  •  "
            f"E2E N{self.current_predictions['e2e']}"
        )
        self.detail.configure(text=self._current_summary)
        self._draw()

    def _schedule_redraw(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        if self._redraw_job is not None:
            self.canvas.after_cancel(self._redraw_job)
        self._redraw_job = self.canvas.after_idle(self._draw)

    def _layout(self, width: int, height: int) -> dict[int, tuple[float, float]]:
        left = 36.0
        right = max(left + 100.0, float(width) - 25.0)
        top = 43.0
        bottom = max(top + 80.0, float(height) - 18.0)
        usable_height = bottom - top
        lane_y = {
            1: top + usable_height * 0.08,
            2: top + usable_height * 0.50,
            3: top + usable_height * 0.92,
        }

        positions: dict[int, tuple[float, float]] = {}

        def place_row(indices: list[int], y: float, x0: float, x1: float) -> None:
            if len(indices) == 1:
                positions[indices[0]] = ((x0 + x1) / 2.0, y)
                return
            for offset, node_idx in enumerate(indices):
                x = x0 + (x1 - x0) * offset / (len(indices) - 1)
                positions[node_idx] = (x, y)

        place_row(list(range(1, 12)), lane_y[1], left + 22, right)
        place_row(list(range(12, 26)), lane_y[2], left, right)
        place_row(list(range(26, 36)), lane_y[3], left, right - 24)
        positions[0] = (left - 22, lane_y[1])
        positions[36] = (right, lane_y[3])
        return positions

    def _draw(self) -> None:
        self._redraw_job = None
        self.canvas.delete("all")
        width = max(300, self.canvas.winfo_width())
        height = max(135, self.canvas.winfo_height())
        self.node_centers = self._layout(width, height)
        self.node_bounds.clear()

        for stage, y_fraction in ((1, 0.04), (2, 0.46), (3, 0.88)):
            y = 43 + (max(80.0, height - 61.0) * y_fraction)
            self.canvas.create_text(
                4,
                y,
                text=f"S{stage}",
                anchor="w",
                fill=self.colors["muted"],
                font=("Segoe UI Semibold", 7),
            )

        for source, target in self.edges:
            if source not in self.node_centers or target not in self.node_centers:
                continue
            x1, y1 = self.node_centers[source]
            x2, y2 = self.node_centers[target]
            immediate_edge = (source, target) in self.immediate_edges
            completed_edge = (
                source in self.completed_true_nodes
                and target in self.completed_true_nodes
            )
            if immediate_edge:
                edge_color = self.IMMEDIATE_COLOR
                edge_width = 3 if completed_edge else 2
            else:
                edge_color = "#4B7893" if completed_edge else self.colors["border"]
                edge_width = 2 if completed_edge else 1
            self.canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                fill=edge_color,
                width=edge_width,
                arrow=tk.LAST,
                arrowshape=(5, 6, 2),
            )

        radius_x = 12
        radius_y = 10
        self._draw_atomic_sequence_brackets(height, radius_x, radius_y)
        for node_idx, (x, y) in self.node_centers.items():
            completed = node_idx in self.completed_true_nodes
            current = node_idx == self.current_true_node
            fill = (
                self.completed_node_colors.get(node_idx, "#245A76")
                if completed
                else self.colors["panel_alt"]
            )
            outline = self.colors["text"] if current else (
                "#5EA6C8" if completed else self.colors["border"]
            )
            width_px = 3 if current else (2 if completed else 1)
            bounds = (x - radius_x, y - radius_y, x + radius_x, y + radius_y)
            self.node_bounds[node_idx] = bounds
            self.canvas.create_oval(
                *bounds,
                fill=fill,
                outline=outline,
                width=width_px,
            )
            self.canvas.create_text(
                x,
                y,
                text=f"N{node_idx}",
                fill=(
                    "#07111E"
                    if completed
                    else (self.colors["text"] if current else self.colors["muted"])
                ),
                font=("Segoe UI Semibold", 7),
            )

        markers_by_node: dict[int, list[str]] = {}
        for model_key in ("m0", "m3", "e2e"):
            node_idx = self.current_predictions.get(model_key)
            if node_idx is not None:
                markers_by_node.setdefault(node_idx, []).append(model_key)

        for node_idx, model_keys in markers_by_node.items():
            if node_idx not in self.node_centers:
                continue
            x, y = self.node_centers[node_idx]
            for stack_index, model_key in enumerate(model_keys):
                marker_y = y - radius_y - 7 - stack_index * 12
                label = self.MODEL_LABELS[model_key]
                marker_width = 23 if label != "E2E" else 27
                self.canvas.create_rectangle(
                    x - marker_width / 2,
                    marker_y - 5,
                    x + marker_width / 2,
                    marker_y + 5,
                    fill=self.MODEL_COLORS[model_key],
                    outline="",
                )
                self.canvas.create_text(
                    x,
                    marker_y,
                    text=label,
                    fill="#07111E",
                    font=("Segoe UI Semibold", 6),
                )

    def _draw_atomic_sequence_brackets(
        self,
        canvas_height: int,
        radius_x: int,
        radius_y: int,
    ) -> None:
        show_labels = canvas_height >= 205
        for sequence in self.atomic_sequences:
            centers = [
                self.node_centers[node_idx]
                for node_idx in sequence["nodes"]
                if node_idx in self.node_centers
            ]
            if len(centers) < 2:
                continue
            y_values = [center[1] for center in centers]
            if max(y_values) - min(y_values) > 2:
                continue
            x_start = min(center[0] for center in centers) - radius_x
            x_end = max(center[0] for center in centers) + radius_x
            bracket_y = centers[0][1] + radius_y + 8
            tick_height = 6
            self.canvas.create_line(
                x_start,
                bracket_y - tick_height,
                x_start,
                bracket_y,
                x_end,
                bracket_y,
                x_end,
                bracket_y - tick_height,
                fill=self.IMMEDIATE_COLOR,
                width=2,
            )
            if show_labels:
                self.canvas.create_text(
                    (x_start + x_end) / 2,
                    bracket_y + 7,
                    text="IMMEDIATE",
                    fill=self.IMMEDIATE_COLOR,
                    font=("Segoe UI Semibold", 6),
                )

    def _on_motion(self, event: tk.Event[tk.Misc]) -> None:
        hovered: int | None = None
        for node_idx, (x1, y1, x2, y2) in self.node_bounds.items():
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                hovered = node_idx
                break
        if hovered is None:
            self.detail.configure(text=self._current_summary)
            return
        node = self.nodes[hovered]
        stage = node.get("stage_id", -1)
        label = node.get("action_label_tier3", node["node_id"])
        prediction_parts = [
            f"{self.MODEL_LABELS[key]} {self.current_confidences[key]:.3f}"
            for key, predicted_node in self.current_predictions.items()
            if predicted_node == hovered
        ]
        suffix = f"  •  {' / '.join(prediction_parts)}" if prediction_parts else ""
        atomic_sequence = self.node_to_atomic_sequence.get(hovered)
        if atomic_sequence is not None:
            sequence_nodes = "→".join(
                f"N{node_idx}" for node_idx in atomic_sequence["nodes"]
            )
            suffix += f"  •  consecutive [{sequence_nodes}]"
        self.detail.configure(
            text=f"N{hovered}  •  Stage {stage}  •  {label}{suffix}"
        )

    def _on_leave(self, _event: tk.Event[tk.Misc]) -> None:
        self.detail.configure(text=self._current_summary)
