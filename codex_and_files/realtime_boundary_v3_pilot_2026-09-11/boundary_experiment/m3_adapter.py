from __future__ import annotations

import sys
import json
from pathlib import Path

import torch


class OnlineTaskGraphState:
    def __init__(self, task_graph_path: str | Path):
        with Path(task_graph_path).open("r", encoding="utf-8-sig") as handle:
            graph = json.load(handle)
        self.nodes = {int(row["node_idx"]): row for row in graph["nodes"] if int(row["node_idx"]) > 0}
        self.predicted_history: list[int] = []

    def audit_and_update(self, node_idx: int) -> dict:
        node = self.nodes[node_idx]
        constraints = node.get("execution_constraints", {})
        required = set(int(x) for x in constraints.get("direct_must_previous_nodes", []) if int(x) > 0)
        immediate = constraints.get("must_immediately_previous_node")
        missing = sorted(required - set(self.predicted_history))
        immediate_valid = immediate in (None, 0) or (bool(self.predicted_history) and self.predicted_history[-1] == int(immediate))
        result = {
            "graph_valid": not missing and immediate_valid,
            "missing_direct_predecessors": missing,
            "immediate_predecessor_valid": immediate_valid,
        }
        self.predicted_history.append(node_idx)
        return result

    def reset(self) -> None:
        self.predicted_history.clear()


class M3AtomicTailOnlineAdapter:
    """M3 Direct Fusion inference using only earlier predicted segment features."""

    def __init__(self, project_root: str | Path, checkpoint: str | Path, device: torch.device, max_history: int = 35, task_graph_path: str | Path | None = None):
        root = str(Path(project_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        from graph_history.models import build_direct_context_model
        from graph_history.utils import load_compatible_state
        self.device = device
        self.max_history = max_history
        self.model = build_direct_context_model(
            model_name="m3_direct", feature_dim=512, d_model=256, num_heads=4,
            max_history=max_history, dropout=0.1,
        ).to(device)
        report = load_compatible_state(self.model, checkpoint)
        if report["missing_keys"] or report["unexpected_keys"] or report["loaded_keys"] != report["model_keys"]:
            raise RuntimeError(f"M3 checkpoint is not an exact architecture match: {report}")
        self.model.eval()
        self.history_features: list[torch.Tensor] = []
        self.history_node_indices: list[int] = []
        self.graph_state = OnlineTaskGraphState(task_graph_path) if task_graph_path else None
        self.load_report = report

    @torch.inference_mode()
    def predict(self, segment_feature: torch.Tensor) -> dict:
        current = segment_feature.detach().float().reshape(1, -1).to(self.device)
        history = self.history_features[-self.max_history :]
        if history:
            history_tensor = torch.stack(history).unsqueeze(0).to(self.device)
        else:
            history_tensor = torch.zeros(1, 0, current.shape[-1], device=self.device)
        length = history_tensor.shape[1]
        positions = torch.arange(1, length + 1, device=self.device).unsqueeze(0)
        padding = torch.zeros(1, length, dtype=torch.bool, device=self.device)
        logits, _ = self.model(current, history_tensor, positions, padding)
        probabilities = torch.softmax(logits, dim=-1)[0]
        node_zero_based = int(torch.argmax(probabilities))
        self.history_features.append(current[0].cpu())
        self.history_node_indices.append(node_zero_based + 1)
        result = {
            "node_idx": node_zero_based + 1,
            "confidence": float(probabilities[node_zero_based]),
            "history_length": length,
        }
        if self.graph_state:
            result.update(self.graph_state.audit_and_update(node_zero_based + 1))
        return result

    def reset(self) -> None:
        self.history_features.clear()
        self.history_node_indices.clear()
        if self.graph_state:
            self.graph_state.reset()
