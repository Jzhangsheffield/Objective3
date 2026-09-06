from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class DetectedSegment:
    start_index: int
    end_index: int
    emitted_at_index: int
    start_score: float
    end_score: float


class CausalBoundaryStateMachine:
    """Online decoder with debounce and a real pending-segment merge buffer."""

    def __init__(
        self,
        start_threshold: float,
        end_threshold: float,
        action_threshold: float,
        start_debounce_steps: int,
        end_debounce_steps: int,
        min_action_steps: int,
        merge_gap_steps: int,
    ):
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.action_threshold = action_threshold
        self.start_debounce = start_debounce_steps
        self.end_debounce = end_debounce_steps
        self.min_action_steps = min_action_steps
        self.merge_gap = merge_gap_steps
        self.reset()

    def reset(self) -> None:
        self.mode = "background"
        self.start_candidate = -1
        self.start_count = 0
        self.end_count = 0
        self.current_start = -1
        self.current_start_score = 0.0
        self.pending: DetectedSegment | None = None

    def update(self, index: int, action_probability: float, start_probability: float, end_probability: float) -> list[DetectedSegment]:
        emitted: list[DetectedSegment] = []
        start_evidence = start_probability >= self.start_threshold or action_probability >= self.action_threshold
        if self.mode == "background":
            if start_evidence:
                if self.start_count == 0:
                    self.start_candidate = index
                    self.current_start_score = start_probability
                self.start_count += 1
                self.current_start_score = max(self.current_start_score, start_probability)
                if self.start_count >= self.start_debounce:
                    if self.pending is not None and self.start_candidate - self.pending.end_index - 1 <= self.merge_gap:
                        self.current_start = self.pending.start_index
                        self.current_start_score = max(self.current_start_score, self.pending.start_score)
                        self.pending = None
                    else:
                        if self.pending is not None:
                            self.pending.emitted_at_index = index
                            emitted.append(self.pending)
                            self.pending = None
                        self.current_start = self.start_candidate
                    self.mode = "action"
                    self.end_count = 0
            else:
                self.start_count = 0
                self.start_candidate = -1
                if self.pending is not None and index - self.pending.end_index - 1 > self.merge_gap:
                    self.pending.emitted_at_index = index
                    emitted.append(self.pending)
                    self.pending = None
            return emitted

        duration = index - self.current_start + 1
        end_evidence = end_probability >= self.end_threshold or action_probability < (1.0 - self.action_threshold)
        self.end_count = self.end_count + 1 if end_evidence else 0
        if self.end_count >= self.end_debounce and duration >= self.min_action_steps:
            end_index = max(self.current_start, index - self.end_debounce + 1)
            self.pending = DetectedSegment(
                self.current_start, end_index, index, self.current_start_score, end_probability
            )
            self.mode = "background"
            self.start_count = 0
            self.end_count = 0
            self.current_start = -1
        return emitted

    def flush(self, final_index: int) -> list[DetectedSegment]:
        result: list[DetectedSegment] = []
        if self.mode == "action" and self.current_start >= 0:
            result.append(DetectedSegment(self.current_start, final_index, final_index, self.current_start_score, 0.0))
        elif self.pending is not None:
            self.pending.emitted_at_index = final_index
            result.append(self.pending)
        self.reset()
        return result


def run_state_machine(state_probability, start_probability, end_probability, settings: dict) -> list[dict]:
    machine = CausalBoundaryStateMachine(**settings)
    segments: list[DetectedSegment] = []
    for index, values in enumerate(zip(state_probability, start_probability, end_probability)):
        segments.extend(machine.update(index, *(float(value) for value in values)))
    segments.extend(machine.flush(len(state_probability) - 1))
    return [asdict(segment) for segment in segments]
