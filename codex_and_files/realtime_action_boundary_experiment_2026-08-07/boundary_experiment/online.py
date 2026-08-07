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
    """Online BACKGROUND/CANDIDATE/ACTION state machine; no future frames are used."""

    def __init__(
        self,
        start_threshold: float = 0.55,
        end_threshold: float = 0.55,
        action_threshold: float = 0.55,
        start_debounce: int = 2,
        end_debounce: int = 2,
        min_action_steps: int = 3,
        merge_gap_steps: int = 0,
    ):
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.action_threshold = action_threshold
        self.start_debounce = start_debounce
        self.end_debounce = end_debounce
        self.min_action_steps = min_action_steps
        self.merge_gap_steps = merge_gap_steps
        self.reset()

    def reset(self) -> None:
        self.mode = "background"
        self.start_candidate = -1
        self.start_count = 0
        self.end_count = 0
        self.current_start = -1
        self.current_start_score = 0.0
        self.last_segment: DetectedSegment | None = None

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
                    self.mode = "action"
                    self.current_start = self.start_candidate
                    self.end_count = 0
            else:
                self.start_count = 0
                self.start_candidate = -1
            return emitted

        duration = index - self.current_start + 1
        end_evidence = end_probability >= self.end_threshold or action_probability < (1.0 - self.action_threshold)
        self.end_count = self.end_count + 1 if end_evidence else 0
        if self.end_count >= self.end_debounce and duration >= self.min_action_steps:
            end_index = index - self.end_debounce + 1
            segment = DetectedSegment(
                start_index=self.current_start,
                end_index=max(self.current_start, end_index),
                emitted_at_index=index,
                start_score=self.current_start_score,
                end_score=end_probability,
            )
            if self.last_segment and self.merge_gap_steps > 0 and segment.start_index - self.last_segment.end_index - 1 <= self.merge_gap_steps:
                segment.start_index = self.last_segment.start_index
                emitted.append(segment)
            else:
                emitted.append(segment)
            self.last_segment = segment
            self.mode = "background"
            self.start_count = 0
            self.end_count = 0
            self.current_start = -1
        return emitted

    def flush(self, final_index: int) -> list[DetectedSegment]:
        if self.mode != "action" or self.current_start < 0:
            return []
        segment = DetectedSegment(
            start_index=self.current_start,
            end_index=final_index,
            emitted_at_index=final_index,
            start_score=self.current_start_score,
            end_score=0.0,
        )
        self.reset()
        return [segment]


def run_state_machine(state_probability, start_probability, end_probability, settings: dict) -> list[dict]:
    machine = CausalBoundaryStateMachine(**settings)
    segments: list[DetectedSegment] = []
    for index, values in enumerate(zip(state_probability, start_probability, end_probability)):
        segments.extend(machine.update(index, *(float(value) for value in values)))
    segments.extend(machine.flush(len(state_probability) - 1))
    return [asdict(segment) for segment in segments]
