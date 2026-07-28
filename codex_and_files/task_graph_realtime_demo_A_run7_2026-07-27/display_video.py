from __future__ import annotations

import threading
from collections import deque
from pathlib import Path
from typing import Any

import cv2


class BufferedVideoReader:
    """Sequential H.264 decoder with a bounded RGB frame buffer."""

    def __init__(
        self,
        path: Path,
        expected_frames: int,
        buffer_frames: int = 120,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Display video is missing: {self.path}. "
                "Run build_display_video.bat first."
            )
        self.expected_frames = int(expected_frames)
        self.buffer_frames = max(30, int(buffer_frames))
        self._condition = threading.Condition()
        self._frames: deque[tuple[int, Any]] = deque()
        self._stop = False
        self._seek_target: int | None = 0
        self._error: Exception | None = None
        self._thread = threading.Thread(
            target=self._decode_loop,
            name="demo-display-video-decoder",
            daemon=True,
        )
        self._thread.start()

    def _open_capture(self, frame_index: int) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(str(self.path), cv2.CAP_FFMPEG)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open display video: {self.path}")
        if frame_index:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        return capture

    def _decode_loop(self) -> None:
        capture: cv2.VideoCapture | None = None
        next_index = 0
        try:
            while True:
                with self._condition:
                    if self._stop:
                        return
                    if self._seek_target is not None:
                        next_index = self._seek_target
                        self._seek_target = None
                        self._frames.clear()
                        if capture is not None:
                            capture.release()
                        capture = self._open_capture(next_index)
                    while len(self._frames) >= self.buffer_frames and not self._stop:
                        self._condition.wait(timeout=0.1)
                        if self._seek_target is not None:
                            break
                    if self._stop:
                        return
                    if self._seek_target is not None:
                        continue

                assert capture is not None
                ok, bgr = capture.read()
                if not ok:
                    with self._condition:
                        self._condition.wait(timeout=0.1)
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                with self._condition:
                    if self._seek_target is not None:
                        continue
                    self._frames.append((next_index, rgb))
                    next_index += 1
                    self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self._error = error
                self._condition.notify_all()
        finally:
            if capture is not None:
                capture.release()

    def latest_at_or_before(self, target_index: int) -> tuple[int, Any] | None:
        """Return the newest decoded frame up to target, discarding older frames."""
        with self._condition:
            if self._error is not None:
                raise RuntimeError("Display video decoder failed") from self._error
            selected: tuple[int, Any] | None = None
            while self._frames and self._frames[0][0] <= target_index:
                selected = self._frames.popleft()
            if selected is not None:
                self._condition.notify_all()
            return selected

    def reset(self) -> None:
        with self._condition:
            self._seek_target = 0
            self._frames.clear()
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        self._thread.join(timeout=2.0)

    @property
    def buffered_frames(self) -> int:
        with self._condition:
            return len(self._frames)
