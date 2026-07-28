from __future__ import annotations

import argparse
import json
import time
import tkinter as tk
from bisect import bisect_right
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from demo_core import (
    DEMO_ROOT,
    DemoData,
    InferenceEngine,
    load_demo_data,
    resolve,
    validate_all,
)
from display_video import BufferedVideoReader
from prepare_demo_metadata import prepare
from task_graph_view import TaskGraphView


COLORS = {
    "page": "#0B1220",
    "panel": "#111C2E",
    "panel_alt": "#16243A",
    "border": "#2A3B54",
    "text": "#EAF0F8",
    "muted": "#91A3BA",
    "blue": "#46A0FF",
    "green": "#3DDC97",
    "red": "#FF6B6B",
    "amber": "#F5C451",
    "background": "#6D7C91",
}


class DemoApp:
    def __init__(self, root: tk.Tk, data: DemoData) -> None:
        self.root = root
        self.data = data
        self.root.title(data.config["demo"]["title"])
        self.root.geometry("1580x960")
        self.root.minsize(1380, 860)
        self.root.configure(bg=COLORS["page"])

        self.relative_frame_times = [
            row["timestamp_seconds"] - data.first_timestamp_seconds
            for row in data.frames
        ]
        self.display_reader = BufferedVideoReader(
            resolve(data.config["paths"]["display_video"]),
            expected_frames=len(data.frames),
            buffer_frames=120,
        )
        self.display_width = int(data.config["display_video"]["width"])
        self.display_height = int(data.config["display_video"]["height"])
        self.displayed_frame_index = -1

        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="demo-inference")
        self.engine_future: Future[InferenceEngine] = self.executor.submit(
            InferenceEngine, data
        )
        self.engine: InferenceEngine | None = None
        self.pending: list[tuple[Future[dict[str, Any]], dict[str, Any]]] = []
        self.playing = False
        self.playback_elapsed = 0.0
        self.playback_wall_start = 0.0
        self.frame_cursor = -1
        self.active_segment_no: int | None = None
        self.submitted_segments: set[int] = set()
        self.photo: ImageTk.PhotoImage | None = None
        self.current_display_rgb: np.ndarray[Any, Any] | None = None
        self.video_resize_job: str | None = None
        self.current_result: dict[str, Any] | None = None

        self._build_style()
        self._build_ui()
        self._show_frame(0)
        self._on_segment_started(self.data.frames[0]["segment_no"])
        self.root.after(150, self._maximize_for_demo)
        self.root.after(80, self._poll_engine)
        self.root.after(25, self._tick)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "Demo.Horizontal.TProgressbar",
            troughcolor=COLORS["panel_alt"],
            background=COLORS["blue"],
            bordercolor=COLORS["panel_alt"],
            lightcolor=COLORS["blue"],
            darkcolor=COLORS["blue"],
        )
        style.configure(
            "Treeview",
            background=COLORS["panel"],
            foreground=COLORS["text"],
            fieldbackground=COLORS["panel"],
            bordercolor=COLORS["border"],
            rowheight=27,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Treeview.Heading",
            background=COLORS["panel_alt"],
            foreground=COLORS["text"],
            bordercolor=COLORS["border"],
            font=("Segoe UI Semibold", 9),
        )
        style.map("Treeview", background=[("selected", "#204D76")])

    def _maximize_for_demo(self) -> None:
        try:
            self.root.state("zoomed")
        except tk.TclError:
            pass

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg=COLORS["page"])
        header.pack(fill="x", padx=22, pady=(16, 10))
        tk.Label(
            header,
            text="TASK GRAPH HISTORY • REAL-TIME REPLAY",
            bg=COLORS["page"],
            fg=COLORS["blue"],
            font=("Segoe UI Semibold", 10),
        ).pack(anchor="w")
        tk.Label(
            header,
            text=(
                f"{self.data.config['demo']['participant']} / "
                f"{self.data.config['demo']['source_run']}   •   "
                f"camera {self.data.config['demo']['camera_id']}   •   "
                f"{self.data.config['demo']['train_scope'].replace('_', '-')} "
                f"seed {self.data.config['demo']['seed']}"
            ),
            bg=COLORS["page"],
            fg=COLORS["text"],
            font=("Segoe UI Semibold", 21),
        ).pack(anchor="w", pady=(2, 0))
        self.system_status = tk.Label(
            header,
            text="Buffering H.264 display video and loading M0, M3 and E2E…",
            bg=COLORS["page"],
            fg=COLORS["amber"],
            font=("Segoe UI", 10),
        )
        self.system_status.pack(anchor="w", pady=(4, 0))

        main = tk.Frame(self.root, bg=COLORS["page"])
        main.pack(fill="both", expand=True, padx=22)
        main.grid_columnconfigure(0, weight=5)
        main.grid_columnconfigure(1, weight=3)
        main.grid_rowconfigure(0, weight=1)

        video_panel = self._panel(main)
        video_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        video_panel.grid_rowconfigure(1, weight=1)
        video_panel.grid_columnconfigure(0, weight=1)
        tk.Label(
            video_panel,
            text="LIVE FRAME STREAM",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=("Segoe UI Semibold", 9),
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(11, 7))
        self.video_label = tk.Label(
            video_panel,
            bg="#030712",
            bd=0,
            highlightthickness=0,
        )
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=12)
        self.video_label.bind("<Configure>", self._schedule_video_redraw)
        self.stream_caption = tk.Label(
            video_panel,
            text="",
            bg=COLORS["panel"],
            fg=COLORS["text"],
            font=("Segoe UI", 10),
            anchor="w",
        )
        self.stream_caption.grid(row=2, column=0, sticky="ew", padx=14, pady=10)

        results_panel = tk.Frame(main, bg=COLORS["page"])
        results_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        results_panel.grid_columnconfigure(0, weight=1)
        results_panel.grid_rowconfigure(5, weight=1, minsize=135)

        status_panel = self._panel(results_panel)
        status_panel.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        tk.Label(
            status_panel,
            text="CURRENT SEGMENT",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=("Segoe UI Semibold", 9),
        ).pack(anchor="w", padx=14, pady=(11, 2))
        self.segment_title = tk.Label(
            status_panel,
            text="Background",
            bg=COLORS["panel"],
            fg=COLORS["background"],
            font=("Segoe UI Semibold", 18),
            wraplength=440,
            justify="left",
        )
        self.segment_title.pack(anchor="w", padx=14)
        self.segment_detail = tk.Label(
            status_panel,
            text="Supplied annotation • not sent to classifiers",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9),
        )
        self.segment_detail.pack(anchor="w", padx=14, pady=(3, 11))

        self.m0_card = self._prediction_card(
            results_panel,
            row=1,
            title="M0 PREDICTION • CURRENT FROZEN RGB FEATURE",
            accent=COLORS["amber"],
        )
        self.m3_card = self._prediction_card(
            results_panel,
            row=2,
            title="M3 PREDICTION • GRAPH-VALID HISTORY",
            accent=COLORS["green"],
        )
        self.e2e_card = self._prediction_card(
            results_panel,
            row=3,
            title="E2E-NODE-SCRATCH PREDICTION • CURRENT RGB ONLY",
            accent=COLORS["blue"],
        )

        truth_panel = self._panel(results_panel)
        truth_panel.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        tk.Label(
            truth_panel,
            text="REVEALED AFTER ALL THREE PREDICTIONS",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=("Segoe UI Semibold", 9),
        ).pack(anchor="w", padx=14, pady=(10, 2))
        self.truth_label = tk.Label(
            truth_panel,
            text="Ground truth hidden while action is in progress",
            bg=COLORS["panel"],
            fg=COLORS["text"],
            font=("Segoe UI", 10),
            wraplength=440,
            justify="left",
        )
        self.truth_label.pack(anchor="w", padx=14, pady=(0, 10))

        self.task_graph_view = TaskGraphView(
            results_panel,
            resolve(self.data.config["paths"]["task_graph"]),
            colors=COLORS,
        )
        self.task_graph_view.grid(
            row=5,
            column=0,
            sticky="nsew",
            pady=(8, 0),
        )

        controls = tk.Frame(self.root, bg=COLORS["page"])
        controls.pack(fill="x", padx=22, pady=(12, 8))
        self.play_button = tk.Button(
            controls,
            text="▶  Play 1×",
            command=self._toggle_play,
            state="disabled",
            bg=COLORS["blue"],
            fg="#07111E",
            activebackground="#76B8FF",
            activeforeground="#07111E",
            relief="flat",
            font=("Segoe UI Semibold", 10),
            padx=18,
            pady=7,
        )
        self.play_button.pack(side="left")
        self.restart_button = tk.Button(
            controls,
            text="↺  Restart",
            command=self._restart,
            state="disabled",
            bg=COLORS["panel_alt"],
            fg=COLORS["text"],
            activebackground=COLORS["border"],
            activeforeground=COLORS["text"],
            relief="flat",
            font=("Segoe UI Semibold", 10),
            padx=16,
            pady=7,
        )
        self.restart_button.pack(side="left", padx=(8, 14))
        self.time_label = tk.Label(
            controls,
            text="00:00.0 / 02:18.4",
            bg=COLORS["page"],
            fg=COLORS["text"],
            font=("Consolas", 10),
        )
        self.time_label.pack(side="right")
        self.progress = ttk.Progressbar(
            controls,
            style="Demo.Horizontal.TProgressbar",
            orient="horizontal",
            maximum=self.data.duration_seconds,
        )
        self.progress.pack(side="left", fill="x", expand=True)

        history_panel = self._panel(self.root)
        history_panel.pack(fill="both", padx=22, pady=(0, 8))
        tk.Label(
            history_panel,
            text="COMPLETED ACTION HISTORY",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=("Segoe UI Semibold", 9),
        ).pack(anchor="w", padx=12, pady=(9, 4))
        columns = ("event", "truth", "m0", "m3", "e2e", "latency")
        self.history_tree = ttk.Treeview(
            history_panel,
            columns=columns,
            show="headings",
            height=2,
        )
        headings = {
            "event": "#",
            "truth": "Completed ground truth",
            "m0": "M0 prediction",
            "m3": "M3 prediction",
            "e2e": "E2E prediction",
            "latency": "Inference",
        }
        widths = {
            "event": 45,
            "truth": 300,
            "m0": 245,
            "m3": 245,
            "e2e": 245,
            "latency": 90,
        }
        for column in columns:
            self.history_tree.heading(column, text=headings[column])
            self.history_tree.column(column, width=widths[column], anchor="w")
        self.history_tree.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.history_tree.tag_configure("correct", foreground=COLORS["green"])
        self.history_tree.tag_configure("mixed", foreground=COLORS["amber"])

    def _panel(self, parent: tk.Widget) -> tk.Frame:
        return tk.Frame(
            parent,
            bg=COLORS["panel"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
            bd=0,
        )

    def _prediction_card(
        self, parent: tk.Widget, row: int, title: str, accent: str
    ) -> dict[str, Any]:
        panel = self._panel(parent)
        panel.grid(row=row, column=0, sticky="ew", pady=3)
        tk.Frame(panel, height=3, bg=accent).pack(fill="x")
        tk.Label(
            panel,
            text=title,
            bg=COLORS["panel"],
            fg=accent,
            font=("Segoe UI Semibold", 9),
        ).pack(anchor="w", padx=12, pady=(5, 1))
        node = tk.Label(
            panel,
            text="Waiting for action segment",
            bg=COLORS["panel"],
            fg=COLORS["text"],
            font=("Segoe UI Semibold", 11),
            wraplength=440,
            justify="left",
        )
        node.pack(anchor="w", padx=12)
        verdict = tk.Label(
            panel,
            text="NO PREDICTION YET",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=("Segoe UI Semibold", 8),
            wraplength=440,
            justify="left",
        )
        verdict.pack(anchor="w", padx=12, pady=(2, 0))
        confidence = tk.Label(
            panel,
            text="—",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=("Segoe UI", 8),
            wraplength=440,
            justify="left",
        )
        confidence.pack(anchor="w", padx=12, pady=(2, 6))
        return {
            "node": node,
            "verdict": verdict,
            "confidence": confidence,
            "accent": accent,
        }

    def _poll_engine(self) -> None:
        if self.engine is None and self.engine_future.done():
            try:
                self.engine = self.engine_future.result()
            except Exception as error:
                self.system_status.configure(
                    text=f"Model loading failed: {error}", fg=COLORS["red"]
                )
                messagebox.showerror("Model loading failed", str(error))
                return
            self.system_status.configure(
                text=(
                    f"Ready • {self.engine.device_name} • buffered H.264 display "
                    "• supplied boundaries • 1× playback"
                ),
                fg=COLORS["green"],
            )
            self.play_button.configure(state="normal")
            self.restart_button.configure(state="normal")
        if self.engine is None:
            self.root.after(80, self._poll_engine)
            return
        self._poll_predictions()
        self.root.after(60, self._poll_engine)

    def _poll_predictions(self) -> None:
        remaining: list[tuple[Future[dict[str, Any]], dict[str, Any]]] = []
        for future, segment in self.pending:
            if not future.done():
                remaining.append((future, segment))
                continue
            try:
                result = future.result()
            except Exception as error:
                self.system_status.configure(
                    text=f"Inference failed on segment {segment['segment_no']}: {error}",
                    fg=COLORS["red"],
                )
                continue
            self._display_result(result)
        self.pending = remaining

    def _toggle_play(self) -> None:
        if self.engine is None:
            return
        if self.playing:
            self.playback_elapsed = time.perf_counter() - self.playback_wall_start
            self.playing = False
            self.play_button.configure(text="▶  Resume 1×")
        else:
            if self.playback_elapsed >= self.data.duration_seconds:
                self._restart()
            self.playback_wall_start = time.perf_counter() - self.playback_elapsed
            self.playing = True
            self.play_button.configure(text="❚❚  Pause")

    def _restart(self) -> None:
        if self.pending:
            messagebox.showinfo(
                "Please wait",
                "A prediction is still finishing. Restart again in a moment.",
            )
            return
        self.playing = False
        self.playback_elapsed = 0.0
        self.playback_wall_start = 0.0
        self.frame_cursor = -1
        self.displayed_frame_index = -1
        self.active_segment_no = None
        self.submitted_segments.clear()
        self.current_result = None
        if self.engine is not None:
            self.engine.reset()
        self.display_reader.reset()
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        self._clear_prediction_cards("Waiting for action segment")
        self.truth_label.configure(text="Ground truth hidden while action is in progress")
        self.task_graph_view.reset()
        self._show_frame(0)
        self._on_segment_started(self.data.frames[0]["segment_no"])
        self.play_button.configure(text="▶  Play 1×")
        self.progress["value"] = 0
        self.time_label.configure(text=f"00:00.0 / {self._format_time(self.data.duration_seconds)}")

    def _tick(self) -> None:
        if self.playing:
            target = min(
                time.perf_counter() - self.playback_wall_start,
                self.data.duration_seconds,
            )
            target_frame_index = max(
                0,
                min(
                    len(self.data.frames) - 1,
                    bisect_right(self.relative_frame_times, target) - 1,
                ),
            )
            if target_frame_index > self.frame_cursor:
                for next_index in range(self.frame_cursor + 1, target_frame_index + 1):
                    frame = self.data.frames[next_index]
                    if frame["segment_no"] != self.active_segment_no:
                        if self.active_segment_no is not None:
                            self._on_segment_finished(self.active_segment_no)
                        self._on_segment_started(frame["segment_no"])
                self.frame_cursor = target_frame_index
                self._show_frame(target_frame_index)
            self.playback_elapsed = target
            self.progress["value"] = target
            self.time_label.configure(
                text=f"{self._format_time(target)} / {self._format_time(self.data.duration_seconds)}"
            )
            if target >= self.data.duration_seconds:
                if self.active_segment_no is not None:
                    self._on_segment_finished(self.active_segment_no)
                self.playing = False
                self.play_button.configure(text="▶  Replay")
                self.system_status.configure(
                    text="Replay complete • all completed actions added once to M3 history",
                    fg=COLORS["green"],
                )
        self.root.after(12, self._tick)

    def _show_frame(self, index: int) -> None:
        index = max(0, min(index, len(self.data.frames) - 1))
        decoded = self.display_reader.latest_at_or_before(index)
        if decoded is None:
            if self.photo is not None:
                return
            row = self.data.frames[index]
            path = self.data.raw_frames_dir / row["frame_name"]
            with Image.open(path) as image:
                rgb = np.asarray(
                    image.convert("RGB").resize(
                        (self.display_width, self.display_height),
                        Image.Resampling.BILINEAR,
                    ),
                    dtype=np.uint8,
                )
            display_index = index
        else:
            display_index, rgb = decoded
        self.current_display_rgb = np.ascontiguousarray(rgb)
        self._render_current_video()
        self.displayed_frame_index = display_index
        row = self.data.frames[display_index]
        relative = row["timestamp_seconds"] - self.data.first_timestamp_seconds
        self.stream_caption.configure(
            text=(
                f"Frame {row['frame_idx']:,} / {len(self.data.frames):,}   "
                f"•   original #{row['original_frame_idx']:,}   "
                f"•   {relative:06.2f}s   •   H.264 display   •   {row['frame_name']}"
            )
        )

    def _schedule_video_redraw(self, _event: tk.Event[tk.Misc]) -> None:
        if self.video_resize_job is not None:
            self.root.after_cancel(self.video_resize_job)
        self.video_resize_job = self.root.after(80, self._render_current_video)

    def _render_current_video(self) -> None:
        self.video_resize_job = None
        if self.current_display_rgb is None:
            return
        source_height, source_width = self.current_display_rgb.shape[:2]
        available_width = self.video_label.winfo_width()
        available_height = self.video_label.winfo_height()
        if available_width <= 10 or available_height <= 10:
            available_width = source_width
            available_height = source_height
        scale = min(
            available_width / source_width,
            available_height / source_height,
        )
        target_width = max(2, int(round(source_width * scale)))
        target_height = max(2, int(round(source_height * scale)))
        if (target_width, target_height) == (source_width, source_height):
            display_rgb = self.current_display_rgb
        else:
            display_rgb = cv2.resize(
                self.current_display_rgb,
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR,
            )
        self.photo = ImageTk.PhotoImage(Image.fromarray(display_rgb))
        self.video_label.configure(image=self.photo)

    def _on_segment_started(self, segment_no: int) -> None:
        self.active_segment_no = int(segment_no)
        segment = self.data.segments_by_no[self.active_segment_no]
        if segment["is_background"]:
            self.segment_title.configure(text="Background", fg=COLORS["background"])
            self.segment_detail.configure(
                text=(
                    f"Segment {segment_no} • supplied annotation • "
                    f"{segment['current_frame_count']} frames • not classified"
                )
            )
        else:
            self.segment_title.configure(
                text=f"Action {segment['annotation_row_index']} in progress",
                fg=COLORS["amber"],
            )
            self.segment_detail.configure(
                text=(
                    f"Stage {segment['stage_id']} • {segment['current_frame_count']} frames • "
                    "prediction will run after segment end"
                )
            )
            self._clear_prediction_cards("Waiting for action to finish…")
            self.truth_label.configure(
                text="Ground truth hidden while action is in progress"
            )
            self.task_graph_view.clear_current_predictions(
                f"Action {segment['annotation_row_index']} in progress • predictions hidden"
            )

    def _on_segment_finished(self, segment_no: int) -> None:
        segment = self.data.segments_by_no[int(segment_no)]
        if (
            segment["is_background"]
            or int(segment_no) in self.submitted_segments
            or self.engine is None
        ):
            return
        self.submitted_segments.add(int(segment_no))
        self.segment_title.configure(
            text=f"Processing action {segment['annotation_row_index']}…",
            fg=COLORS["amber"],
        )
        future = self.executor.submit(self.engine.predict, segment)
        self.pending.append((future, segment))

    def _clear_prediction_cards(self, text: str) -> None:
        for card in (self.m0_card, self.m3_card, self.e2e_card):
            card["node"].configure(text=text, fg=COLORS["text"])
            card["verdict"].configure(text="NO PREDICTION YET", fg=COLORS["muted"])
            card["confidence"].configure(text="—", fg=COLORS["muted"])

    def _display_result(self, result: dict[str, Any]) -> None:
        self.current_result = result
        for key, card in (
            ("m0", self.m0_card),
            ("m3", self.m3_card),
            ("e2e", self.e2e_card),
        ):
            model = result[key]
            card["node"].configure(
                text=(
                    f"Action {result['annotation_row_index']} • "
                    f"Predicted Node {model['pred_node_idx']}\n"
                    f"{model['pred_node_id']}"
                ),
                fg=card["accent"],
            )
            if model["correct"]:
                verdict = (
                    f"CORRECT • matches Ground-truth Node {result['true_node_idx']}"
                )
                verdict_color = COLORS["green"]
            else:
                verdict = (
                    f"INCORRECT • Ground truth is Node {result['true_node_idx']}"
                )
                verdict_color = COLORS["red"]
            card["verdict"].configure(text=verdict, fg=verdict_color)
            top3_parts = []
            for row in model["top3"]:
                occurrence_suffix = (
                    f"#{row['occurrence']}"
                    if row["occurrence"] is not None
                    else ""
                )
                top3_parts.append(
                    f"N{row['node_idx']}{occurrence_suffix}:{row['confidence']:.2f}"
                )
            top3 = "   ".join(top3_parts)
            occurrence = (
                f" • occurrence {model['occurrence']}"
                if model["occurrence"] is not None
                else ""
            )
            card["confidence"].configure(
                text=(
                    f"Tier-3: {model['label']}{occurrence}\n"
                    f"confidence {model['confidence']:.3f}   •   "
                    f"top-3  {top3}"
                ),
                fg=COLORS["muted"],
            )
        truth_occurrence = (
            f" • occurrence {result['true_occurrence']}"
            if result["true_occurrence"] is not None
            else ""
        )
        self.truth_label.configure(
            text=(
                f"Action {result['annotation_row_index']} • "
                f"Ground-truth Node {result['true_node_idx']}\n"
                f"{result['true_node_id']}\n"
                f"Tier-3: {result['true_label']}{truth_occurrence}\n"
                f"Stage {result['stage_id']} • "
                f"History before current: {result['history_length_before_current']} completed actions"
            )
        )
        self.task_graph_view.show_result(result)
        m0_text = (
            f"N{result['m0']['pred_node_idx']} "
            f"{'CORRECT' if result['m0']['correct'] else 'WRONG'} "
            f"({result['m0']['confidence']:.2f})"
        )
        m3_text = (
            f"N{result['m3']['pred_node_idx']} "
            f"{'CORRECT' if result['m3']['correct'] else 'WRONG'} "
            f"({result['m3']['confidence']:.2f})"
        )
        e2e_text = (
            f"N{result['e2e']['pred_node_idx']} "
            f"{'CORRECT' if result['e2e']['correct'] else 'WRONG'} "
            f"({result['e2e']['confidence']:.2f})"
        )
        tag = (
            "correct"
            if result["m0"]["correct"]
            and result["m3"]["correct"]
            and result["e2e"]["correct"]
            else "mixed"
        )
        item = self.history_tree.insert(
            "",
            "end",
            values=(
                result["annotation_row_index"],
                result["true_display_name"],
                m0_text,
                m3_text,
                e2e_text,
                f"{result['inference_ms']:.0f} ms",
            ),
            tags=(tag,),
        )
        self.history_tree.see(item)
        self.system_status.configure(
            text=(
                f"Action {result['annotation_row_index']} complete • "
                f"M0 predicted N{result['m0']['pred_node_idx']} "
                f"({'correct' if result['m0']['correct'] else 'incorrect'}) • "
                f"M3 predicted N{result['m3']['pred_node_idx']} "
                f"({'correct' if result['m3']['correct'] else 'incorrect'}) • "
                f"E2E predicted N{result['e2e']['pred_node_idx']} "
                f"({'correct' if result['e2e']['correct'] else 'incorrect'}) • "
                f"{result['inference_ms']:.0f} ms"
            ),
            fg=COLORS["green"] if result["m3"]["correct"] else COLORS["amber"],
        )

    @staticmethod
    def _format_time(seconds: float) -> str:
        minutes = int(seconds // 60)
        remainder = seconds - minutes * 60
        return f"{minutes:02d}:{remainder:04.1f}"

    def _close(self) -> None:
        self.playing = False
        self.display_reader.close()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def run_validation(config_path: Path) -> None:
    prepare(config_path)
    data = load_demo_data(config_path)
    print("Loading models…", flush=True)
    engine = InferenceEngine(data)
    action_count = sum(not row["is_background"] for row in data.segments)
    print(
        f"Running {action_count} action segments for "
        f"{data.config['demo']['participant']}/{data.config['demo']['source_run']} "
        f"on {engine.device_name}…",
        flush=True,
    )
    summary = validate_all(data, engine)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="M0/M3/E2E task-graph real-time demo")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEMO_ROOT / "config.json",
        help="Demo profile config JSON",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run all actions without opening the GUI and compare existing predictions",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.validate:
        run_validation(config_path)
        return
    prepare(config_path)
    data = load_demo_data(config_path)
    root = tk.Tk()
    DemoApp(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
