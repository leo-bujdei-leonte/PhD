#!/usr/bin/env python
"""Step through a model's UCS-Bench answers next to the video.

Usage example:
    python ./ucs-bench/inspect_generations.py ../UCS-Bench/data/UCS-Bench-results/subset_directme_qwen3vl.json
    python ./ucs-bench/inspect_generations.py <gen.json> --root ../UCS-Bench/data/UCS-Bench --video 01169

<gen.json> is the output of evaluate_ucsbench_vlm.py; --root is the UCS-Bench
dataset (videos + QAs/, see inspect_ucsbench.py). The window first lists the
videos in the generation file with the model's accuracy on each; double-click
one (or --video) to play it next to a panel with the upcoming question, its
options, the option the model picked and the correct one.

Playback stops at every question timestamp and the panel moves to that
question. When several questions share a timestamp, Play shows the next one
(still paused) until all of them have been shown, then resumes.

Controls (buttons, or keys)
---------------------------
    space            play / pause (at a shared timestamp: next question)
    <- / ->          back / forward 10 s   (shift: 60 s)
    Home             back to the start
    [ / ]            slower / faster (0.25x .. 8x)
    p / n, up / dn   view the previous / next question without moving the
                     video (Play, a seek or the next stop returns the panel
                     to the upcoming question); clicking a row does the same
    click timeline   seek
    Esc              back to the video list

Needs OpenCV (pip install opencv-python) and Tk (part of most Python installs).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tkinter import font as tkfont
from tkinter import messagebox, ttk
from typing import Any

from inspect_ucsbench import DEFAULT_ROOT, Dataset, Generations, Question, fmt_ts, score

SPEEDS = (0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0)


# --------------------------------------------------------------------------- #
# Playback logic (no GUI, no decoding)
# --------------------------------------------------------------------------- #
class PlaybackState:
    """Playback position and question cursor.

    Positions are frame indices. A question fires on the first frame at or
    after its timestamp; questions on frames <= ``passed`` have been reached,
    so playing forward stops at every later one. ``questions`` must be sorted
    by timestamp with untimed questions last (as Dataset.questions returns).
    """

    def __init__(self, questions: list[Question], fps: float, n_frames: int) -> None:
        self.questions = questions
        self.fps = fps if fps > 0 and math.isfinite(fps) else 30.0
        self.n_frames = max(1, int(n_frames))
        self.qframes = self._question_frames()
        self.frame = 0
        self.passed = -1
        self.paused = True
        self.speed_i = SPEEDS.index(1.0)
        self._stop: int | None = None  # question shown at the current stop
        self._queue: list[int] = []  # further questions sharing that stop
        self._browse: int | None = None  # question picked with prev/next

    def _question_frames(self) -> list[int | None]:
        return [None if q.t is None else self.frame_of(q.t) for q in self.questions]

    def frame_of(self, t: float) -> int:
        f = math.ceil(t * self.fps - 1e-6)
        return min(max(f, 0), self.n_frames - 1)

    @property
    def t(self) -> float:
        return self.frame / self.fps

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps

    @property
    def speed(self) -> float:
        return SPEEDS[self.speed_i]

    @property
    def at_stop(self) -> bool:
        return self._stop is not None

    @property
    def remaining_at_stop(self) -> int:
        return len(self._queue)

    @property
    def browsing(self) -> bool:
        return self._browse is not None

    # ---- cursor ---------------------------------------------------------- #
    def upcoming(self) -> int | None:
        """First question not reached yet."""
        for i, f in enumerate(self.qframes):
            if f is not None and f > self.passed:
                return i
        return None

    def follow_cursor(self) -> int | None:
        """The question the panel shows when the user isn't browsing."""
        if self._stop is not None:
            return self._stop
        up = self.upcoming()
        if up is not None:
            return up
        timed = [i for i, f in enumerate(self.qframes) if f is not None]
        if timed:
            return timed[-1]
        return 0 if self.questions else None

    def cursor(self) -> int | None:
        return self._browse if self._browse is not None else self.follow_cursor()

    def status(self, i: int) -> str:
        f = self.qframes[i]
        if f is None:
            return "untimed"
        if f > self.frame:
            return "upcoming"
        return "now" if f == self.frame else "passed"

    # ---- actions --------------------------------------------------------- #
    def seek(self, frame: int) -> None:
        self.frame = min(max(int(frame), 0), self.n_frames - 1)
        self.passed = self.frame - 1  # a question on this very frame still fires
        self._stop, self._queue, self._browse = None, [], None

    def skip(self, seconds: float) -> None:
        self.seek(self.frame + round(seconds * self.fps))

    def _unreached_here(self) -> bool:
        return self.frame > self.passed and self.frame in self.qframes

    def _arrive(self, f: int) -> None:
        group = [i for i, qf in enumerate(self.qframes) if qf == f]
        self.frame = self.passed = f
        self.paused = True
        self._stop, self._queue, self._browse = group[0], group[1:], None

    def play_pause(self) -> None:
        if not self.paused:
            self.paused = True
            return
        self._browse = None
        if self._queue:  # several questions at this stop: show the next one
            self._stop = self._queue.pop(0)
            return
        if self.frame >= self.n_frames - 1 and not self._unreached_here():
            self.seek(0)  # at the end: play again from the start
        if self._unreached_here():  # parked on a question timestamp (after a seek)
            self._arrive(self.frame)
            return
        self._stop = None
        self.passed = self.frame
        self.paused = False

    def advance(self, n: int = 1) -> int:
        """Play forward up to n frames, stopping at the first question timestamp."""
        if self.paused or n < 1:
            return self.frame
        target = min(self.frame + n, self.n_frames - 1)
        hits = [f for f in self.qframes if f is not None and self.passed < f <= target]
        if hits:
            self._arrive(min(hits))
        else:
            self.frame = self.passed = target
            if target >= self.n_frames - 1:
                self.paused = True
        return self.frame

    def end_at(self, last_frame: int) -> None:
        """The decoder ran out before the reported frame count: shorten the video."""
        self.n_frames = max(1, last_frame + 1)
        self.qframes = self._question_frames()
        self.frame = min(self.frame, self.n_frames - 1)
        self.passed = min(self.passed, self.frame - 1)
        self.paused = True

    def faster(self) -> None:
        self.speed_i = min(self.speed_i + 1, len(SPEEDS) - 1)

    def slower(self) -> None:
        self.speed_i = max(self.speed_i - 1, 0)

    def browse(self, delta: int) -> None:
        cur = self.cursor()
        if cur is not None:
            self.select(cur + delta)

    def select(self, i: int) -> None:
        """Show question i in the panel without moving the video."""
        if not self.questions:
            return
        i = min(max(i, 0), len(self.questions) - 1)
        self._browse = None if i == self.follow_cursor() else i


# --------------------------------------------------------------------------- #
# Text shown in the GUI (pure functions, so they can be tested)
# --------------------------------------------------------------------------- #
@dataclass
class VideoRow:
    uid: str
    n_questions: int
    n_correct: int
    n_scored: int
    length: str
    has_video: bool

    @property
    def accuracy(self) -> float | None:
        return self.n_correct / self.n_scored if self.n_scored else None


def video_rows(ds: Dataset) -> list[VideoRow]:
    """One row per video in the generation file."""
    assert ds.gen is not None
    rows = []
    for uid in sorted(ds.gen.videos()):
        qs = ds.questions(uid)
        c, n = score(qs)
        length = str(qs[0].raw.get("video_length", "")).strip() if qs else ""
        rows.append(VideoRow(uid, len(qs), c, n, length, ds.video_path(uid).exists()))
    return rows


def _ts(t: float | None) -> str:
    return fmt_ts(t) if t is not None else "??:??:??"


def fmt_delta(secs: float) -> str:
    secs = max(0.0, secs)
    if secs < 60:
        return f"{secs:.1f} s"
    m, s = divmod(round(secs), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def picked_letter(q: Question) -> str | None:
    if q.pred is None:
        return None
    return (q.pred.letter or "").strip().upper() or None


def model_cell(q: Question) -> tuple[str, str]:
    """(model column text, row tag) for the question list."""
    p = q.pred
    if p is None:
        return "-", "none"
    letter = picked_letter(q) or "?"
    if p.correct is None:
        return letter, "none"
    return (f"{letter} ✓", "ok") if p.correct else (f"{letter} ✗", "bad")


def question_lines(q: Question, max_raw: int = 300) -> list[tuple[str, str]]:
    """The question panel as (text, tag) segments; tags pick colours in the GUI.

    Option tags: "correct" for the ground truth, "wrong" for a wrong model pick,
    "option" otherwise. The model's pick is also marked in the option text.
    """
    gt = q.answer_label.strip().upper()
    picked = picked_letter(q)
    out = [(q.question.strip() + "\n", "question")]
    for k, v in q.options.items():
        key = k.strip().upper()
        marks = []
        if key == gt:
            marks.append("correct")
        if q.pred is not None and key == picked:
            marks.append("model")
        if key == gt:
            sym, tag = "✓", "correct"
        elif "model" in marks:
            sym, tag = "✗", "wrong"
        else:
            sym, tag = "", "option"
        suffix = f"   ← {' + '.join(marks)}" if marks else ""
        out.append((f"{sym}\t{k}. {v}{suffix}\n", tag))
    out.append(("\n", ""))

    p = q.pred
    if p is None:
        out.append(("Model: no prediction for this question\n", "model_none"))
    else:
        letter = picked or "?"
        if p.correct is None:
            verdict, tag = "unscored", "model_none"
        elif p.correct:
            verdict, tag = "correct", "model_ok"
        else:
            verdict, tag = f"wrong (correct: {gt or '?'})", "model_bad"
        out.append((f"Model answered {letter}: {verdict}\n", tag))
        raw = " ".join(p.raw.split())
        if raw and raw.upper() != letter:
            if len(raw) > max_raw:
                raw = raw[: max_raw - 3] + "..."
            out.append((f"raw output: {raw}\n", "raw"))
        if p.gt_letter and p.gt_letter != gt:
            out.append(
                (
                    (
                        f"note: ground truth is {p.gt_letter} in the generation file "
                        f"but {gt} in the dataset\n"
                    ),
                    "warn",
                )
            )

    if q.evidence:
        out.append(("Evidence\n", "section"))
        for e in q.evidence:
            out.append((f"{_ts(e.start)}-{_ts(e.end)}  {e.description}\n", "evidence"))
    return out


def header_text(st: PlaybackState, i: int) -> str:
    q = st.questions[i]
    status = st.status(i)
    if status == "upcoming" and q.t is not None:
        when = f"in {fmt_delta(st.frame_of(q.t) / st.fps - st.t)}"
    elif status == "untimed":
        when = "no timestamp"
    else:
        when = status
    return f"Question {i + 1}/{len(st.questions)}  @ {_ts(q.t)}  ({when})"


def stop_hint(st: PlaybackState) -> str:
    k = st.remaining_at_stop
    if k:
        return f"{k} more question{'s' if k > 1 else ''} at this timestamp: Play shows the next"
    if st.browsing:
        return "Browsing: Play or a seek returns to the upcoming question"
    if st.paused and st.at_stop:
        return "Stopped at a question: Play to continue"
    return ""


# --------------------------------------------------------------------------- #
# Video decoding
# --------------------------------------------------------------------------- #
class VideoReader:
    """Random access to frames, decoding forward instead of seeking for short hops."""

    GRAB_LIMIT = 50  # frames; HEVC seeks cost more than decoding this many

    def __init__(self, path: Path) -> None:
        import cv2  # lazy, so the logic above is usable without OpenCV

        self._cv2 = cv2
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and math.isfinite(fps) and fps > 0 else 30.0
        self.n_frames = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        self.pos = -1  # index of self.frame
        self.frame: Any = None  # last decoded BGR frame
        self._next: int | None = 0  # index the next cap.read() returns

    def read(self, idx: int) -> Any:
        """Frame idx as a BGR array, or None if it can't be decoded."""
        if idx == self.pos and self.frame is not None:
            return self.frame
        nxt = self._next
        if nxt is None or not nxt <= idx <= nxt + self.GRAB_LIMIT:
            self.cap.set(self._cv2.CAP_PROP_POS_FRAMES, idx)
            nxt = idx
        for _ in range(idx - nxt):
            if not self.cap.grab():
                self._next = None
                return None
        ok, frame = self.cap.read()
        if not ok:
            self._next = None
            return None
        self.pos, self.frame, self._next = idx, frame, idx + 1
        return frame

    def close(self) -> None:
        self.cap.release()


def frame_to_photo(
    frame: Any, width: int, height: int, master: tk.Misc
) -> tk.PhotoImage:
    """Scale a BGR frame to fit width x height (keeping its aspect) as a Tk image."""
    import cv2

    h, w = frame.shape[:2]
    s = min(width / w, height / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    rgb = cv2.cvtColor(
        cv2.resize(frame, (nw, nh), interpolation=interp), cv2.COLOR_BGR2RGB
    )
    ppm = b"P6 %d %d 255 " % (nw, nh) + rgb.tobytes()
    return tk.PhotoImage(master=master, data=ppm, format="PPM")


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
GREEN, RED, GREY = "#1a7f37", "#cf222e", "#8c959f"
STATUS_COLORS = {
    "upcoming": "#0969da",
    "now": "#bc4c00",
    "passed": "#6e7781",
    "untimed": "#6e7781",
}


def pred_color(q: Question) -> str:
    if q.pred is None or q.pred.correct is None:
        return GREY
    return GREEN if q.pred.correct else RED


class App:
    """Tk front end: a video list, and a player with the question panel."""

    SKIP_S = 10.0
    BIG_SKIP_S = 60.0
    PANEL_W = 480
    POLL_MS = 40  # timer period while paused

    def __init__(self, root: tk.Tk | tk.Toplevel, ds: Dataset) -> None:
        if ds.gen is None:
            raise ValueError("the dataset has no generations attached")
        self.root, self.ds = root, ds
        self.gen: Generations = ds.gen
        self.rows = video_rows(ds)
        self.state: PlaybackState | None = None
        self.reader: VideoReader | None = None
        self.uid: str | None = None
        self.player: ttk.Frame | None = None
        self._photo: tk.PhotoImage | None = None
        self._after: str | None = None
        self._acc = 0.0  # frames owed to the wall clock while playing
        self._last = 0.0
        self._panel_cursor: int | None = None
        self._seek_to: int | None = None
        self._tl_x = (0, 1)  # timeline bar x range, set when it is drawn

        root.title(f"UCS-Bench generations | {self.gen.model}")
        w = min(1500, root.winfo_screenwidth() - 60)
        h = min(900, root.winfo_screenheight() - 100)
        root.geometry(f"{w}x{h}")
        root.minsize(1000, 640)
        self._init_style()
        self.picker = self._build_picker()
        self.picker.pack(fill="both", expand=True)
        self._bind_keys()

    # ---- setup ------------------------------------------------------------ #
    def _init_style(self) -> None:
        fam = tkfont.nametofont("TkDefaultFont").actual("family")
        self.f_title = (fam, 13, "bold")
        self.f_header = (fam, 12, "bold")
        self.f_body = (fam, 11)
        self.f_bold = (fam, 11, "bold")
        self.f_small = (fam, 9)
        style = ttk.Style(self.root)
        style.configure("Title.TLabel", font=self.f_title)
        style.configure("Header.TLabel", font=self.f_header)
        style.configure("Small.TLabel", font=self.f_small, foreground="#57606a")

    def _bind_keys(self) -> None:
        self.keys: dict[str, Callable[[], None]] = {
            "<space>": self.play_pause,
            "<Left>": lambda: self.skip(-self.SKIP_S),
            "<Right>": lambda: self.skip(self.SKIP_S),
            "<Shift-Left>": lambda: self.skip(-self.BIG_SKIP_S),
            "<Shift-Right>": lambda: self.skip(self.BIG_SKIP_S),
            "<Home>": self.restart,
            "<bracketleft>": self.slower,
            "<bracketright>": self.faster,
            "<KeyPress-p>": lambda: self.browse(-1),
            "<KeyPress-n>": lambda: self.browse(1),
            "<Up>": lambda: self.browse(-1),
            "<Down>": lambda: self.browse(1),
            "<Escape>": self.close_video,
        }
        for seq, fn in self.keys.items():
            self.root.bind(seq, self._key_handler(fn))

    def _key_handler(self, fn: Callable[[], None]) -> Callable[[tk.Event], str | None]:
        return lambda _e: self._on_key(fn)

    def _on_key(self, fn: Callable[[], None]) -> str | None:
        if self.state is None:  # video list: let the list handle its keys
            return None
        fn()
        return "break"

    def _build_picker(self) -> ttk.Frame:
        f = ttk.Frame(self.root, padding=12)
        ttk.Label(f, text=f"{self.gen.model}", style="Title.TLabel").pack(anchor="w")
        c = sum(r.n_correct for r in self.rows)
        n = sum(r.n_scored for r in self.rows)
        acc = f"{c}/{n} correct ({c / n:.1%})" if n else "no scored predictions"
        lost = sum(len(v) for v in self.gen.unmatched.values())
        info = f"{self.gen.path}   |   {len(self.rows)} videos, {acc}"
        if lost:
            info += f"   |   {lost} predictions matched no dataset question"
        ttk.Label(f, text=info, style="Small.TLabel").pack(anchor="w", pady=(2, 8))
        ttk.Label(
            f, text="Double-click a video, or select it and press Enter, to play it."
        ).pack(anchor="w", pady=(0, 6))

        body = ttk.Frame(f)
        body.pack(fill="both", expand=True)
        cols = ("questions", "correct", "accuracy", "length", "file")
        tree = ttk.Treeview(
            body, columns=cols, show="tree headings", selectmode="browse"
        )
        tree.heading("#0", text="video")
        tree.column("#0", width=380, stretch=True)
        for col, w in zip(cols, (90, 90, 90, 90, 90)):
            tree.heading(col, text=col)
            tree.column(col, width=w, anchor="center", stretch=False)
        for r in self.rows:
            tree.insert(
                "",
                "end",
                iid=r.uid,
                text=r.uid,
                values=(
                    r.n_questions,
                    f"{r.n_correct}/{r.n_scored}",
                    f"{r.accuracy:.0%}" if r.accuracy is not None else "-",
                    r.length,
                    "ok" if r.has_video else "missing",
                ),
                tags=() if r.has_video else ("missing",),
            )
        tree.tag_configure("missing", foreground=GREY)
        sb = ttk.Scrollbar(body, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        tree.bind("<Double-1>", lambda _e: self._open_selected())
        tree.bind("<Return>", lambda _e: self._open_selected())
        ttk.Button(f, text="Play selected video", command=self._open_selected).pack(
            anchor="e", pady=(8, 0)
        )
        if self.rows:
            tree.selection_set(self.rows[0].uid)
            tree.focus(self.rows[0].uid)
        tree.focus_set()
        self.picker_tree = tree
        return f

    def _build_player(self) -> ttk.Frame:
        assert self.state is not None and self.uid is not None
        st = self.state
        f = ttk.Frame(self.root, padding=8)
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=1)

        # -- left: video, timeline, controls
        left = ttk.Frame(f)
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        top = ttk.Frame(left)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(
            top, text="◀ Videos (Esc)", command=self.close_video, takefocus=False
        ).pack(side="left")
        c, n = score(st.questions)
        acc = f"{c}/{n} correct ({c / n:.0%})" if n else "no scored predictions"
        ttk.Label(top, text=f"  {self.uid}", style="Header.TLabel").pack(side="left")
        info = f"   {len(st.questions)} questions, {acc}   |   {self.gen.model}"
        lost = len(self.gen.unmatched.get(self.uid, []))
        if lost:
            info += f"   |   {lost} predictions matched no question"
        ttk.Label(top, text=info, style="Small.TLabel").pack(side="left")

        self.video = tk.Canvas(
            left, width=960, height=560, bg="black", highlightthickness=0
        )
        self.video.grid(row=1, column=0, sticky="nsew")
        self._img = self.video.create_image(480, 280, anchor="center")
        self.video.bind("<Configure>", lambda _e: self._redisplay())

        self.timeline = tk.Canvas(
            left, height=44, bg="#f6f8fa", highlightthickness=0, cursor="hand2"
        )
        self.timeline.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self.timeline.bind("<Configure>", lambda _e: self._draw_timeline())
        self.timeline.bind("<Button-1>", self._on_timeline)
        self.timeline.bind("<B1-Motion>", self._on_timeline)

        ctl = ttk.Frame(left)
        ctl.grid(row=3, column=0, sticky="ew", pady=(6, 0))

        def btn(
            text: str, cmd: Callable[[], None], width: int | None = None
        ) -> ttk.Button:
            b = ttk.Button(ctl, text=text, command=cmd, takefocus=False)
            if width:
                b.configure(width=width)
            b.pack(side="left", padx=2)
            return b

        btn("⏮ Start", self.restart)
        btn(f"« {self.SKIP_S:g} s", lambda: self.skip(-self.SKIP_S))
        self.play_btn = btn("▶ Play", self.play_pause, width=10)
        btn(f"{self.SKIP_S:g} s »", lambda: self.skip(self.SKIP_S))
        ttk.Label(ctl, text="   speed").pack(side="left")
        btn("−", self.slower, width=3)
        self.speed_lbl = ttk.Label(ctl, width=6, anchor="center", font=self.f_bold)
        self.speed_lbl.pack(side="left")
        btn("+", self.faster, width=3)
        self.time_lbl = ttk.Label(ctl, font=self.f_body)
        self.time_lbl.pack(side="left", padx=(14, 0))
        self.hint_lbl = ttk.Label(
            left, foreground=STATUS_COLORS["now"], font=self.f_body
        )
        self.hint_lbl.grid(row=4, column=0, sticky="w", pady=(4, 0))

        # -- right: question panel
        right = ttk.Frame(f, width=self.PANEL_W)
        right.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        right.grid_propagate(False)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=3)
        right.rowconfigure(4, weight=2)
        self.q_header = ttk.Label(right, style="Header.TLabel")
        self.q_header.grid(row=0, column=0, sticky="w")
        self.q_meta = ttk.Label(
            right, style="Small.TLabel", wraplength=self.PANEL_W - 10
        )
        self.q_meta.grid(row=1, column=0, sticky="w", pady=(2, 6))

        self.text = tk.Text(
            right, wrap="word", font=self.f_body, relief="flat", padx=8, pady=8,
            bg="white", cursor="arrow", takefocus=False, height=14, tabs=("26p",),
        )  # fmt: skip
        self.text.grid(row=2, column=0, sticky="nsew")
        opt: dict[str, Any] = {
            "lmargin1": 6,
            "lmargin2": "26p",
            "spacing1": 3,
            "spacing3": 3,
        }
        self.text.tag_configure(
            "question", font=(self.f_header[0], 12, "bold"), spacing3=10
        )
        self.text.tag_configure("option", **opt)
        self.text.tag_configure(
            "correct", foreground="#116329", background="#dafbe1", **opt
        )
        self.text.tag_configure(
            "wrong", foreground="#82071e", background="#ffebe9", **opt
        )
        self.text.tag_configure("model_ok", foreground=GREEN, font=self.f_bold)
        self.text.tag_configure("model_bad", foreground=RED, font=self.f_bold)
        self.text.tag_configure("model_none", foreground="#6e7781", font=self.f_bold)
        self.text.tag_configure("raw", foreground="#57606a")
        self.text.tag_configure("warn", foreground="#9a6700")
        self.text.tag_configure(
            "section", font=self.f_bold, foreground="#57606a", spacing1=10
        )
        self.text.tag_configure("evidence", foreground="#57606a", lmargin2=20)
        self.text.configure(state="disabled")

        nav = ttk.Frame(right)
        nav.grid(row=3, column=0, sticky="ew", pady=6)
        ttk.Button(
            nav,
            text="◀ Previous question",
            command=lambda: self.browse(-1),
            takefocus=False,
        ).pack(side="left")
        ttk.Button(
            nav, text="Next question ▶", command=lambda: self.browse(1), takefocus=False
        ).pack(side="right")

        qf = ttk.Frame(right)
        qf.grid(row=4, column=0, sticky="nsew")
        cols = ("time", "model", "gt", "question")
        qt = ttk.Treeview(
            qf, columns=cols, show="tree headings", selectmode="browse", takefocus=False
        )
        qt.heading("#0", text="#")
        qt.column("#0", width=40, stretch=False, anchor="e")
        for col, w in zip(cols[:3], (70, 52, 36)):
            qt.heading(col, text=col)
            qt.column(col, width=w, stretch=False, anchor="center")
        qt.heading("question", text="question", anchor="w")
        qt.column("question", width=260, stretch=True)
        for i, q in enumerate(st.questions):
            model, tag = model_cell(q)
            values = (
                _ts(q.t),
                model,
                q.answer_label.strip(),
                " ".join(q.question.split()),
            )
            qt.insert(
                "", "end", iid=str(i), text=str(i + 1), values=values, tags=(tag,)
            )
        qt.tag_configure("ok", foreground=GREEN)
        qt.tag_configure("bad", foreground=RED)
        qt.tag_configure("none", foreground=GREY)
        sb = ttk.Scrollbar(qf, orient="vertical", command=qt.yview)
        qt.configure(yscrollcommand=sb.set)
        qt.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        qt.bind("<<TreeviewSelect>>", self._on_question_row)
        self.qtree = qt
        return f

    # ---- screens ---------------------------------------------------------- #
    def _open_selected(self) -> None:
        sel = self.picker_tree.selection()
        if not sel:
            return
        try:
            self.open_video(sel[0])
        except (FileNotFoundError, RuntimeError) as e:
            messagebox.showerror("Cannot play video", str(e), parent=self.root)

    def open_video(self, uid: str) -> None:
        path = self.ds.video_path(uid)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        reader = VideoReader(path)
        self.close_video()
        self.uid, self.reader = uid, reader
        self.state = PlaybackState(self.ds.questions(uid), reader.fps, reader.n_frames)
        self.picker.pack_forget()
        self.player = self._build_player()
        self.player.pack(fill="both", expand=True)
        self.root.title(f"UCS-Bench generations | {self.gen.model} | {uid}")
        self.root.focus_set()  # keys go to the player, not the hidden video list
        self.root.update_idletasks()
        self._panel_cursor = None
        self._render()
        self._after = self.root.after(self.POLL_MS, self._tick)

    def close_video(self) -> None:
        if self._after is not None:
            self.root.after_cancel(self._after)
            self._after = None
        if self.reader is not None:
            self.reader.close()
            self.reader = None
        self.state, self.uid, self._photo = None, None, None
        if self.player is not None:
            self.player.destroy()
            self.player = None
        if not self.picker.winfo_ismapped():
            self.picker.pack(fill="both", expand=True)
        self.root.title(f"UCS-Bench generations | {self.gen.model}")
        self.picker_tree.focus_set()

    # ---- player actions --------------------------------------------------- #
    def _act(self, fn: Callable[[PlaybackState], object]) -> None:
        if self.state is not None:
            fn(self.state)
            self._render()

    def play_pause(self) -> None:
        self._last, self._acc = time.perf_counter(), 0.0
        self._act(PlaybackState.play_pause)

    def skip(self, seconds: float) -> None:
        self._act(lambda st: st.skip(seconds))

    def restart(self) -> None:
        self._act(lambda st: st.seek(0))

    def faster(self) -> None:
        self._act(PlaybackState.faster)

    def slower(self) -> None:
        self._act(PlaybackState.slower)

    def browse(self, delta: int) -> None:
        self._act(lambda st: st.browse(delta))

    def step(self, n: int = 1) -> None:
        """Advance playback by n frames (the timer calls this while playing)."""
        self._act(lambda st: st.advance(n))

    def _tick(self) -> None:
        self._after = None
        st = self.state
        if st is None:
            return
        delay = self.POLL_MS
        if not st.paused:
            now = time.perf_counter()
            rate = st.fps * st.speed
            # if decoding can't keep up, skip frames, but owe at most ~0.5 s
            self._acc = min(self._acc + (now - self._last) * rate, rate * 0.5 + 1)
            self._last = now
            n = int(self._acc)
            if n >= 1:
                self._acc -= n
                self.step(n)
            if st.paused:
                self._acc = 0.0
            else:
                delay = max(1, int((1.0 - self._acc) / rate * 1000))
        self._after = self.root.after(delay, self._tick)

    def _on_question_row(self, _e: object) -> None:
        st = self.state
        sel = self.qtree.selection()
        if st is not None and sel and int(sel[0]) != st.cursor():
            st.select(int(sel[0]))
            self._render()
        self.root.focus_set()  # keep arrow keys / space for the player

    def _on_timeline(self, e: tk.Event) -> None:
        st = self.state
        if st is None:
            return
        x0, x1 = self._tl_x
        frac = min(max((e.x - x0) / max(1, x1 - x0), 0.0), 1.0)
        # coalesce drags: decode only the latest position
        if self._seek_to is None:
            self.root.after(15, self._do_timeline_seek)
        self._seek_to = round(frac * (st.n_frames - 1))

    def _do_timeline_seek(self) -> None:
        f, self._seek_to = self._seek_to, None
        if f is not None:
            self._act(lambda st: st.seek(f))

    # ---- drawing ---------------------------------------------------------- #
    def _render(self) -> None:
        st, reader = self.state, self.reader
        if st is None or reader is None:
            return
        frame = reader.read(st.frame)
        while frame is None and st.frame > 0:  # frame count was too high
            st.end_at(st.frame - 1)
            frame = reader.read(st.frame)
        if frame is not None:
            self._show_frame(frame)
        self._draw_timeline()
        self._update_panel()
        self.play_btn.configure(text="⏸ Pause" if not st.paused else "▶ Play")
        self.speed_lbl.configure(text=f"{st.speed:g}x")
        self.time_lbl.configure(text=f"{fmt_ts(st.t)} / {fmt_ts(st.duration)}")
        self.hint_lbl.configure(text=stop_hint(st))

    def _show_frame(self, frame: Any) -> None:
        c = self.video
        w, h = c.winfo_width(), c.winfo_height()
        if w < 10 or h < 10:  # not laid out yet
            w, h = int(c["width"]), int(c["height"])
        self._photo = frame_to_photo(frame, w, h, self.root)
        c.itemconfigure(self._img, image=self._photo)
        c.coords(self._img, w // 2, h // 2)

    def _redisplay(self) -> None:
        if self.reader is not None and self.reader.frame is not None:
            self._show_frame(self.reader.frame)

    def _update_panel(self) -> None:
        st = self.state
        assert st is not None
        cur = st.cursor()
        if cur is None:
            self.q_header.configure(text="No questions for this video", foreground=GREY)
            return
        status = st.status(cur)
        self.q_header.configure(
            text=header_text(st, cur), foreground=STATUS_COLORS[status]
        )
        if cur == self._panel_cursor:
            return
        self._panel_cursor = cur
        q = st.questions[cur]
        self.q_meta.configure(text=f"{q.qid}  |  {q.category} / {q.subcategory}")
        t = self.text
        t.configure(state="normal")
        t.delete("1.0", "end")
        for text, tag in question_lines(q):
            t.insert("end", text, tag)
        t.configure(state="disabled")
        iid = str(cur)
        if self.qtree.selection() != (iid,):
            self.qtree.selection_set(iid)
        self.qtree.see(iid)

    def _draw_timeline(self) -> None:
        st = self.state
        if st is None:
            return
        c = self.timeline
        c.delete("all")
        w = c.winfo_width() if c.winfo_width() > 10 else int(c["width"])
        h = c.winfo_height() if c.winfo_height() > 10 else int(c["height"])
        x0, x1, y0, y1 = 8, w - 8, 14, h - 8
        self._tl_x = (x0, x1)
        last = max(1, st.n_frames - 1)

        def X(f: float) -> float:
            return x0 + (x1 - x0) * min(max(f / last, 0.0), 1.0)

        c.create_rectangle(x0, y0, x1, y1, fill="#d0d7de", outline="")
        c.create_rectangle(x0, y0, X(st.frame), y1, fill="#afb8c1", outline="")
        cur = st.cursor()
        if cur is not None:
            for e in st.questions[cur].evidence:
                if e.start is not None and e.end is not None:
                    a, b = X(st.frame_of(e.start)), X(st.frame_of(e.end + 1))
                    c.create_rectangle(
                        a, y0 + 4, max(b, a + 2), y1 - 4, fill="#4ac26b", outline=""
                    )
        for i, f in enumerate(st.qframes):
            if f is not None and i != cur:
                c.create_line(
                    X(f),
                    y0 - 6,
                    X(f),
                    y1 + 2,
                    fill=pred_color(st.questions[i]),
                    width=1,
                )
        if cur is not None and st.qframes[cur] is not None:
            x = X(st.qframes[cur] or 0)
            col = pred_color(st.questions[cur])
            c.create_line(x, y0 - 6, x, y1 + 2, fill=col, width=3)
            c.create_polygon(x - 6, 1, x + 6, 1, x, y0 - 4, fill=col, outline="")
        x = X(st.frame)
        c.create_line(x, y0 - 2, x, y1 + 4, fill="#24292f", width=2)


# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "gen", type=Path, help="generation/results json from evaluate_ucsbench_vlm.py"
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"dataset root (default: {DEFAULT_ROOT})",
    )
    ap.add_argument("--video", help="open this video directly (uid or unique prefix)")
    return ap.parse_args(argv)


def load(args: argparse.Namespace) -> Dataset:
    if not args.gen.is_file():
        sys.exit(f"Generation file not found: {args.gen}")
    ds = Dataset(args.root)
    ds.gen = Generations(args.gen)
    return ds


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ds = load(args)
    uid = ds.resolve(args.video) if args.video else None
    if sys.platform == "win32":  # sharp text on high-DPI screens
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    app = App(root, ds)
    if uid is not None:
        try:
            app.open_video(uid)
        except (FileNotFoundError, RuntimeError) as e:
            root.destroy()
            sys.exit(str(e))
    root.mainloop()


if __name__ == "__main__":
    main()
