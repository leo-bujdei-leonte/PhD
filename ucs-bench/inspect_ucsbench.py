#!/usr/bin/env python
"""Inspect the UCS-Bench dataset (https://huggingface.co/datasets/cocowy1/UCS-Bench).

Usage example:
    python ./ucs-bench/inspect_ucsbench.py --root '../UCS-Bench/data/UCS-Bench/' play 002d2729-df71-438d-8396-5895b349e8fd

Expected layout (as downloaded with `huggingface-cli download cocowy1/UCS-Bench`):

    <root>/
        <video_uid>.mp4            # egocentric videos (flat)
        metadata.jsonl             # {"file_name": "<video_uid>.mp4"} per line
        QAs/<video_uid>-mcq.json   # list of timestamped multiple-choice questions

Sub-commands
------------
    stats              dataset-wide summary (videos, questions, categories, ...)
    list               one line per video: uid, #questions, length, categories
    show  <uid>        print every question of a video (with evidence + answer)
    play  <uid>        replay a video with question / evidence overlays

Player controls (`play`)
------------------------
    space          pause / resume (at a stop shared by several questions,
                   shows the next one first)
    <- / ->        skip back / forward 5 s      (also j / l)
    up / down      skip back / forward 30 s     (also J / L)
    , / .          step one frame back / forward (pauses)
    r              rewind to the start
    n / p          jump to next / previous question (one at a time)
    [ / ]          slower / faster playback (0.25x .. 8x)
    a              hide / show the correct answer (shown by default)
    g              toggle model answers (with --gen)
    e              toggle evidence-span overlay
    mouse click    seek by clicking on the timeline bar
    q / Esc        quit

Generations (`--gen results.json`, for list / show / play)
--------------------------------------------------------
Load the output json of evaluate_ucsbench_vlm.py to show the model's answer
for each question and whether it is correct: `list` adds a per-video accuracy
column, `show` marks the model's option with '>' and prints CORRECT/WRONG, and
`play` shows the model answer in the side panel (correctness and green/red
question ticks only while the answer is revealed; answers are shown by default,
press 'a' to hide them and quiz yourself).

    python ./ucs-bench/inspect_ucsbench.py show 01169 --gen ../ucs-bench-results/2026-11-18/subset_directme_qwen3vl.json

On the timeline, evidence spans of the current question are drawn in green and
question timestamps as yellow ticks. When playback crosses a question timestamp
the player auto-pauses (disable with --no-autopause) so you can answer it first;
if several questions share that timestamp, space shows each in turn, then resumes.
The side panel shows the upcoming question; it switches to the next one on the
frame after the pause, so the question you stopped at stays visible while paused.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "UCS-Bench" / "data" / "UCS-Bench"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def parse_ts(value) -> float | None:
    """Parse the (sometimes messy) HH:MM:SS timestamps used in the annotations.

    Handles 'HH:MM:SS', 'MM:SS', trailing colons/spaces and odd digit counts
    such as '00:1:05' or '00:01:005'.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    parts = [p for p in re.split(r"[:\s]+", str(value).strip()) if p != ""]
    if not parts or not all(p.replace(".", "", 1).isdigit() for p in parts):
        return None
    nums = [float(p) for p in parts][-3:]
    secs = 0.0
    for n in nums:
        secs = secs * 60 + n
    return secs


def fmt_ts(secs: float) -> str:
    secs = max(0.0, secs)
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class Evidence:
    description: str
    start: float | None
    end: float | None


@dataclass
class Prediction:
    letter: str | None  # parsed answer letter (None if unparsable)
    raw: str  # raw model output
    correct: bool | None
    gt_letter: str | None  # ground truth according to the generation file

    @classmethod
    def from_json(cls, d: dict) -> Prediction:
        gt = d.get("gt_idx")
        return cls(
            letter=d.get("predicted_letter"),
            raw=str(d.get("raw_answer") or ""),
            correct=d.get("correct"),
            gt_letter="ABCDE"[int(gt)] if isinstance(gt, int) and 0 <= gt < 5 else None,
        )

    def verdict(self) -> str:
        if self.correct is None:
            return "UNSCORED"
        return "CORRECT" if self.correct else "WRONG"


def _norm(s: str) -> str:
    return " ".join(str(s).split()).lower()


class Generations:
    """Model answers from an evaluate_ucsbench_vlm.py output json.

    Predictions carry no qid, so they are matched to dataset questions by
    (video_uid, query_timestamp, question text), falling back to the question
    text with the closest timestamp (<= 2 s) for messy annotation timestamps.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        d = json.loads(self.path.read_text(encoding="utf-8"))
        self.model = d.get("model") or self.path.stem
        self.summary = d
        self.by_vid: dict[str, list[dict]] = collections.defaultdict(list)
        for p in d.get("predictions") or []:
            self.by_vid[str(p.get("video_uid", "")).strip()].append(p)
        self.unmatched: dict[str, list[dict]] = {}

    def attach(self, uid: str, qs: list[Question]) -> None:
        left = list(self.by_vid.get(uid.strip(), []))
        # pass 1: exact timestamp + question text
        for q in qs:
            for i, p in enumerate(left):
                if (
                    q.t is not None
                    and abs(float(p.get("query_timestamp", -1e9)) - q.t) < 0.01
                    and _norm(p.get("question", "")) == _norm(q.question)
                ):
                    q.pred = Prediction.from_json(left.pop(i))
                    break
        # pass 2: same text, nearest timestamp
        for q in qs:
            if q.pred is not None:
                continue
            cands = [
                (abs(float(p.get("query_timestamp", 0)) - (q.t or 0)), i)
                for i, p in enumerate(left)
                if _norm(p.get("question", "")) == _norm(q.question)
            ]
            if cands and min(cands)[0] <= 2.0:
                q.pred = Prediction.from_json(left.pop(min(cands)[1]))
        self.unmatched[uid] = left

    def videos(self) -> set[str]:
        return set(self.by_vid)


def score(qs: list[Question]) -> tuple[int, int]:
    """(#correct, #scored) over questions that have a prediction."""
    scored = [q.pred for q in qs if q.pred is not None and q.pred.correct is not None]
    return sum(bool(p.correct) for p in scored), len(scored)


@dataclass
class Question:
    qid: str
    category: str
    subcategory: str
    question: str
    t: float | None
    options: dict[str, str]
    answer_label: str
    answer: str
    evidence: list[Evidence] = field(default_factory=list)
    raw: dict = field(default_factory=dict, repr=False)
    pred: Prediction | None = None  # model answer, if a generation file is loaded

    @classmethod
    def from_json(cls, d: dict) -> Question:
        ev = []
        for e in d.get("evidence") or []:
            ts = e.get("timestamps") or {}
            ev.append(
                Evidence(
                    e.get("description", ""),
                    parse_ts(ts.get("start")),
                    parse_ts(ts.get("end")),
                )
            )
        return cls(
            qid=d.get("qid") or d.get("q_id") or "?",
            category=d.get("category", ""),
            subcategory=d.get("subcategory", ""),
            question=d.get("question", ""),
            t=parse_ts(d.get("question_timestamps")),
            options=d.get("options") or {},
            answer_label=d.get("answer_label", ""),
            answer=d.get("answer_text")
            or d.get("correct_answer")
            or d.get("answer", ""),
            evidence=ev,
            raw=d,
        )


class Dataset:
    def __init__(self, root: Path):
        self.root = Path(root)
        if not self.root.is_dir():
            sys.exit(f"Dataset root not found: {self.root}  (use --root)")
        self.qa_dir = self.root / "QAs"
        self.gen: Generations | None = None

    def video_uids(self) -> list[str]:
        meta = self.root / "metadata.jsonl"
        uids = []
        if meta.exists():
            for line in meta.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    uids.append(Path(json.loads(line)["file_name"]).stem)
        if not uids:
            uids = [p.stem for p in self.root.glob("*.mp4")]
        return sorted(uids)

    def qa_uids(self) -> list[str]:
        return sorted(
            p.name[: -len("-mcq.json")] for p in self.qa_dir.glob("*-mcq.json")
        )

    def video_path(self, uid: str) -> Path:
        return self.root / f"{uid}.mp4"

    def questions(self, uid: str) -> list[Question]:
        p = self.qa_dir / f"{uid}-mcq.json"
        if not p.exists():
            return []
        qs = [Question.from_json(d) for d in json.loads(p.read_text(encoding="utf-8"))]
        qs = sorted(qs, key=lambda q: (q.t is None, q.t or 0.0))
        if self.gen is not None:
            self.gen.attach(uid, qs)
        return qs

    def resolve(self, uid: str) -> str:
        """Allow unique prefixes, e.g. `play 0031` -> '0031d268-...'."""
        uids = set(self.video_uids()) | set(self.qa_uids())
        if uid in uids:
            return uid
        uid = Path(uid).stem
        if uid in uids:
            return uid
        hits = sorted(u for u in uids if u.startswith(uid))
        if len(hits) == 1:
            return hits[0]
        if not hits:
            sys.exit(f"No video matching '{uid}'")
        sys.exit(
            f"'{uid}' is ambiguous: {', '.join(hits[:10])}{' ...' if len(hits) > 10 else ''}"
        )


# --------------------------------------------------------------------------- #
# CLI: stats / list / show
# --------------------------------------------------------------------------- #
def cmd_stats(ds: Dataset, args) -> None:
    vids, qas = set(ds.video_uids()), set(ds.qa_uids())
    on_disk = {p.stem for p in ds.root.glob("*.mp4")}
    cat: collections.Counter[str] = collections.Counter()
    sub: collections.Counter[tuple[str, str]] = collections.Counter()
    qtype: collections.Counter[str] = collections.Counter()
    nq_per_vid: list[int] = []
    bad_ts, length_s = 0, []
    for uid in sorted(qas):
        qs = ds.questions(uid)
        nq_per_vid.append(len(qs))
        for q in qs:
            cat[q.category] += 1
            sub[(q.category, q.subcategory)] += 1
            qtype[q.raw.get("qtype", "")] += 1
            bad_ts += q.t is None
        if qs and (L := parse_ts(qs[0].raw.get("video_length"))):
            length_s.append(L)

    print(f"root               : {ds.root}")
    print(f"videos (metadata)  : {len(vids)}")
    print(f"videos (on disk)   : {len(on_disk)}   missing: {len(vids - on_disk)}")
    print(
        f"QA files           : {len(qas)}   videos without QA: {len(vids - qas)}   QA without video: {len(qas - on_disk)}"
    )
    print(
        f"questions          : {sum(nq_per_vid)}  (per video min/mean/max = "
        f"{min(nq_per_vid)}/{sum(nq_per_vid) / len(nq_per_vid):.1f}/{max(nq_per_vid)})"
    )
    if bad_ts:
        print(f"unparsable q. ts   : {bad_ts}")
    if length_s:
        print(
            f"annotated length   : {sum(length_s) / 3600:.1f} h total, "
            f"{min(length_s) / 60:.1f}-{max(length_s) / 60:.1f} min per video"
        )
    print("\ncategories:")
    for c, n in cat.most_common():
        print(f"  {n:6d}  {c}")
        for (c2, s), m in sorted(sub.items(), key=lambda kv: -kv[1]):
            if c2 == c:
                print(f"  {'':6s}    {m:5d}  {s}")
    print(
        "\nqtype:", ", ".join(f"{k or '(empty)'}={v}" for k, v in qtype.most_common())
    )
    if args.missing:
        print("\nvideos without QA:", *sorted(vids - qas), sep="\n  ")
        print("\nQA without video:", *sorted(qas - on_disk), sep="\n  ")


def cmd_list(ds: Dataset, args) -> None:
    qas = set(ds.qa_uids())
    gen = ds.gen
    if gen:
        print(f"generation: {gen.path}  model: {gen.model}")
    gen_hdr = f" {'gen':>12s}" if gen else ""
    print(f"{'video_uid':40s} {'#q':>4s} {'length':>9s}{gen_hdr}  categories")
    tot_c = tot_n = 0
    for uid in ds.video_uids():
        if gen and args.gen_only and uid not in gen.videos():
            continue
        qs = ds.questions(uid) if uid in qas else []
        if args.category and not any(
            args.category.lower() in q.category.lower() for q in qs
        ):
            continue
        length = qs[0].raw.get("video_length", "") if qs else ""
        cats = collections.Counter(q.category.split()[0] for q in qs)
        gen_col = ""
        if gen:
            c, n = score(qs)
            tot_c, tot_n = tot_c + c, tot_n + n
            gen_col = f" {f'{c}/{n} {100 * c / n:3.0f}%' if n else '-':>12s}"
        print(
            f"{uid:40s} {len(qs):4d} {length:>9s}{gen_col}  "
            + " ".join(f"{k}:{v}" for k, v in cats.most_common())
        )
    if gen:
        lost = sum(len(v) for v in gen.unmatched.values()) + sum(
            len(v) for u, v in gen.by_vid.items() if u not in gen.unmatched
        )
        acc = f"{tot_c}/{tot_n} = {100 * tot_c / tot_n:.1f}%" if tot_n else "-"
        print(f"\ngeneration accuracy over listed videos: {acc}")
        if lost and not args.category:
            print(f"predictions not matched to any listed question: {lost}")


def cmd_show(ds: Dataset, args) -> None:
    uid = ds.resolve(args.uid)
    qs = ds.questions(uid)
    gen = ds.gen
    print(f"{uid}  ({len(qs)} questions)  video: {ds.video_path(uid)}")
    if gen:
        c, n = score(qs)
        acc = (
            f"{c}/{n} correct ({100 * c / n:.1f}%)" if n else "no matching predictions"
        )
        print(f"model: {gen.model}  -- {acc}")
        print("legend: * ground truth, > model answer")
    print()
    w = textwrap.TextWrapper(width=110, initial_indent="    ", subsequent_indent="    ")
    for q in qs:
        if gen and args.wrong and (q.pred is None or q.pred.correct is not False):
            continue
        verdict = f"  [{q.pred.verdict()}]" if q.pred else ""
        print(
            f"[{fmt_ts(q.t) if q.t is not None else '??:??:??'}] {q.qid}  -- {q.category} / {q.subcategory}{verdict}"
        )
        print(w.fill("Q: " + q.question))
        for k, v in q.options.items():
            mark = "*" if k == q.answer_label else " "
            if gen:
                mark += ">" if q.pred and q.pred.letter == k else " "
            print(w.fill(f"{mark}{k}. {v}"))
        if gen:
            if q.pred is None:
                print(w.fill("model: (no prediction)"))
            else:
                p = q.pred
                line = f"model: {p.letter or '?'}  {p.verdict()}"
                if p.raw.strip() != (p.letter or ""):
                    line += f"  raw: {p.raw.strip()!r}"
                if p.gt_letter and p.gt_letter != q.answer_label.strip():
                    line += f"  (!! generation gt={p.gt_letter}, dataset gt={q.answer_label})"
                print(w.fill(line))
        for e in q.evidence:
            span = f"{fmt_ts(e.start) if e.start is not None else '?'}-{fmt_ts(e.end) if e.end is not None else '?'}"
            print(w.fill(f"evidence {span}: {e.description}"))
        print()
    if gen and gen.unmatched.get(uid):
        print(
            f"{len(gen.unmatched[uid])} prediction(s) for this video matched no question:"
        )
        for u in gen.unmatched[uid]:
            print(f"    @{u.get('query_timestamp')}  {u.get('question')}")


# --------------------------------------------------------------------------- #
# Player
# --------------------------------------------------------------------------- #
# cv2.waitKeyEx codes for arrow keys (Windows / Linux-GTK / macOS)
KEY_LEFT = {2424832, 65361, 63234}
KEY_RIGHT = {2555904, 65363, 63235}
KEY_UP = {2490368, 65362, 63232}
KEY_DOWN = {2621440, 65364, 63233}

SPEEDS = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0]
PANEL_W = 520
BAR_H = 46


class Player:
    def __init__(self, ds: Dataset, uid: str, args):
        import cv2  # imported lazily so stats/list/show work without OpenCV
        import numpy as np

        self.cv2, self.np = cv2, np
        self.uid = uid
        self.questions = ds.questions(uid)
        path = ds.video_path(uid)
        if not path.exists():
            sys.exit(f"Video not found: {path}")
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            sys.exit(f"Could not open {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.duration = self.n_frames / self.fps
        w, h = (
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        )
        scale = min(1.0, args.max_width / w, args.max_height / h)
        self.vw, self.vh = int(w * scale), int(h * scale)
        self.canvas_h = max(self.vh, 360) + BAR_H

        self.speed_i = SPEEDS.index(1.0)
        self.paused = args.start_paused
        self.autopause = not args.no_autopause
        self.show_answer = True
        self.show_evidence = True
        self.gen = ds.gen
        self.show_pred = ds.gen is not None
        self.frame: Any = None  # current BGR frame (numpy array) or None
        self.pos = 0  # index of the frame currently held in self.frame
        self.win = f"UCS-Bench | {uid}"
        self.focus_q: Question | None = None  # question pinned by n/p / autopause
        self.pending_qs: list[Question] = []  # more questions at the current stop

        start = args.start
        if args.question:
            q = next(
                (
                    q
                    for q in self.questions
                    if q.qid == args.question or q.qid.endswith(args.question)
                ),
                None,
            )
            if q is None:
                sys.exit(f"Question '{args.question}' not found in {uid}")
            self.focus_q = q
            start = max(0.0, (q.t or 0) - args.lead_in)
        self.seek(start or 0.0)

    # ---- frame access ---------------------------------------------------- #
    def seek(self, t: float) -> None:
        idx = int(min(max(t, 0.0), self.duration) * self.fps)
        idx = min(max(idx, 0), self.n_frames - 1)
        self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if ok:
            self.frame, self.pos = frame, idx
        self._last_t = self.t

    def read_next(self, skip: int = 0) -> bool:
        for _ in range(skip):
            if not self.cap.grab():
                return False
        ok, frame = self.cap.read()
        if not ok:
            return False
        self.frame = frame
        self.pos += 1 + skip
        return True

    @property
    def t(self) -> float:
        return self.pos / self.fps

    # ---- question bookkeeping -------------------------------------------- #
    def _timed_questions(self) -> list[tuple[float, Question]]:
        """(timestamp, question) for every question that has a timestamp."""
        return [(q.t, q) for q in self.questions if q.t is not None]

    def current_question(self) -> Question | None:
        if self.focus_q is not None:
            return self.focus_q
        timed = self._timed_questions()
        if not timed:
            return None
        # Show the question reached on this very frame (where autopause stops);
        # from the next frame on, show the upcoming question instead.
        prev_t = (self.pos - 1) / self.fps
        upcoming = [q for t, q in timed if t > prev_t + 1e-6]
        return upcoming[0] if upcoming else timed[-1][1]

    def jump_question(self, direction: int) -> None:
        # step through questions in order (not by time), so questions sharing a
        # timestamp are each visited
        timed = self._timed_questions()
        qs = [q for _, q in timed]
        cur = self.current_question()
        if not timed or cur not in qs:
            return
        i = qs.index(cur)
        reached = self.t >= timed[i][0] - 1.0 / self.fps
        if direction > 0:
            i = i + 1 if reached else i  # an upcoming question is the "next" one
        else:
            i = i - 1
        t, q = timed[min(max(i, 0), len(timed) - 1)]
        self.focus_q = q
        # land on the first frame at/after q.t so resuming doesn't autopause on
        # the same question again
        self.seek(math.ceil(t * self.fps - 1e-6) / self.fps + 1e-6)
        self.paused = True

    # ---- rendering ------------------------------------------------------- #
    def _text(self, img, s, x, y, color=(230, 230, 230), scale=0.5, thick=1):
        cv2 = self.cv2
        # Black outline: same stroke width drawn at 1-px offsets. (A thicker
        # stroke would also widen OpenCV's glyph advance, so the outline would
        # drift right and leave ghost letters after the text.)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    cv2.putText(
                        img,
                        s,
                        (x + dx, y + dy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        scale,
                        (0, 0, 0),
                        thick,
                        cv2.LINE_AA,
                    )
        cv2.putText(
            img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA
        )

    def _wrap(
        self,
        img,
        s,
        x,
        y,
        width_chars,
        color=(230, 230, 230),
        scale=0.45,
        lh=18,
        max_lines=None,
    ):
        s = s.encode("ascii", "replace").decode()  # Hershey fonts are ASCII-only
        lines = textwrap.wrap(s, width_chars) or [""]
        if max_lines and len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = lines[-1][: width_chars - 3] + "..."
        for line in lines:
            self._text(img, line, x, y, color, scale)
            y += lh
        return y

    def render(self):
        cv2, np = self.cv2, self.np
        H, W = self.canvas_h, self.vw + PANEL_W
        canvas = np.full((H, W, 3), 24, np.uint8)
        if self.frame is not None:
            canvas[: self.vh, : self.vw] = cv2.resize(
                self.frame, (self.vw, self.vh), interpolation=cv2.INTER_AREA
            )
        q = self.current_question()
        t = self.t

        # active evidence on the video
        if q and self.show_evidence:
            y = 24
            for e in q.evidence:
                if (
                    e.start is not None
                    and e.end is not None
                    and e.start <= t <= e.end + 0.999
                ):
                    cv2.rectangle(
                        canvas, (0, 0), (self.vw - 1, self.vh - 1), (60, 200, 60), 3
                    )
                    y = self._wrap(
                        canvas,
                        "EVIDENCE: " + e.description,
                        10,
                        y,
                        max(20, self.vw // 9),
                        (120, 255, 120),
                        0.55,
                        22,
                    )

        # HUD
        state = "PAUSED" if self.paused else f"{SPEEDS[self.speed_i]:g}x"
        self._text(
            canvas,
            f"{fmt_ts(t)} / {fmt_ts(self.duration)}  [{state}]  frame {self.pos}",
            10,
            self.vh - 12,
            scale=0.55,
        )

        self._draw_panel(canvas, q)
        self._draw_timeline(canvas, q)
        return canvas

    def _draw_panel(self, canvas, q: Question | None):
        x0, cw = self.vw + 14, (PANEL_W - 28) // 8
        y = 26
        self._text(canvas, f"{self.uid}", x0, y, (200, 200, 255), 0.5)
        y += 20
        self._text(
            canvas, f"{len(self.questions)} questions", x0, y, (170, 170, 170), 0.45
        )
        if self.gen and self.show_pred:
            c, n = score(self.questions)
            acc = f"{c}/{n} correct" if self.show_answer else f"{n} answered"
            y += 18
            y = (
                self._wrap(
                    canvas,
                    f"model: {self.gen.model} ({acc})",
                    x0,
                    y,
                    cw,
                    (255, 200, 150),
                    0.42,
                    16,
                    max_lines=2,
                )
                - 18
            )
        y += 26
        if q is None:
            self._text(canvas, "No questions for this video.", x0, y)
            return
        idx = self.questions.index(q) + 1
        reached = q.t is not None and self.t >= q.t - 1e-6
        tag = "" if reached else "  (upcoming)"
        self._text(
            canvas,
            f"Q{idx}/{len(self.questions)}  @ {fmt_ts(q.t or 0)}{tag}",
            x0,
            y,
            (0, 220, 255),
            0.55,
            1,
        )
        y += 20
        y = self._wrap(
            canvas,
            f"{q.qid} | {q.category} / {q.subcategory}",
            x0,
            y,
            cw,
            (170, 170, 170),
            0.4,
            16,
        )
        y += 6
        y = self._wrap(canvas, q.question, x0, y, cw, (255, 255, 255), 0.47, 19)
        y += 8
        pred = q.pred if self.show_pred else None
        for k, v in q.options.items():
            correct = self.show_answer and k == q.answer_label
            picked = pred is not None and pred.letter == k
            col = (90, 230, 90) if correct else (215, 215, 215)
            if picked and not correct:
                col = (80, 80, 240) if self.show_answer else (255, 200, 150)
            y = self._wrap(
                canvas,
                f"{'>' if picked else ' '}{k}. {v}",
                x0,
                y,
                cw,
                col,
                0.43,
                17,
                max_lines=4,
            )
            y += 3
        y += 6
        if self.show_pred and self.gen:
            if pred is None:
                txt, col = "Model: (no prediction)", (130, 130, 130)
            elif self.show_answer:
                txt = f"Model: {pred.letter or '?'}  {pred.verdict()}"
                if pred.correct is None:
                    col = (200, 200, 200)
                else:
                    col = (90, 230, 90) if pred.correct else (80, 80, 240)
            else:
                txt, col = (
                    f"Model: {pred.letter or '?'}  (press 'a' to check)",
                    (255, 200, 150),
                )
            if pred is not None and pred.raw.strip() != (pred.letter or ""):
                txt += f"  raw: {pred.raw.strip()}"
            y = self._wrap(canvas, txt, x0, y, cw, col, 0.47, 19, max_lines=3)
        if self.show_answer:
            y = self._wrap(
                canvas, f"Answer: {q.answer_label}", x0, y, cw, (90, 230, 90), 0.47, 19
            )
        else:
            y = self._wrap(
                canvas,
                "Answer hidden (press 'a')",
                x0,
                y,
                cw,
                (130, 130, 130),
                0.43,
                18,
            )
        y += 8
        help_top = self.canvas_h - BAR_H - 10 - 16 * 4  # keep clear of the help
        if q.evidence and y < help_top - 20:
            self._text(canvas, "Evidence:", x0, y, (120, 255, 120), 0.45)
            y += 18
            for e in q.evidence:
                if y > help_top - 3 * 16:
                    break
                span = f"{fmt_ts(e.start or 0)}-{fmt_ts(e.end or 0)}"
                active = (
                    e.start is not None
                    and e.end is not None
                    and e.start <= self.t <= e.end + 0.999
                )
                y = self._wrap(
                    canvas,
                    f"{span} {e.description}",
                    x0,
                    y,
                    cw,
                    (120, 255, 120) if active else (160, 160, 160),
                    0.4,
                    16,
                    max_lines=3,
                )
                y += 2

        help_lines = [
            "spc play/pause (next q at a shared stop)  r rewind  q quit",
            "<-/-> 5s  up/dn 30s  ,/. frame  click bar: seek",
            "n/p next/prev question  [/] speed",
            "a hide/show answer  e evidence"
            + ("  g model answers" if self.gen else ""),
        ]
        help_y = self.canvas_h - BAR_H - 10 - 16 * (len(help_lines) - 1)
        for line in help_lines:
            self._text(canvas, line, x0, help_y, (120, 120, 120), 0.38)
            help_y += 16

    def _bar_geom(self):
        return (
            10,
            self.vw + PANEL_W - 10,
            self.canvas_h - BAR_H + 14,
            self.canvas_h - 12,
        )

    def _draw_timeline(self, canvas, q):
        cv2 = self.cv2
        x0, x1, y0, y1 = self._bar_geom()
        span = x1 - x0
        X = lambda s: int(x0 + span * min(max(s / self.duration, 0), 1))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (60, 60, 60), -1)
        cv2.rectangle(canvas, (x0, y0), (X(self.t), y1), (110, 110, 110), -1)
        if q and self.show_evidence:
            for e in q.evidence:
                if e.start is not None and e.end is not None:
                    cv2.rectangle(
                        canvas,
                        (X(e.start), y0 + 4),
                        (max(X(e.end), X(e.start) + 2), y1 - 4),
                        (60, 200, 60),
                        -1,
                    )
        for qq in self.questions:
            if qq.t is not None:
                col, w = ((0, 220, 255), 2) if qq is q else ((0, 150, 200), 1)
                if self.show_pred and self.show_answer and qq.pred is not None:
                    col = {True: (90, 230, 90), False: (80, 80, 240)}.get(
                        qq.pred.correct, col
                    )
                cv2.line(canvas, (X(qq.t), y0 - 6), (X(qq.t), y1 + 2), col, w)
        cv2.line(canvas, (X(self.t), y0 - 8), (X(self.t), y1 + 4), (255, 255, 255), 2)

    def fit_to_window(self, canvas):
        """Scale the canvas to the window keeping its aspect ratio, padding with
        black (letterbox) instead of letting OpenCV stretch it (e.g. fullscreen)."""
        cv2, np = self.cv2, self.np
        self._view = (0, 0, 1.0)
        try:
            _, _, ww, wh = cv2.getWindowImageRect(self.win)
        except cv2.error:
            return canvas
        H, W = canvas.shape[:2]
        if ww <= 0 or wh <= 0 or (ww, wh) == (W, H):
            return canvas
        s = min(ww / W, wh / H)
        nw, nh = max(1, round(W * s)), max(1, round(H * s))
        ox, oy = (ww - nw) // 2, (wh - nh) // 2
        out = np.zeros((wh, ww, 3), np.uint8)
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        out[oy : oy + nh, ox : ox + nw] = cv2.resize(
            canvas, (nw, nh), interpolation=interp
        )
        self._view = (ox, oy, s)
        return out

    def on_mouse(self, event, x, y, flags, _):
        if event != self.cv2.EVENT_LBUTTONDOWN:
            return
        ox, oy, s = getattr(self, "_view", (0, 0, 1.0))
        x, y = (x - ox) / s, (y - oy) / s  # window -> canvas coordinates
        x0, x1, y0, y1 = self._bar_geom()
        if y0 - 12 <= y <= y1 + 8 and x0 <= x <= x1:
            self.focus_q, self.pending_qs = None, []
            self.seek((x - x0) / (x1 - x0) * self.duration)

    # ---- main loop ------------------------------------------------------- #
    def advance(self, skip: int = 0) -> None:
        """Play forward one (+skip) frame, auto-pausing on question timestamps."""
        prev_t = self.t
        if not self.read_next(skip=skip):
            self.paused = True  # end of video
            return
        if not self.autopause:
            return
        # pin the question(s) we stopped for (frame drops may have skipped past
        # the exact frame); space shows the next one sharing this stop before
        # playback resumes
        hit = [q for q in self.questions if q.t is not None and prev_t < q.t <= self.t]
        if hit:
            self.paused = True
            self.focus_q, self.pending_qs = hit[0], hit[1:]

    def run(self) -> None:
        cv2 = self.cv2
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, self.vw + PANEL_W, self.canvas_h)
        cv2.setMouseCallback(self.win, self.on_mouse)
        next_tick = time.perf_counter()
        while True:
            cv2.imshow(self.win, self.fit_to_window(self.render()))

            if self.paused:
                key = cv2.waitKeyEx(30)
                next_tick = time.perf_counter()
            else:
                frame_dt = 1.0 / (self.fps * SPEEDS[self.speed_i])
                next_tick += frame_dt
                now = time.perf_counter()
                # if rendering can't keep up, drop frames instead of slowing down
                behind = max(0, int((now - next_tick) / frame_dt))
                if behind:
                    next_tick += behind * frame_dt
                key = cv2.waitKeyEx(max(1, int((next_tick - now) * 1000)))
                if key == -1:
                    self.advance(skip=behind)

            if cv2.getWindowProperty(self.win, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == -1:
                continue
            if not self.handle_key(key):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def handle_key(self, key: int) -> bool:
        ch = chr(key & 0xFF) if key < 0x10000 and (key & 0xFF) else ""
        if ch in ("q", "\x1b"):
            return False
        if ch == " " and self.paused and self.pending_qs:
            # several questions at this stop: show the next before resuming
            self.focus_q = self.pending_qs.pop(0)
            return True
        if ch not in ("a", "e", "g", "[", "]"):  # any navigation drops the queue
            self.pending_qs = []
        seek_by = None
        if key in KEY_LEFT or ch == "j":
            seek_by = -5
        elif key in KEY_RIGHT or ch == "l":
            seek_by = 5
        elif key in KEY_UP or ch == "J":
            seek_by = -30
        elif key in KEY_DOWN or ch == "L":
            seek_by = 30
        if seek_by is not None:
            self.focus_q = None
            self.seek(self.t + seek_by)
        elif ch == " ":
            self.paused = not self.paused
            if not self.paused:
                self.focus_q = None
        elif ch == ".":
            self.paused, self.focus_q = True, None
            self.read_next()
        elif ch == ",":
            self.paused, self.focus_q = True, None
            self.seek((self.pos - 1) / self.fps)
        elif ch == "r":
            self.focus_q = None
            self.seek(0)
        elif ch == "n":
            self.jump_question(+1)
        elif ch == "p":
            self.jump_question(-1)
        elif ch == "]":
            self.speed_i = min(self.speed_i + 1, len(SPEEDS) - 1)
        elif ch == "[":
            self.speed_i = max(self.speed_i - 1, 0)
        elif ch == "a":
            self.show_answer = not self.show_answer
        elif ch == "e":
            self.show_evidence = not self.show_evidence
        elif ch == "g" and self.gen is not None:
            self.show_pred = not self.show_pred
        return True


def cmd_play(ds: Dataset, args) -> None:
    Player(ds, ds.resolve(args.uid), args).run()


# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"dataset root (default: {DEFAULT_ROOT})",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    gen_p = argparse.ArgumentParser(add_help=False)
    gen_p.add_argument(
        "--gen",
        type=Path,
        help="generation/results json from evaluate_ucsbench_vlm.py to overlay",
    )

    p = sub.add_parser("stats", help="dataset summary")
    p.add_argument(
        "--missing",
        action="store_true",
        help="also list videos/QA files without a counterpart",
    )
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("list", help="list videos", parents=[gen_p])
    p.add_argument(
        "--gen-only",
        action="store_true",
        help="with --gen: only videos that have predictions",
    )
    p.add_argument(
        "--category",
        help="only videos with a question in this category (substring match)",
    )
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("show", help="print the questions of a video", parents=[gen_p])
    p.add_argument("uid", help="video uid (unique prefix ok)")
    p.add_argument(
        "--wrong",
        action="store_true",
        help="with --gen: only wrongly answered questions",
    )
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("play", help="replay a video with QA overlays", parents=[gen_p])
    p.add_argument("uid", help="video uid (unique prefix ok)")
    p.add_argument(
        "--start", type=parse_ts, default=0.0, help="start time, seconds or HH:MM:SS"
    )
    p.add_argument(
        "--question", help="start just before this question id (e.g. 01020_1_T1)"
    )
    p.add_argument(
        "--lead-in",
        type=float,
        default=10.0,
        help="seconds before --question to start (default 10)",
    )
    p.add_argument("--start-paused", action="store_true")
    p.add_argument(
        "--no-autopause",
        action="store_true",
        help="don't pause when reaching a question timestamp",
    )
    p.add_argument("--max-width", type=int, default=1100)
    p.add_argument("--max-height", type=int, default=720)
    p.set_defaults(fn=cmd_play)

    args = ap.parse_args(argv)
    try:
        ds = Dataset(args.root)
        if getattr(args, "gen", None):
            ds.gen = Generations(args.gen)
        args.fn(ds, args)
    except (BrokenPipeError, OSError) as e:  # e.g. `... | head` closed stdout
        if isinstance(e, BrokenPipeError) or getattr(e, "errno", None) == 22:
            sys.stderr.close()
            return
        raise


if __name__ == "__main__":
    main()
