"""Tests for ucs-bench/inspect_generations.py.  Run with `pytest` from the repo root."""

from __future__ import annotations

import itertools
import json
import sys
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ucs-bench"))

import inspect_generations as ig
from inspect_ucsbench import DEFAULT_ROOT, Dataset, Generations, Prediction, Question

FPS = 10
N_FRAMES = 60  # the synthetic video is 6 s long
LEVEL = 4  # frame i is filled with grey level i * LEVEL


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def qa(uid: str, qid: str, ts: str, question: str, answer: str, evidence=None) -> dict:
    return {
        "video_uid": uid,
        "video_length": "00:00:06",
        "qid": qid,
        "category": "Ego-centric Position & Orientation Memory",
        "subcategory": "Relative Direction",
        "question": question,
        "question_timestamps": ts,
        "options": {
            "A": "left",
            "B": "right",
            "C": "ahead",
            "D": "behind",
            "E": "above",
        },
        "answer_label": answer,
        "answer_text": "",
        "evidence": evidence or [],
    }


def pred(
    uid: str,
    ts: float,
    question: str,
    gt: str,
    letter: str | None,
    raw: str | None = None,
) -> dict:
    return {
        "video_uid": uid,
        "query_timestamp": ts,
        "question": question,
        "dimension": "position_orientation",
        "gt_idx": "ABCDE".index(gt),
        "predicted_letter": letter,
        "raw_answer": raw if raw is not None else letter,
        "correct": letter == gt,
    }


def write_video(path: Path) -> list[float]:
    """Write a video whose frames are flat grey levels.

    Returns each frame's mean as decoded sequentially, the reference for "frame
    i" (OpenCV's mp4v writer shifts the content by a frame, so comparing with
    what was written would test the encoder, not the reader).
    """
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
    assert out.isOpened()
    for i in range(N_FRAMES):
        out.write(np.full((48, 64, 3), i * LEVEL, np.uint8))
    out.release()
    cap = cv2.VideoCapture(str(path))
    means = []
    while (frame := cap.read()[1]) is not None:
        means.append(float(frame.mean()))
    cap.release()
    assert len(means) == N_FRAMES
    assert min(b - a for a, b in itertools.pairwise(means)) > 1  # frames tell apart
    return means


def frame_index(frame, means: list[float]) -> int:
    m = float(frame.mean())
    return min(range(len(means)), key=lambda i: abs(means[i] - m))


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> SimpleNamespace:
    """Dataset root + generation file.

    vid1 (has a video): Q one @1 s (model right), Q two and Q three sharing @3 s
    (wrong, right), Q four @5 s (no prediction), Q five without a timestamp.
    vid2: predictions but no video file.  vid3: no predictions.
    """
    base = tmp_path_factory.mktemp("ucs")
    root = base / "UCS-Bench"
    (root / "QAs").mkdir(parents=True)
    vid1 = [  # deliberately not in time order
        qa("vid1", "vid1_4", "00:00:05", "Q four", "E"),
        qa("vid1", "vid1_2", "00:00:03", "Q two", "A", [{"description": "turned left", "timestamps": {"start": "00:00:02", "end": "00:00:02"}}]),
        qa("vid1", "vid1_5", "??", "Q five", "C"),
        qa("vid1", "vid1_1", "00:00:01", "Q one", "B"),
        qa("vid1", "vid1_3", "00:00:03", "Q three", "D"),
    ]  # fmt: skip
    (root / "QAs" / "vid1-mcq.json").write_text(json.dumps(vid1), encoding="utf-8")
    (root / "QAs" / "vid2-mcq.json").write_text(
        json.dumps([qa("vid2", "vid2_1", "00:00:02", "Other", "A")])
    )
    (root / "QAs" / "vid3-mcq.json").write_text(
        json.dumps([qa("vid3", "vid3_1", "00:00:02", "Unused", "A")])
    )
    means = write_video(root / "vid1.mp4")

    gen = base / "results.json"
    preds = [
        pred("vid1", 1.0, "Q one", "B", "B"),
        pred("vid1", 3.0, "Q two", "A", "C", raw="The answer is C."),
        pred("vid1", 3.0, "Q three", "D", "D"),
        pred("vid2", 2.0, "Other", "A", "A"),
    ]
    gen.write_text(
        json.dumps(
            {"model": "test/model", "overall_accuracy": 0.75, "predictions": preds}
        )
    )
    return SimpleNamespace(root=root, gen=gen, means=means)


@pytest.fixture
def ds(data_dir) -> Dataset:
    d = Dataset(data_dir.root)
    d.gen = Generations(data_dir.gen)
    return d


@pytest.fixture(scope="session")
def tk_session():
    # One interpreter per session: creating and destroying many tk.Tk() in a
    # process is flaky (Tcl sometimes fails to find its init scripts again).
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"Tk unavailable: {e}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def tk_root(tk_session):
    """A fresh hidden window per test."""
    win = tk.Toplevel(tk_session)
    win.withdraw()
    yield win
    win.destroy()
    tk_session.update()


def make_questions(*times: float | None) -> list[Question]:
    return [
        Question(
            f"q{i}", "cat", "sub", f"question {i}", t, {"A": "a", "B": "b"}, "A", "a"
        )
        for i, t in enumerate(times)
    ]


def stopped_at_shared(st: ig.PlaybackState) -> ig.PlaybackState:
    """Play from the start to the stop at frame 30 (questions 1 and 2)."""
    st.play_pause()
    st.advance(100)  # stops at frame 10
    st.play_pause()
    st.advance(100)  # stops at frame 30
    return st


@pytest.fixture
def st() -> ig.PlaybackState:
    # questions at 1 s, 3 s, 3 s, 5 s -> frames 10, 30, 30, 50
    return ig.PlaybackState(make_questions(1.0, 3.0, 3.0, 5.0), FPS, N_FRAMES)


# --------------------------------------------------------------------------- #
# PlaybackState
# --------------------------------------------------------------------------- #
def test_starts_paused_showing_first_upcoming_question(st):
    assert st.paused and st.frame == 0
    assert st.cursor() == 0
    assert st.status(0) == "upcoming"
    assert st.qframes == [10, 30, 30, 50]


def test_playback_auto_pauses_at_question_timestamp(st):
    st.play_pause()
    assert not st.paused
    assert st.advance(5) == 5
    assert not st.paused and st.cursor() == 0
    assert st.advance(100) == 10  # stops on the question frame, not past it
    assert st.paused and st.at_stop
    assert st.cursor() == 0 and st.status(0) == "now"


def test_play_after_stop_resumes_and_moves_cursor_to_next_question(st):
    st.play_pause()
    st.advance(100)
    st.play_pause()
    assert not st.paused
    assert st.cursor() == 1 and st.status(1) == "upcoming"


def test_shared_timestamp_play_shows_next_question_while_paused(st):
    stopped_at_shared(st)
    assert st.frame == 30 and st.paused
    assert st.cursor() == 1 and st.remaining_at_stop == 1

    st.play_pause()  # second question at the same stop
    assert st.paused and st.frame == 30
    assert st.cursor() == 2 and st.remaining_at_stop == 0

    st.play_pause()  # all shown: resume
    assert not st.paused
    assert st.cursor() == 3
    assert st.advance(100) == 50 and st.cursor() == 3


def test_three_questions_at_one_timestamp():
    st = ig.PlaybackState(make_questions(2.0, 2.0, 2.0), FPS, N_FRAMES)
    st.play_pause()
    st.advance(100)
    seen = [st.cursor()]
    for _ in range(2):
        st.play_pause()
        assert st.paused
        seen.append(st.cursor())
    assert seen == [0, 1, 2]
    st.play_pause()
    assert not st.paused


def test_advance_never_skips_past_a_stop():
    st = ig.PlaybackState(make_questions(1.0, 1.2), FPS, N_FRAMES)
    st.play_pause()
    assert st.advance(59) == 10
    st.play_pause()
    assert st.advance(59) == 12 and st.cursor() == 1


def test_advance_does_nothing_while_paused(st):
    assert st.advance(10) == 0


def test_end_of_video_pauses_and_play_restarts(st):
    st.play_pause()
    for _ in range(10):
        if st.paused:
            st.play_pause()
        st.advance(100)
    assert st.frame == N_FRAMES - 1 and st.paused
    assert st.cursor() == 3 and st.status(3) == "passed"  # stays on the last question
    st.play_pause()
    assert st.frame == 0 and not st.paused
    assert st.cursor() == 0


def test_skip_clears_stop_and_follows_new_position(st):
    stopped_at_shared(st)
    st.skip(-1.5)
    assert st.frame == 15 and st.paused
    assert not st.at_stop and st.remaining_at_stop == 0
    assert st.cursor() == 1  # the 3 s question is upcoming again
    st.skip(100)
    assert st.frame == N_FRAMES - 1
    st.skip(-100)
    assert st.frame == 0


def test_rewinding_replays_stops(st):
    st.play_pause()
    st.advance(100)  # stop at 10
    st.seek(5)
    st.play_pause()
    assert st.advance(100) == 10 and st.paused


def test_seek_onto_timestamp_stops_there_before_playing(st):
    st.seek(30)
    assert st.cursor() == 1 and st.status(1) == "now"
    st.play_pause()
    assert st.paused and st.at_stop and st.frame == 30
    assert st.cursor() == 1 and st.remaining_at_stop == 1


def test_seek_onto_timestamp_while_playing_stops_immediately(st):
    st.seek(20)
    st.play_pause()
    st.seek(50)
    assert st.advance(1) == 50 and st.paused and st.cursor() == 3


def test_browse_moves_cursor_without_moving_video(st):
    st.browse(1)
    assert st.cursor() == 1 and st.browsing
    st.browse(1)
    assert st.cursor() == 2
    assert st.frame == 0 and st.paused
    st.browse(10)
    assert st.cursor() == 3  # clamped
    st.browse(-10)
    assert st.cursor() == 0 and not st.browsing  # back on the followed question


def test_browse_persists_while_playing_until_next_stop(st):
    st.play_pause()
    st.browse(2)
    st.advance(3)
    assert st.cursor() == 2 and st.browsing
    st.advance(100)
    assert st.frame == 10 and st.cursor() == 0 and not st.browsing


def test_play_and_seek_end_browsing(st):
    st.browse(2)
    st.play_pause()
    assert not st.browsing and st.cursor() == 0
    st.browse(2)
    st.seek(40)
    assert not st.browsing and st.cursor() == 3


def test_browse_at_shared_stop_then_play_shows_next_question(st):
    stopped_at_shared(st)
    st.browse(-1)
    assert st.cursor() == 0
    st.play_pause()
    assert st.paused and st.cursor() == 2


def test_select_specific_question(st):
    st.select(3)
    assert st.cursor() == 3 and st.browsing
    st.select(0)
    assert not st.browsing


def test_speed_limits(st):
    assert st.speed == 1.0
    st.faster()
    assert st.speed == 1.5
    for _ in range(10):
        st.slower()
    assert st.speed == ig.SPEEDS[0]
    for _ in range(10):
        st.faster()
    assert st.speed == ig.SPEEDS[-1]


def test_untimed_questions_never_stop_but_can_be_viewed():
    st = ig.PlaybackState(make_questions(1.0, None), FPS, N_FRAMES)
    st.play_pause()
    st.advance(100)
    st.play_pause()
    assert st.advance(100) == N_FRAMES - 1
    assert st.cursor() == 0
    st.select(1)
    assert st.cursor() == 1 and st.status(1) == "untimed"


def test_no_questions():
    st = ig.PlaybackState([], FPS, N_FRAMES)
    assert st.cursor() is None
    st.browse(1)
    st.play_pause()
    assert st.advance(100) == N_FRAMES - 1


def test_question_frames_round_up_to_a_frame():
    st = ig.PlaybackState(make_questions(7.0, 2.5, 100.0), 29.97, 300)
    assert st.qframes == [210, 75, 299]  # 209.79 -> 210; past the end -> last frame
    st10 = ig.PlaybackState([], 10, 100)
    assert st10.frame_of(0.3) == 3  # 0.3 * 10 == 3.0000000000000004
    assert st10.frame_of(-1) == 0


def test_bad_fps_falls_back():
    assert ig.PlaybackState([], 0.0, 10).fps == 30.0
    assert ig.PlaybackState([], float("nan"), 10).fps == 30.0


def test_end_at_shortens_video_and_keeps_late_questions_reachable():
    st = ig.PlaybackState(make_questions(5.5), FPS, N_FRAMES)  # frame 55
    st.play_pause()
    st.advance(52)
    st.end_at(49)  # the decoder has no frames past 49
    assert st.n_frames == 50 and st.frame == 49 and st.paused
    assert st.qframes == [49]
    st.play_pause()
    assert st.at_stop and st.cursor() == 0


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def test_video_rows(ds):
    rows = {r.uid: r for r in ig.video_rows(ds)}
    assert sorted(rows) == ["vid1", "vid2"]  # vid3 has no predictions
    r = rows["vid1"]
    assert (r.n_questions, r.n_correct, r.n_scored) == (5, 2, 3)
    assert r.accuracy == pytest.approx(2 / 3)
    assert r.length == "00:00:06" and r.has_video
    assert not rows["vid2"].has_video


def test_dataset_questions_sorted_with_predictions_attached(ds):
    qs = ds.questions("vid1")
    assert [q.question for q in qs] == ["Q one", "Q two", "Q three", "Q four", "Q five"]
    assert [q.pred.letter if q.pred else None for q in qs] == [
        "B",
        "C",
        "D",
        None,
        None,
    ]


def option_line(lines, letter):
    return next(
        (text, tag)
        for text, tag in lines
        if text.partition("\t")[2].startswith(f"{letter}.")
    )


def test_question_lines_wrong_answer(ds):
    q = ds.questions("vid1")[1]  # Q two: gt A, model C
    lines = ig.question_lines(q)
    assert lines[0] == ("Q two\n", "question")
    text, tag = option_line(lines, "A")
    assert (
        tag == "correct"
        and text.startswith("✓")
        and "correct" in text
        and "model" not in text
    )
    text, tag = option_line(lines, "C")
    assert tag == "wrong" and text.startswith("✗") and "model" in text
    for letter in "BDE":
        assert option_line(lines, letter)[1] == "option"
    assert ("Model answered C: wrong (correct: A)\n", "model_bad") in lines
    assert ("raw output: The answer is C.\n", "raw") in lines
    assert ("Evidence\n", "section") in lines
    assert ("00:00:02-00:00:02  turned left\n", "evidence") in lines


def test_question_lines_right_answer(ds):
    q = ds.questions("vid1")[0]  # Q one: gt B, model B
    lines = ig.question_lines(q)
    text, tag = option_line(lines, "B")
    assert tag == "correct" and "correct + model" in text
    assert ("Model answered B: correct\n", "model_ok") in lines
    assert not any(tag == "raw" for _, tag in lines)  # raw == letter
    assert not any(tag == "wrong" for _, tag in lines)


def test_question_lines_without_prediction(ds):
    q = ds.questions("vid1")[3]  # Q four
    lines = ig.question_lines(q)
    assert ("Model: no prediction for this question\n", "model_none") in lines
    assert [t for t, tag in lines if tag == "correct"] == ["✓\tE. above   ← correct\n"]


def test_question_lines_flags_ground_truth_mismatch_and_long_raw():
    q = make_questions(1.0)[0]
    q.pred = Prediction.from_json(
        {
            "predicted_letter": None,
            "raw_answer": "x" * 500,
            "correct": False,
            "gt_idx": 1,
        }
    )
    lines = ig.question_lines(q, max_raw=50)
    assert ("Model answered ?: wrong (correct: A)\n", "model_bad") in lines
    raw = next(t for t, tag in lines if tag == "raw")
    assert raw.endswith("...\n") and len(raw) == len("raw output: ") + 50 + 1
    assert any(tag == "warn" and "B in the generation file" in t for t, tag in lines)


def test_model_cell(ds):
    qs = ds.questions("vid1")
    assert [ig.model_cell(q) for q in qs] == [
        ("B ✓", "ok"),
        ("C ✗", "bad"),
        ("D ✓", "ok"),
        ("-", "none"),
        ("-", "none"),
    ]


def test_header_and_hint(st):
    assert ig.header_text(st, 0) == "Question 1/4  @ 00:00:01  (in 1.0 s)"
    assert ig.stop_hint(st) == ""
    stopped_at_shared(st)
    assert ig.header_text(st, 1) == "Question 2/4  @ 00:00:03  (now)"
    assert ig.header_text(st, 0).endswith("(passed)")
    assert ig.stop_hint(st).startswith("1 more question at this timestamp")
    st.play_pause()
    assert ig.stop_hint(st).startswith("Stopped at a question")
    st.browse(1)
    assert ig.stop_hint(st).startswith("Browsing")


def test_fmt_delta():
    assert ig.fmt_delta(4.3) == "4.3 s"
    assert ig.fmt_delta(-2) == "0.0 s"
    assert ig.fmt_delta(75) == "1:15"
    assert ig.fmt_delta(3725) == "1:02:05"


# --------------------------------------------------------------------------- #
# Video decoding
# --------------------------------------------------------------------------- #
def test_video_reader_random_access(data_dir):
    r = ig.VideoReader(data_dir.root / "vid1.mp4")
    try:
        assert r.fps == pytest.approx(FPS)
        assert r.n_frames == N_FRAMES
        means = data_dir.means
        for idx in [
            0,
            1,
            2,
            10,
            59,
            5,
            40,
            41,
            0,
            0,
        ]:  # forward hops, backward seeks, repeats
            assert frame_index(r.read(idx), means) == idx
        r.GRAB_LIMIT = 3  # force far forward jumps to seek
        for idx in [10, 30, 31, 35]:
            assert frame_index(r.read(idx), means) == idx
        assert r.read(N_FRAMES) is None
        assert frame_index(r.frame, means) == 35  # last good frame kept
        assert frame_index(r.read(3), means) == 3  # recovers after a failed read
    finally:
        r.close()


def test_video_reader_missing_file(tmp_path):
    with pytest.raises(RuntimeError):
        ig.VideoReader(tmp_path / "nope.mp4")


def test_frame_to_photo_keeps_aspect(tk_root):
    np = pytest.importorskip("numpy")
    img = ig.frame_to_photo(np.zeros((100, 200, 3), np.uint8), 400, 400, tk_root)
    assert (img.width(), img.height()) == (400, 200)


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
def panel_text(app: ig.App) -> str:
    return app.text.get("1.0", "end")


def shown_frame(app: ig.App) -> int:
    assert app.reader is not None
    return frame_index(app.reader.frame, app.test_means)  # type: ignore[attr-defined]


@pytest.fixture
def app(tk_root, ds, data_dir) -> ig.App:
    a = ig.App(tk_root, ds)
    a.test_means = data_dir.means  # type: ignore[attr-defined]
    return a


def test_app_lists_generation_videos(tk_root, app):
    assert app.picker_tree.get_children() == ("vid1", "vid2")
    assert app.picker_tree.item("vid2", "values")[-1] == "missing"
    assert app.picker_tree.selection() == ("vid1",)


def test_app_plays_through_stops(tk_root, app):
    app.open_video("vid1")
    tk_root.update()
    st = app.state
    assert st is not None and st.n_frames == N_FRAMES
    assert panel_text(app).startswith("Q one")
    assert app.qtree.selection() == ("0",)
    assert "upcoming" not in app.q_header.cget("text")  # header says "in 1.0 s"
    assert "in 1.0 s" in app.q_header.cget("text")

    app.play_pause()
    assert "Pause" in app.play_btn.cget("text")
    app.step(100)
    assert st.frame == 10 and st.paused and shown_frame(app) == 10
    assert "Play" in app.play_btn.cget("text")
    assert panel_text(app).startswith("Q one")

    app.play_pause()
    app.step(100)
    tk_root.update()
    assert st.frame == 30 and shown_frame(app) == 30
    assert panel_text(app).startswith("Q two")
    assert app.qtree.selection() == ("1",)
    assert "1 more question" in app.hint_lbl.cget("text")

    app.play_pause()  # same timestamp: next question, still paused
    tk_root.update()
    assert st.paused and st.frame == 30
    assert panel_text(app).startswith("Q three")
    assert app.qtree.selection() == ("2",)

    app.play_pause()
    tk_root.update()
    assert not st.paused
    assert panel_text(app).startswith("Q four")
    assert app.qtree.selection() == ("3",)


def test_app_controls(tk_root, app):
    app.open_video("vid1")
    st = app.state
    assert st is not None
    app.skip(2.0)
    assert st.frame == 20 and shown_frame(app) == 20
    app.faster()
    assert app.speed_lbl.cget("text") == "1.5x"
    app.slower()
    app.slower()
    assert app.speed_lbl.cget("text") == "0.5x"
    assert app.time_lbl.cget("text") == "00:00:02 / 00:00:06"
    app.restart()
    assert st.frame == 0 and shown_frame(app) == 0


def test_app_browse_and_question_row_click(tk_root, app):
    app.open_video("vid1")
    st = app.state
    assert st is not None
    app.browse(1)
    assert panel_text(app).startswith("Q two") and st.frame == 0
    app.qtree.selection_set("3")  # as if the row was clicked
    tk_root.update()
    assert st.cursor() == 3 and panel_text(app).startswith("Q four")
    assert st.frame == 0
    app.browse(1)
    assert panel_text(app).startswith("Q five")
    assert "no timestamp" in app.q_header.cget("text")


def test_app_keyboard(tk_root, app):
    # Synthetic key events don't reach a withdrawn window, so run the bound
    # actions the way the bindings do.
    for seq in app.keys:
        assert tk_root.bind(seq), seq

    def press(seq: str) -> None:
        app._on_key(app.keys[seq])

    press("<space>")  # in the video list: ignored
    assert app.state is None
    app.open_video("vid1")
    st = app.state
    assert st is not None
    press("<Right>")  # +10 s, clamped to the end
    assert st.frame == N_FRAMES - 1
    press("<Home>")
    assert st.frame == 0
    press("<Shift-Right>")
    assert st.frame == N_FRAMES - 1
    press("<Left>")
    assert st.frame == 0
    press("<KeyPress-n>")
    press("<Down>")
    assert st.cursor() == 2
    press("<Up>")
    press("<KeyPress-p>")
    assert st.cursor() == 0
    press("<bracketright>")
    assert st.speed == 1.5
    press("<bracketleft>")
    assert st.speed == 1.0
    press("<KeyPress-n>")
    press("<space>")  # play ends browsing
    assert not st.paused and st.cursor() == 0
    press("<Escape>")
    assert app.state is None


def test_app_timeline_click_seeks(tk_root, app):
    app.open_video("vid1")
    st = app.state
    assert st is not None
    x0, x1 = app._tl_x
    app._on_timeline(SimpleNamespace(x=(x0 + x1) / 2))
    app._do_timeline_seek()
    assert abs(st.frame - (N_FRAMES - 1) / 2) <= 1


def test_app_close_and_reopen(tk_root, app):
    app.open_video("vid1")
    app.close_video()
    assert app.state is None and app.reader is None and app.player is None
    assert app.picker.winfo_manager() == "pack"
    app.open_video("vid1")
    assert app.state is not None and app.state.frame == 0


def test_app_missing_video(tk_root, app):
    with pytest.raises(FileNotFoundError):
        app.open_video("vid2")
    assert app.state is None


def test_app_requires_generations(tk_root, data_dir):
    with pytest.raises(ValueError):
        ig.App(tk_root, Dataset(data_dir.root))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_parse_args_defaults():
    args = ig.parse_args(["gen.json"])
    assert (
        args.gen == Path("gen.json")
        and args.root == DEFAULT_ROOT
        and args.video is None
    )


def test_load(data_dir):
    ds = ig.load(
        ig.parse_args(
            [str(data_dir.gen), "--root", str(data_dir.root), "--video", "vid1"]
        )
    )
    assert ds.gen is not None and ds.gen.model == "test/model"
    assert ds.resolve("vid1") == "vid1"


def test_load_missing_generation_file(data_dir, tmp_path):
    with pytest.raises(SystemExit):
        ig.load(
            ig.parse_args([str(tmp_path / "nope.json"), "--root", str(data_dir.root)])
        )
