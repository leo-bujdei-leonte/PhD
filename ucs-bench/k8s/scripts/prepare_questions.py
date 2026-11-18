#!/usr/bin/env python3
"""Convert UCS-Bench QAs/*-mcq.json into the JSONL that evaluate_ucsbench_vlm.py expects,
and split the videos into shards for graph building.

Usage: prepare_questions.py <dataset_dir> <out_dir> [--shards 8] [--subset 10]

Writes to <out_dir>:
  questions_full.jsonl      every question whose video exists
  questions_subset.jsonl    questions for the subset videos only
  videos_subset.txt         subset video ids, one per line
  shards/shard-<i>.txt      full-run video ids per shard (balanced by file size)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

DIMENSIONS = {
    "Ego-centric Position & Orientation Memory": "position_orientation",
    "Embodied trajectory & Movement Memory": "trajectory_movement",
    "Perceived Proximity & Reachability Memory": "proximity_reachability",
    "Object Recall & Quantity Tracking": "category_quantity",
}
LETTERS = "ABCDE"


def hms_to_seconds(ts: str) -> float:
    parts = [float(p) for p in str(ts).strip().split(":")]
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return secs


def convert(q: dict) -> dict:
    opts = q["options"]
    letters = sorted(opts)
    if letters != list(LETTERS[: len(letters)]):
        raise ValueError(f"unexpected option letters {letters}")
    label = str(q["answer_label"]).strip()
    return {
        "video_uid": q["video_uid"].strip(),
        "qid": q["qid"],
        "query_timestamp": hms_to_seconds(q["question_timestamps"]),
        "question": q["question"],
        "options": [f"{k}. {opts[k]}" for k in letters],
        "answer_idx": letters.index(label),
        "dimension": DIMENSIONS[q["category"]],
        "category": q["category"],
        "subcategory": q.get("subcategory"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--subset", type=int, default=10)
    args = ap.parse_args()

    videos = {p.stem: p.stat().st_size for p in args.dataset_dir.glob("*.mp4")}
    rows: list[dict] = []
    bad: Counter[str] = Counter()
    for f in sorted((args.dataset_dir / "QAs").glob("*.json")):
        for q in json.loads(f.read_text(encoding="utf-8")):
            try:
                r = convert(q)
            except Exception as exc:  # noqa: BLE001
                bad[f"{type(exc).__name__}: {exc}"] += 1
                continue
            if r["video_uid"] not in videos:
                bad["video file missing"] += 1
                continue
            rows.append(r)
    rows.sort(key=lambda r: (r["video_uid"], r["query_timestamp"], r["qid"]))

    by_vid: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_vid[r["video_uid"]].append(r)

    # Subset: one video per size decile band (10th-55th percentile), most questions first,
    # so it covers several sources without the multi-GB videos.
    ranked = sorted(by_vid, key=lambda v: videos[v])
    lo, hi = int(len(ranked) * 0.10), int(len(ranked) * 0.55)
    band = ranked[lo:hi]
    step = max(1, len(band) // args.subset)
    subset = []
    for i in range(0, len(band), step):
        group = band[i : i + step]
        subset.append(max(group, key=lambda v: (len(by_vid[v]), v)))
        if len(subset) == args.subset:
            break

    # Full run: greedy longest-processing-time sharding by file size.
    loads = [0] * args.shards
    shards: list[list[str]] = [[] for _ in range(args.shards)]
    for v in sorted(by_vid, key=lambda v: -videos[v]):
        i = loads.index(min(loads))
        shards[i].append(v)
        loads[i] += videos[v]

    out = args.out_dir
    (out / "shards").mkdir(parents=True, exist_ok=True)

    def dump(path: Path, items: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for it in items:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    dump(out / "questions_full.jsonl", rows)
    dump(out / "questions_subset.jsonl", [r for v in subset for r in by_vid[v]])
    (out / "videos_subset.txt").write_text("\n".join(subset) + "\n")
    for i, s in enumerate(shards):
        (out / "shards" / f"shard-{i}.txt").write_text("\n".join(s) + "\n")

    print(f"questions: {len(rows)} converted, skipped {sum(bad.values())}: {dict(bad)}")
    print(f"videos with questions: {len(by_vid)}")
    print("dimensions:", dict(Counter(r["dimension"] for r in rows)))
    print(
        f"subset: {len(subset)} videos, {sum(len(by_vid[v]) for v in subset)} questions, "
        f"{sum(videos[v] for v in subset) / 1e9:.2f} GB"
    )
    for v in subset:
        print(f"  {v:45s} {videos[v] / 1e6:8.1f} MB {len(by_vid[v]):4d} q")
    for i, s in enumerate(shards):
        print(
            f"shard {i}: {len(s):3d} videos {loads[i] / 1e9:6.2f} GB "
            f"{sum(len(by_vid[v]) for v in s):5d} q"
        )


if __name__ == "__main__":
    main()
