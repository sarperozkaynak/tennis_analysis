#!/usr/bin/env python3
"""
Tennis analysis output video evaluator.

Samples frames from an output video, sends them to Claude vision,
and reports what is working and what needs fixing — exactly as a human
reviewer would after watching the video.

Usage:
    python3 evaluate_output.py output_videos/tennis_sarp_output.avi
    python3 evaluate_output.py output_videos/tennis_sarp_output.avi --frames 10
    python3 evaluate_output.py output_videos/tennis_sarp_output.avi --interval 60
"""

import anthropic
import base64
import sys
import argparse
import cv2
import numpy as np


EVAL_PROMPT = """\
This is frame {frame_num} (of {total}) from a tennis match analysis video.
The pipeline overlays the following onto the raw footage:
  • Red bounding boxes + "Player ID: N" labels for each detected player
  • Yellow bounding box + "Ball ID: 1" label for the ball
  • Red numbered dots (0–13) for court keypoints (should sit on actual court lines)
  • A small mini-court diagram in the top-right corner showing player/ball positions
  • A stats panel (top-right) showing shot speed and player speed for Player 1 and Player 2

Evaluate each aspect and respond with a JSON object with these exact keys:

{{
  "court_keypoints": "ok" | "wrong" | "partial",
  "court_keypoints_detail": "<one sentence — where are the dots, are they on court lines?>",
  "player1": "ok" | "missing" | "wrong_location" | "wrong_size",
  "player1_detail": "<one sentence>",
  "player2": "ok" | "missing" | "wrong_location" | "wrong_size",
  "player2_detail": "<one sentence>",
  "ball": "ok" | "missing" | "wrong_location",
  "ball_detail": "<one sentence>",
  "mini_court": "ok" | "wrong" | "empty",
  "stats": "ok" | "all_zero" | "nan" | "missing",
  "overall": "good" | "partial" | "broken",
  "issues": ["<concise issue 1>", "<concise issue 2>"]
}}

Return only the JSON object, no markdown fences.
"""


def frame_to_b64(frame: np.ndarray) -> str:
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode()


def evaluate_frame(client: anthropic.Anthropic, frame: np.ndarray,
                   frame_num: int, total: int) -> dict:
    import json
    b64 = frame_to_b64(frame)
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                                              "media_type": "image/jpeg",
                                              "data": b64}},
                {"type": "text",
                 "text": EVAL_PROMPT.format(frame_num=frame_num, total=total)},
            ]
        }]
    )
    text = resp.content[0].text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text, "overall": "unknown"}


def sample_frames(video_path: str, n_frames: int = 8,
                  interval: int | None = None) -> list[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        raise ValueError(f"Cannot read {video_path}")

    if interval:
        indices = list(range(0, total, interval))
    else:
        # Evenly spaced, skip first and last 5% (often blank)
        margin = max(1, int(total * 0.05))
        indices = np.linspace(margin, total - margin - 1, n_frames, dtype=int).tolist()

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))
    cap.release()
    return frames, total


def print_report(results: list[dict]) -> None:
    STATUS = {"ok": "✓", "wrong": "✗", "partial": "~",
              "missing": "–", "wrong_location": "?", "wrong_size": "?",
              "all_zero": "0", "nan": "N", "empty": "–", "broken": "✗",
              "good": "✓", "unknown": "?"}

    print("\n" + "="*72)
    print(f"{'Frame':>6}  {'Court':^7}  {'P1':^6}  {'P2':^6}  {'Ball':^6}  {'Mini':^5}  {'Stats':^5}  Overall")
    print("-"*72)

    all_issues: list[str] = []
    category_counts: dict[str, dict] = {
        "court_keypoints": {}, "player1": {}, "player2": {},
        "ball": {}, "mini_court": {}, "stats": {}, "overall": {}
    }

    for r in results:
        fnum = r.get("frame_num", "?")
        row = f"{fnum:>6}  "
        for key in ["court_keypoints", "player1", "player2", "ball", "mini_court", "stats", "overall"]:
            val = r.get(key, "?")
            sym = STATUS.get(val, "?")
            width = 7 if key == "court_keypoints" else 6 if key in ("player1","player2","ball") else 5
            row += f"{sym:^{width}}  "
            category_counts[key][val] = category_counts[key].get(val, 0) + 1

        print(row.rstrip())
        for issue in r.get("issues", []):
            all_issues.append(f"  frame {fnum}: {issue}")

    print("="*72)

    # Detail lines
    print("\nPer-frame details:")
    for r in results:
        fnum = r.get("frame_num", "?")
        for key in ["court_keypoints", "player1", "player2", "ball"]:
            detail = r.get(f"{key}_detail", "")
            if detail and r.get(key) != "ok":
                print(f"  [{fnum}] {key}: {detail}")

    print("\nIssues found:")
    if all_issues:
        for issue in dict.fromkeys(all_issues):  # deduplicate, preserve order
            print(issue)
    else:
        print("  None — everything looks good!")

    print("\nSummary by category:")
    for cat, counts in category_counts.items():
        dominant = max(counts, key=counts.__getitem__) if counts else "?"
        print(f"  {cat:20s}: {dominant} ({counts})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to output video file")
    parser.add_argument("--frames", type=int, default=8,
                        help="Number of frames to sample (default 8)")
    parser.add_argument("--interval", type=int, default=None,
                        help="Sample every N frames instead of evenly spaced")
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    args = parser.parse_args()

    import os
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Anthropic API key required.")
        print("  Set it with:  export ANTHROPIC_API_KEY=sk-ant-...")
        print("  Or pass it:   --api-key sk-ant-...")
        sys.exit(1)

    print(f"Loading {args.video} …")
    frame_list, total = sample_frames(args.video, args.frames, args.interval)
    print(f"Sampled {len(frame_list)} frames from {total} total.")

    client = anthropic.Anthropic(api_key=api_key)
    results = []
    for frame_num, frame in frame_list:
        print(f"  Evaluating frame {frame_num} …", end=" ", flush=True)
        r = evaluate_frame(client, frame, frame_num, total)
        r["frame_num"] = frame_num
        results.append(r)
        print(r.get("overall", "?"))

    print_report(results)


if __name__ == "__main__":
    main()
