"""Laya intent router on a labelled command set (EN + TR): accuracy, latency, and the min_confidence threshold.

    python scripts/laya_eval.py
The threshold is chosen as the lowest value whose accepted commands are >= 95 % correct; below it the
command goes to the Muse Spark deep parse.
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from mercy.voice.commands import INTENT_QUESTION  # noqa: E402

DATA = [
    ("isolate the vehicles", "isolate"), ("highlight the blue hatchback", "isolate"), ("segment the shattered glass", "isolate"),
    ("show me only the second car", "isolate"), ("isolate the traffic light", "isolate"), ("trafik ışığını izole et", "isolate"),
    ("araçları göster", "isolate"),
    ("rewind to three seconds before impact", "seek"), ("go to the impact", "seek"), ("jump to 5 seconds", "seek"),
    ("go back to the start", "seek"), ("skip to when the car started braking", "seek"), ("çarpışma anına git", "seek"),
    ("play", "playback"), ("pause the video", "playback"), ("slow motion", "playback"), ("resume playback", "playback"),
    ("durdur", "playback"), ("play it at half speed", "playback"),
    ("orbit the scene", "camera"), ("show me the top view", "camera"), ("switch to the driver view", "camera"),
    ("fly around the intersection", "camera"), ("üstten göster", "camera"),
    ("zoom in on vehicle two", "zoom"), ("enhance the blue car", "zoom"), ("focus closely on the license plate", "zoom"),
    ("show the braking trajectory of car one", "trajectory"), ("draw the path of the hatchback", "trajectory"),
    ("trace the motion trail of vehicle two", "trajectory"),
    ("who is at fault?", "analyze"), ("analyze the crash", "analyze"), ("who caused this collision", "analyze"),
    ("kaza kimin hatası", "analyze"),
    ("generate the overhead blueprint", "diagram"), ("make a diagram of the scene", "diagram"), ("create the reconstruction image", "diagram"),
    ("note that the light was red", "note"), ("record that the driver did not signal", "note"),
    ("reset the view", "reset"), ("clear everything", "reset"), ("start over", "reset"),
    ("hide the masks", "layers"), ("show the tracks", "layers"), ("turn off the point cloud", "layers"),
    ("open the map", "view"), ("where did this happen", "view"), ("show me the 3D scene", "view"),
    ("go back to the video", "view"), ("open the report", "view"), ("haritayı aç", "view"), ("kanıtları göster", "view"),
    ("zoom in on the licence plate", "zoom"), ("enhance the image", "zoom"), ("read the plate of the white car", "zoom"),
    ("plakayı netleştir", "zoom"), ("make it sharper", "zoom"),
]


def main():
    from laya import Router
    r = Router(preload=False, default="english")
    r.preload(["english", "multilingual"])
    r.predict({"request": "warm up"}, INTENT_QUESTION)
    t0 = time.perf_counter()
    res = r.predict_batch([{"state": {"request": s}, "questions": INTENT_QUESTION} for s, _ in DATA])
    ms = (time.perf_counter() - t0) * 1000 / len(DATA)
    rows = []
    for (s, y), out in zip(DATA, res):
        a = out["answers"]["intent"]
        conf = float(a.get("answer_confidence", a.get("confidence", 0)))
        rows.append((s, y, a["choice"], conf, out["routing"]["model"]))
    correct = np.array([p == y for _, y, p, _, _ in rows]); confs = np.array([c for *_, c, _ in rows])
    print(f"accuracy {correct.mean():.1%} on {len(rows)} commands, {ms:.0f} ms/command (batched, CPU)\n")
    for s, y, p, c, m in rows:
        print(f"  {'✓' if p == y else '✗'} {c:.2f} {m[:5]:5s} {y:10s} -> {p:10s} {s}")
    best = None
    for th in np.arange(0.3, 0.96, 0.05):
        acc = correct[confs >= th]
        if len(acc) and acc.mean() >= 0.95:
            best = th; break
    th = best if best is not None else 0.9
    cov = (confs >= th).mean()
    print(f"\nchosen min_confidence {th:.2f}: accepted {cov:.0%} of commands at "
          f"{correct[confs >= th].mean():.0%} accuracy; the rest go to keyword rules, then the Muse Spark deep parse.")
    from mercy.voice.commands import rule_intent
    from mercy import config
    th = config.LAYA_MIN_CONFIDENCE
    final = [rule_intent(s) or (p if c >= th else "deep") for (s, y, p, c, _) in rows]
    ok = [f == y for f, (_, y, *_) in zip(final, rows)]
    deep = sum(f == "deep" for f in final)
    wrong = [(s, y, f) for f, (s, y, *_) in zip(final, rows) if f != y and f != "deep"]
    print(f"production router (rules > Laya @ {th:.2f} > deep parse): {np.mean(ok):.1%} correct without any LLM call, "
          f"{deep} of {len(rows)} sent to deep parse, {len(wrong)} wrong: {wrong}")


if __name__ == "__main__":
    main()
