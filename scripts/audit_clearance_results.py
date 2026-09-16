"""Summarize path heights and pre-pick disturbances from evaluator evidence."""
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
reports = {}
robot = {"link" + str(i) for i in range(8)} | {"hand", "left_finger", "right_finger"}
for folder in sorted(root.iterdir()):
    log = folder / "events.jsonl"
    if not log.exists():
        continue
    events = [json.loads(line) for line in log.read_text().splitlines()]
    contacts = root / (folder.name + "_contacts.jsonl")
    truth = [json.loads(line) for line in contacts.read_text().splitlines()] if contacts.exists() else []
    first_descent = {}
    heights = {}
    for event in events:
        decision = event["decision"]
        stage = decision["stage"]
        target = decision["target_id"]
        if stage == "descend":
            first_descent.setdefault(target, event["observation"]["time"])
        if stage in ("approach_high", "transfer", "observe_clear", "park_high"):
            heights.setdefault(stage, []).append(event["observation"]["ee_position"][2])
    report = {"travel_height_min_m": {k: min(v) for k, v in heights.items()}}
    disturbances = {}
    if truth:
        for name, state in truth[0]["objects"].items():
            start = np.array(state["position"])
            before = [t for t in truth if t["time"] < first_descent.get(name, float("inf"))]
            hits = [
                {"time": t["time"], "contact": c}
                for t in before for c in t["contacts"]
                if name in c[:2] and (
                    c[0] in robot or c[1] in robot
                    or (c[0] in t["objects"] and c[1] in t["objects"] and c[0] != c[1])
                )
            ]
            disturbances[name] = {
                "pre_pick_robot_or_object_contacts": len(hits),
                "contact_examples": hits[:5],
                "max_pre_pick_displacement_m": max(
                    (float(np.linalg.norm(np.array(t["objects"][name]["position"]) - start))
                     for t in before), default=0.0),
            }
    report["pre_pick_disturbance"] = disturbances
    summary = folder / "summary.json"
    if summary.exists():
        s = json.loads(summary.read_text())
        report.update({k: s[k] for k in ("success", "control_steps", "terminal_reason", "goal_status")})
    reports[folder.name] = report
(root / "clearance_audit.json").write_text(json.dumps(reports, indent=2))
print(json.dumps(reports, indent=2))
