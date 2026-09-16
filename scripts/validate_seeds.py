"""Reproducible real-perception evaluation with an unchanged course runner."""

import argparse
import hashlib
import json
import os
import platform
import sys
import urllib.request
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from graspbench.env import GraspEnv
from graspbench.evaluate import aggregate, load_policy, load_tasks
from graspbench.runner import run_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    tasks = load_tasks(ROOT / args.tasks)
    if args.seeds:
        tasks = [
            replace(t, seed=seed, episode_id=f"{t.episode_id}_seed{seed}")
            for t in tasks
            for seed in args.seeds
        ]
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((ROOT / "original_files.sha256.json").read_text())
    for name, expected in original.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Official baseline was modified: {name}")
    sources = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "policies").glob("*.py"))
    }
    metadata = {
        "sources_sha256": sources,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "execution_mode": "async",
        "realtime": True,
        "original_physics_verified": True,
        "sam3_configured": bool(os.getenv("GRASPBENCH_SAM3_URL")),
        "seeds": [{"episode": t.episode_id, "seed": t.seed} for t in tasks],
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2))
    rows = []
    with GraspEnv() as env:
        for task in tasks:
            endpoint = os.getenv("GRASPBENCH_SAM3_URL", "http://127.0.0.1:8765/infer")
            health_url = endpoint.rsplit("/", 1)[0] + "/health"
            with urllib.request.urlopen(health_url, timeout=5) as response:
                health = json.load(response)
            if not health.get("ready", True):
                raise RuntimeError("SAM3 service is not ready; no episode was started")
            policy = load_policy("policies.student_policy:StudentPolicy")
            row = run_episode(
                env,
                policy,
                task,
                output_dir=output,
                max_steps=task.control_budget,
                record_video=args.video,
                async_decisions=True,
                realtime=True,
            )
            rows.append(row)
            print(task.episode_id, row["success"], row["control_steps"], flush=True)
            report = {**aggregate(rows), "completed": len(rows), "requested": len(tasks)}
            (output / "aggregate.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
