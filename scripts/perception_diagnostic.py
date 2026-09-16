"""Real SAM3 geometry audit against hidden truth, outside the policy."""

import sys, json, argparse
from pathlib import Path
import numpy as np
import imageio.v3 as iio

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from graspbench.env import GraspEnv
from graspbench.config import TaskSpec
from policies.grounding import Perception

p = argparse.ArgumentParser()
p.add_argument("--task", default="task3_sort_public.json")
p.add_argument("--episode", type=int, default=2)
a = p.parse_args()
t = TaskSpec.from_dict(json.loads((ROOT / "configs" / a.task).read_text())["episodes"][a.episode])
out = ROOT / "runs" / ("perception_" + t.episode_id)
out.mkdir(parents=True, exist_ok=True)
with GraspEnv() as env:
    o = env.reset(t)
    names = list(t.active_objects) + ["square_tray", "round_tray"]
    iio.imwrite(out / "overhead.png", o.cameras[0].rgb)
    e = Perception(env.model).estimate(o, names)
    rows = []
    for n in names:
        truth = env.data.xpos[env.model.body(n).id]
        r = {
            "name": n,
            "center": e[n].center.tolist(),
            "size": e[n].size.tolist(),
            "rotation": e[n].rotation.tolist(),
            "truth": truth.tolist(),
            "xy_error_mm": float(np.linalg.norm(e[n].center[:2] - truth[:2]) * 1000),
            "evidence": e[n].evidence,
        }
        rows.append(r)
        print(json.dumps(r), flush=True)
    (out / "geometry.json").write_text(json.dumps(rows, indent=2))
