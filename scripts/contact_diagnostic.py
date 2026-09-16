"""Original async evaluator plus hidden-state diagnostics written ONLY to a file.

The production policy still receives only its ordinary calibrated observations.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from graspbench.env import GraspEnv
from graspbench.evaluate import load_policy, load_tasks
from graspbench.runner import run_episode


class TraceEnv(GraspEnv):
    def step(self, command):
        result = super().step(command)
        objects = {}
        for name in self.task.active_objects:
            body = self.model.body(name).id
            joint = self._object_joint_ids[name]
            adr = self.model.jnt_dofadr[joint]
            objects[name] = {
                "position": self.data.xpos[body].tolist(),
                "rotation": self.data.xmat[body].reshape(3, 3).tolist(),
                "velocity": self.data.qvel[adr : adr + 6].tolist(),
            }
        contacts = []
        for c in self.data.contact:
            a = self.model.body(int(self.model.geom_bodyid[c.geom1])).name
            b = self.model.body(int(self.model.geom_bodyid[c.geom2])).name
            if a in self.task.active_objects or b in self.task.active_objects:
                contacts.append([a, b, float(c.dist)])
        self.trace.write(
            json.dumps(
                {
                    "time": float(self.data.time),
                    "objects": objects,
                    "contacts": contacts,
                    "goals": self.task_goal_status(),
                }
            )
            + "\n"
        )
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--video", action="store_true")
    args = p.parse_args()
    task = load_tasks(ROOT / "configs" / args.task)[args.episode]
    output = ROOT / "runs" / args.output
    output.mkdir(parents=True, exist_ok=True)
    with TraceEnv() as env, (output / "diagnostic_truth.jsonl").open("w") as trace:
        env.trace = trace
        summary = run_episode(
            env,
            load_policy("policies.student_policy:StudentPolicy"),
            task,
            output_dir=output,
            max_steps=task.control_budget,
            record_video=args.video,
            async_decisions=True,
            realtime=True,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
