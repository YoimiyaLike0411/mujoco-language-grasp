"""DEV ONLY: ground-truth perception control diagnostic, never a graded evaluation."""

from __future__ import annotations
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from graspbench.config import TaskSpec, OBJECT_SPEC_BY_NAME
from graspbench.env import GraspEnv
from graspbench.types import Observation
from policies.student_policy import StudentPolicy
from policies.grasp_geometry import Estimate


class Oracle:
    def held_center(self, observation, name, near=None):
        return self.env.data.xpos[self.env.model.body(name).id].copy()

    def __init__(self, env):
        from policies.cad_geometry import CadGeometry

        self.env = env
        self.cad = CadGeometry(env.model)

    def estimate(self, observation, names):
        out = {}
        for name in names:
            body = self.env.model.body(name).id
            pos = self.env.data.xpos[body].copy()
            rot = self.env.data.xmat[body].reshape(3, 3).copy()
            size = np.array([0.2, 0.2, 0.01])
            if name in OBJECT_SPEC_BY_NAME:
                geoms = [
                    g
                    for g in range(
                        self.env.model.body_geomadr[body],
                        self.env.model.body_geomadr[body] + self.env.model.body_geomnum[body],
                    )
                    if self.env.model.geom_contype[g]
                ]
                g = geoms[0]
                size = self.env.model.geom_size[g].copy() * 2
                if name == "green_cylinder":
                    size[1] = size[0]
                    size[2] = 0.062
            out[name] = Estimate(name, pos, size, rot, {"service": "oracle_diagnostic_only"})
            if name == "banana":
                out[name].size = np.ptp(self.cad.models[name][0], axis=0)
                out[name].grasp_center, out[name].grasp_rotation, out[name].grasp_width = (
                    self.cad.banana_grasp(pos, rot)
                )
        return out


def run(args):
    if args.task:
        payload = json.loads((ROOT / "configs" / args.task).read_text())["episodes"][args.episode]
        task = replace(TaskSpec.from_dict(payload), seed=args.seed)
    else:
        task = TaskSpec(
            f"dev_{args.object}_{args.seed}",
            f"Place the {args.object.replace('_', ' ')} into the square tray.",
            args.object,
            args.seed,
            destination="square_tray",
            scene_objects=(args.object,),
        )
    if getattr(args, "instruction", None):
        task = replace(task, instruction=args.instruction)
    output = ROOT / "runs" / args.output
    output.mkdir(parents=True, exist_ok=True)
    writer = (
        imageio.get_writer(output / "video.mp4", fps=12.5, codec="libx264", quality=7)
        if args.video
        else None
    )
    with GraspEnv() as env, (output / "events.jsonl").open("w") as events:
        cameras = ()

        # No rendered input is consumed in this explicitly labelled diagnostic.
        def observe():
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, env.data.site_xmat[env.grasp_site_id])
            return Observation(
                float(env.data.time),
                task.instruction,
                env.data.qpos[env.arm_qpos_ids].copy(),
                env.data.qvel[env.arm_dof_ids].copy(),
                env.data.site_xpos[env.grasp_site_id].copy(),
                quat,
                float(np.mean(env.data.qpos[env.finger_qpos_ids]) / 0.04),
                cameras,
            )

        env.observe = observe
        obs = env.reset(task)
        policy = StudentPolicy()
        policy.reset(task.public_dict(), env.model)
        policy.perception = Oracle(env)
        previous = None
        streak = 0
        for step in range(args.steps):
            dec = policy.act(obs)
            obs, info = env.step(dec.command)
            goals = env.task_goal_status()
            streak = streak + 1 if all(goals.values()) else 0
            if step % 10 == 0 or dec.stage != previous:
                row = {
                    "step": step,
                    "stage": dec.stage,
                    "reason": dec.rationale,
                    "motion": policy.motion.last_error,
                    "ee": obs.ee_position.tolist(),
                    "opening": obs.gripper_opening,
                    "velocity": {
                        n: env.data.qvel[
                            env.model.jnt_dofadr[env._object_joint_ids[n]] : env.model.jnt_dofadr[
                                env._object_joint_ids[n]
                            ]
                            + 6
                        ].tolist()
                        for n in task.active_objects
                    },
                    "objects": {
                        n: env.data.xpos[env._object_body_ids[n]].tolist()
                        for n in task.active_objects
                    },
                    "goals": goals,
                    "contacts": [
                        (
                            env.model.body(int(env.model.geom_bodyid[c.geom1])).name,
                            env.model.body(int(env.model.geom_bodyid[c.geom2])).name,
                            float(c.dist),
                        )
                        for c in env.data.contact
                        if env.model.body(int(env.model.geom_bodyid[c.geom1])).name
                        in ("hand", "left_finger", "right_finger")
                        or env.model.body(int(env.model.geom_bodyid[c.geom2])).name
                        in ("hand", "left_finger", "right_finger")
                    ],
                }
                events.write(json.dumps(row) + "\n")
                events.flush()
            if dec.stage != previous:
                print(
                    step,
                    dec.stage,
                    dec.rationale,
                    "ee",
                    np.round(obs.ee_position, 3),
                    "opening",
                    round(obs.gripper_opening, 3),
                    flush=True,
                )
                previous = dec.stage
            if writer and step % 2 == 0:
                from PIL import Image, ImageDraw

                frame = Image.fromarray(env.render(width=960, height=720))
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 960, 38), fill="black")
                draw.text(
                    (8, 5),
                    f"ORIGINAL PHYSICS | ORACLE CONTROL DIAGNOSTIC | seed {args.seed} | step {step}",
                    fill="white",
                )
                draw.text((8, 22), dec.stage + ": " + dec.rationale, fill="white")
                writer.append_data(np.asarray(frame))
            if dec.done or (policy.stage == "settle" and streak >= 10):
                break
        if writer:
            writer.close()
        result = {
            "diagnostic": "oracle; not real perception or async evaluation",
            "success": streak >= 10 and policy.stage == "settle",
            "physical_streak": streak,
            "goals": goals,
            "steps": step + 1,
            "stage": dec.stage,
            "seed": args.seed,
        }
        (output / "summary.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task")
    p.add_argument("--instruction")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--object", default="red_cube")
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--output", default="dev")
    p.add_argument("--video", action="store_true")
    run(p.parse_args())
