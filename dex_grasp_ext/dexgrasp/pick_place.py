"""Closed loop for the extension: perceive -> grasp -> transport -> verify."""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

from .config import HAND_OPEN, PERCEPTION_PARK_XYZ, TABLE_Z
from .env import DexEnv
from .experiment import (
    VideoRecorder,
    cage_solution,
    count_object_finger_contacts,
    move_tcp,
    move_tcp_with_fallback,
    release_and_retract,
    servo_to,
)
from .hand import HandSynergy, tcp_position
from .ik import ArmIK
from .perception import PerceptionError, Sam3Client, detect_object, render_camera
from .planning import GraspPlan, plan_grasp, rank_plans

TRAY_INNER = {"square_tray": 0.114, "round_tray": 0.112}


@dataclass
class EpisodeResult:
    object: str
    destination: str
    closure: float
    yaw_deg: float
    plan_source: str | None = None
    cage_radius_m: float | None = None
    perceived_object: list[float] | None = None
    perceived_tray: list[float] | None = None
    perception_error_m: float | None = None
    lifted: bool = False
    lift_m: float = 0.0
    placed_in_tray: bool = False
    placement_offset_m: float | None = None
    object_drift_after_close_m: float | None = None
    finger_contacts: int = 0
    sam3_calls: int = 0
    sam3_latency_s: list[float] = field(default_factory=list)
    stages: list[str] = field(default_factory=list)
    failure: str | None = None
    attempts: list[dict] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.lifted and self.placed_in_tray

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["success"] = self.success
        return payload


def perceive(env: DexEnv, ik: ArmIK, client: Sam3Client, name: str, camera: str = "overhead"):
    """Park the arm out of the view, take a shot and segment `name`."""
    move_tcp(ik, env, PERCEPTION_PARK_XYZ, HandSynergy(HAND_OPEN), steps=350)
    frame = render_camera(env.model, env.data, camera)
    return detect_object(client, frame, name), frame


def prepare_scene(
    env: DexEnv,
    object_name: str,
    *,
    x: float = 0.45,
    y: float = 0.02,
    settle_steps: int = 500,
    ik: ArmIK | None = None,
) -> np.ndarray:
    """Reset to a single-object table-top scene and return the object's pose.

    Every grasp candidate must start from the same state: the previous attempt
    may have left the object in the hand or the arm parked inside the camera's
    view, both of which would silently change the next trial.
    """
    env.park_objects(keep=object_name)
    env.reset(settle_steps=50)
    if ik is not None:
        # Move the arm out of the workspace *before* dropping the object in:
        # spawning under the resting hand lets spheres get nudged off the table.
        move_tcp(ik, env, PERCEPTION_PARK_XYZ, HandSynergy(HAND_OPEN), steps=400)
    env.place_object(object_name, [x, y, TABLE_Z + 0.06])
    env.reset(settle_steps=settle_steps)
    if ik is not None:
        move_tcp(ik, env, PERCEPTION_PARK_XYZ, HandSynergy(HAND_OPEN), steps=300)
    return env.object_position(object_name).copy()


def pick_and_lift(
    env: DexEnv,
    ik: ArmIK,
    client: Sam3Client,
    object_name: str,
    *,
    plan: GraspPlan | None = None,
    closure: float | None = None,
    yaw_deg: float = 0.0,
    approach: float = 0.10,
    lift: float = 0.12,
    recorder: VideoRecorder | None = None,
    result: EpisodeResult | None = None,
) -> EpisodeResult:
    """Perceive the object, grasp it and lift it.  Shared by synthesis + episodes."""
    plan = plan or plan_grasp(object_name, yaw_deg=yaw_deg, closure_override=closure)
    result = result or EpisodeResult(
        object=object_name, destination="", closure=plan.closure, yaw_deg=plan.yaw_deg
    )
    result.closure = plan.closure
    result.yaw_deg = plan.yaw_deg
    result.plan_source = plan.source
    result.cage_radius_m = plan.radius_m

    # ---------------------------------------------------------------- perceive
    try:
        object_detection, _ = perceive(env, ik, client, object_name)
    except PerceptionError as exc:
        result.failure = f"perception: {exc}"
        return result
    result.perceived_object = object_detection.position.tolist()
    result.sam3_latency_s.append(object_detection.latency_s)
    truth = env.object_position(object_name).copy()
    result.perception_error_m = float(np.linalg.norm(object_detection.position - truth))
    result.stages.append("perceive")

    # -------------------------------------------------------------- grasp plan
    _, palm_rotation = ik.forward(env.data.qpos[: env.arm_dof])
    if plan.yaw_deg:
        palm_rotation = _yaw_about_world_z(palm_rotation, np.deg2rad(plan.yaw_deg))
    target = object_detection.position.copy()
    above = target + np.array([0.0, 0.0, approach])
    centre = plan.centre_local

    # Transit high above the workspace first: joint-space servoing straight from
    # the perception park pose can sweep the fingers through the object.
    transit = np.array([target[0], target[1], TABLE_Z + 0.32])
    if not move_tcp_with_fallback(
        ik,
        env,
        transit,
        HandSynergy(HAND_OPEN),
        steps=600,
        recorder=recorder,
        centre_local=centre,
    ):
        result.failure = "transit IK failed"
        return result

    approach_solution = cage_solution(ik, env, above, palm_rotation, centre_local=centre)
    if not approach_solution.converged:
        result.failure = "approach IK failed"
        return result
    servo_to(env, approach_solution.joint_position, HandSynergy(HAND_OPEN), recorder=recorder)

    grasp_solution = cage_solution(ik, env, target, palm_rotation, centre_local=centre)
    if not grasp_solution.converged:
        result.failure = "grasp IK failed"
        return result
    servo_to(env, grasp_solution.joint_position, HandSynergy(HAND_OPEN), steps=500, recorder=recorder)
    result.stages.append("approach")

    servo_to(
        env, grasp_solution.joint_position, HandSynergy(plan.closure), steps=900, recorder=recorder
    )
    result.finger_contacts = count_object_finger_contacts(env, object_name)
    result.object_drift_after_close_m = float(
        np.linalg.norm(env.object_position(object_name) - truth)
    )
    result.stages.append("close")

    lift_solution = cage_solution(
        ik, env, above + np.array([0.0, 0.0, lift]), palm_rotation, centre_local=centre
    )
    if not lift_solution.converged:
        result.failure = "lift IK failed"
        return result
    servo_to(
        env, lift_solution.joint_position, HandSynergy(plan.closure), steps=800, recorder=recorder
    )
    held_position = env.object_position(object_name).copy()
    result.lift_m = float(held_position[2] - truth[2])
    result.lifted = result.lift_m > 0.04
    result.stages.append("lift")
    if not result.lifted:
        result.failure = "object not lifted"
    return result


def run_episode(
    env: DexEnv,
    ik: ArmIK,
    client: Sam3Client,
    object_name: str,
    destination: str,
    *,
    plan: GraspPlan | None = None,
    retry_plans: list[GraspPlan] | None = None,
    closure: float = 0.6,
    yaw_deg: float = 0.0,
    approach: float = 0.10,
    lift: float = 0.12,
    release_gap: float = 0.015,
    recorder: VideoRecorder | None = None,
    out_dir: pathlib.Path | None = None,
) -> EpisodeResult:
    plan = plan or plan_grasp(object_name, yaw_deg=yaw_deg, closure_override=closure)
    result = EpisodeResult(
        object=object_name, destination=destination, closure=plan.closure, yaw_deg=plan.yaw_deg
    )

    # Both the object and its destination are perceived up front, while the
    # table is still clear: once the object is held, the hand occludes the tray.
    try:
        tray_detection, _ = perceive(env, ik, client, destination)
    except PerceptionError as exc:
        result.failure = f"tray perception: {exc}"
        _write(out_dir, result, client)
        return result
    result.perceived_tray = tray_detection.position.tolist()
    result.sam3_latency_s.append(tray_detection.latency_s)

    # Try the best plan first; on a failed lift, re-perceive and retry with the
    # next-best candidate instead of repeating the same motion.
    candidates = [plan] + [
        candidate for candidate in (retry_plans or []) if candidate is not plan
    ]
    for attempt_index, candidate in enumerate(candidates):
        attempt = EpisodeResult(
            object=object_name, destination=destination,
            closure=candidate.closure, yaw_deg=candidate.yaw_deg,
        )
        pick_and_lift(
            env,
            ik,
            client,
            object_name,
            plan=candidate,
            approach=approach,
            lift=lift,
            recorder=recorder,
            result=attempt,
        )
        _merge(result, attempt)
        result.attempts.append(
            {
                "closure": candidate.closure,
                "yaw_deg": candidate.yaw_deg,
                "lifted": attempt.lifted,
                "lift_m": attempt.lift_m,
                "drift_m": attempt.object_drift_after_close_m,
                "failure": attempt.failure,
            }
        )
        if attempt.lifted:
            result.lifted = True
            result.closure = candidate.closure
            result.yaw_deg = candidate.yaw_deg
            result.plan_source = candidate.source
            result.cage_radius_m = candidate.radius_m
            result.failure = None
            break
        result.failure = attempt.failure or "object not lifted"
        if attempt.failure and "perception" in attempt.failure:
            break  # the object is not visible any more: retrying cannot help
    if not result.lifted:
        _write(out_dir, result, client)
        return result

    # -------------------------------------------------------------- transport
    tray_center = tray_detection.position
    hover = np.array([tray_center[0], tray_center[1], TABLE_Z + 0.26])
    if not move_tcp_with_fallback(
        ik,
        env,
        hover,
        HandSynergy(plan.closure),
        heights=(0.0, -0.05, -0.10, -0.16),
        steps=700,
        recorder=recorder,
        centre_local=plan.centre_local,
    ):
        result.failure = "transport IK failed"
        _write(out_dir, result, client)
        return result
    result.stages.append("transport")

    # ---------------------------------------------------------------- release
    resting_centre = _tray_resting_height(env, object_name, destination)
    place_height = resting_centre + release_gap
    if not move_tcp_with_fallback(
        ik,
        env,
        np.array([tray_center[0], tray_center[1], place_height]),
        HandSynergy(plan.closure),
        heights=(0.0, -0.03, -0.06),
        steps=600,
        recorder=recorder,
        centre_local=plan.centre_local,
    ):
        result.failure = "place IK failed"
        _write(out_dir, result, client)
        return result

    retreat = np.array([tray_center[0], tray_center[1], TABLE_Z + 0.22])
    release_and_retract(
        ik,
        env,
        retreat,
        start_closure=plan.closure,
        steps=600,
        recorder=recorder,
        centre_local=plan.centre_local,
    )
    result.stages.append("retreat")

    # ----------------------------------------------------------------- verify
    try:
        verify_detection, _ = perceive(env, ik, client, object_name)
    except PerceptionError as exc:
        result.failure = f"verification perception: {exc}"
        _write(out_dir, result, client)
        return result
    result.sam3_latency_s.append(verify_detection.latency_s)
    offset = float(np.linalg.norm(verify_detection.position[:2] - tray_center[:2]))
    result.placement_offset_m = offset
    result.placed_in_tray = offset < TRAY_INNER[destination] - 0.02
    if not result.placed_in_tray:
        result.failure = "object not verified inside the tray"
    result.stages.append("verify")
    _write(out_dir, result, client)
    return result


def _write(out_dir: pathlib.Path | None, result: EpisodeResult, client: Sam3Client | None = None) -> None:
    if out_dir is None:
        return
    if client is not None:
        result.sam3_calls = client.calls
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episode.json").write_text(json.dumps(result.as_dict(), indent=2) + "\n")


def _merge(target: EpisodeResult, source: EpisodeResult) -> None:
    """Copy the per-attempt observations into the episode-level record."""
    for name in (
        "perceived_object",
        "perception_error_m",
        "finger_contacts",
        "object_drift_after_close_m",
        "lift_m",
    ):
        value = getattr(source, name)
        if value is not None:
            setattr(target, name, value)
    target.stages = source.stages
    target.sam3_latency_s.extend(source.sam3_latency_s)


def _yaw_about_world_z(rotation: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a tool frame about the world vertical axis (grasp yaw search)."""
    cosine, sine = np.cos(angle), np.sin(angle)
    yaw = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    return yaw @ rotation


def _tray_resting_height(env: DexEnv, object_name: str, destination: str) -> float:
    """Where the object's centre sits once it rests on the tray floor."""
    from .perception import CATALOGUE

    half_height = float(CATALOGUE.get(object_name, {}).get("half_height", 0.03))
    tray_floor = TABLE_Z + 0.016  # tray floor panel sits on the table top
    return tray_floor + half_height
