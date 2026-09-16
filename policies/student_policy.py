"""Shared measured-state pick/place controller for all three course tasks."""

from __future__ import annotations

import numpy as np

from graspbench.perception import ModelServiceError
from graspbench.types import PolicyDecision

from .fruit_tracking import TrackingPerception as Perception
from .grasp_geometry import Estimate, plan_grasp
from .grounding import parse_actions
from .motion import Motion
from .placed_surface import placed_center


class StudentPolicy:
    def reset(self, task, model):
        self.motion = Motion(model)
        self.perception = Perception(model)
        self.actions = []
        self.instruction = task["instruction"]
        self.index = 0
        self.stage = "initialize"
        self.entered = None
        self.retry = 0
        self.reason = "Read the instruction and ground the current scene"
        self.estimates = {}
        self.grasp = None
        self.validation = {}
        self.verification_q = None
        self.fruit_tracking_evidence = {}
        self.travel_height = 0.64
        self.park_position = None
        self.scene_cache_complete = False

    def enter(self, stage, observation, reason):
        self.stage, self.entered, self.reason = stage, observation.time, reason
        if stage in ("recover", "startup_raise"):
            self.raise_target = observation.ee_position.copy()
            self.raise_target[2] = max(self.raise_target[2], getattr(self, "travel_height", 0.64))
            self.raise_quat = observation.ee_quaternion.copy()
        if stage == "release":
            self.release_opening = float(np.clip(observation.gripper_opening, 0.0, 1.0))
        if stage in ("perceive", "verify"):
            self.verification_q = observation.joint_position.copy()
        if stage == "verify":
            self.verification_centers = []

    def fail(self, observation, reason):
        self.retry += 1
        self.enter("recover" if self.retry <= 2 else "safe_stop", observation, reason)

    def act(self, observation):
        if self.entered is None:
            self.entered = observation.time
        age = observation.time - self.entered
        opening = (
            0.0
            if self.stage
            in (
                "close",
                "lift",
                "retract",
                "clearance",
                "align",
                "transfer",
                "measure_payload",
                "place",
            )
            else 1.0
        )
        # A slow vision request can outlast the runner's feed-forward hold.
        # Restore the cleared pose instead of adopting each sagged snapshot.
        cmd = (
            self.motion.command(self.verification_q, observation, 1.0)
            if self.stage in ("perceive", "verify", "settle") and self.verification_q is not None
            else self.motion.hold(observation, opening)
        )
        target = (
            self.actions[self.index][0] if self.actions and self.index < len(self.actions) else None
        )
        if self.stage == "initialize":
            try:
                try:
                    self.actions = parse_actions(self.instruction)
                except ValueError:
                    self.actions = self.perception.language_plan(observation)
                self.park_position = observation.ee_position.copy()
                self.park_position[2] = 0.64
                self.park_quat = observation.ee_quaternion.copy()
                self.enter(
                    "startup_raise" if len(self.actions) > 1 else "perceive",
                    observation,
                    "Raise for full-scene grounding" if len(self.actions) > 1
                    else "Ground the single target before the initial vertical raise",
                )
            except (ModelServiceError, ValueError) as exc:
                self.enter("safe_stop", observation, str(exc))
        elif self.stage == "startup_raise":
            cmd, reached = self.motion.pose_move(
                observation, self.raise_target, self.raise_quat, 1.0, speed=0.18, joint_step=0.065
            )
            if reached:
                self.enter("perceive", observation, "Raised; request visual grounding")
        elif self.stage == "perceive":
            target, destination = self.actions[self.index]
            try:
                if self.index == 0 and self.grasp is None:
                    names = list(
                        dict.fromkeys(
                            [n for pair in self.actions for n in pair]
                            + (["square_tray", "round_tray"] if len(self.actions) > 1 else [])
                        )
                    )
                    self.estimates = self.perception.estimate(observation, names)
                    self.scene_cache_complete = True
                    # Remove tall obstacles first; ordering is derived from perception.
                    self.actions.sort(key=lambda pair: -self.estimates[pair[0]].size[2])
                    target, destination = self.actions[0]
                elif (
                    self.retry > 0
                    or not getattr(self, "scene_cache_complete", False)
                    or target not in self.estimates
                    or destination not in self.estimates
                ):
                    self.estimates.update(self.perception.estimate(observation, [target, destination]))
                self.motion.set_scene(self.estimates)
                self.travel_height = 0.64
                self.grasp = plan_grasp(
                    self.estimates[target], self.motion, observation.joint_position
                )
                self.tray = self.estimates[destination].center.copy()
                same_tray = [a for a in self.actions if a[1] == destination]
                slot = same_tray.index((target, destination))
                if target in ("apple", "orange"):
                    sign = -1 if slot % 2 == 0 else 1
                    if destination == "square_tray":
                        self.tray[:2] += sign * np.array([0.0855, 0.0855])
                    else:
                        angle = np.deg2rad(-75 if sign < 0 else 105)
                        self.tray[:2] += 0.0805 * np.array([np.cos(angle), np.sin(angle)])
                elif len(same_tray) > 1:
                    self.tray[:2] += -0.05 if slot % 2 == 0 else 0.05
                self.carry_quat = self.grasp.quaternion.copy()
                # A dropped object may already be correctly supported in its tray.
                item = self.estimates[target]
                offset = item.center[:2] - self.estimates[destination].center[:2]
                inside = (
                    np.linalg.norm(offset) < 0.10
                    if destination == "round_tray"
                    else max(abs(offset)) < 0.105
                )
                already_placed = (
                    target in ("apple", "orange")
                    and inside
                    and abs(item.center[2] - (0.416 + 0.036)) < 0.008
                    and observation.gripper_opening > 0.95
                    and np.linalg.norm(observation.ee_position - item.center) > 0.12
                )
                self.enter(
                    "verify" if already_placed else "approach_raise",
                    observation,
                    "Recheck the fruit already resting inside its tray"
                    if already_placed
                    else "Feasible grasp and approach found",
                )
            except (ModelServiceError, ValueError) as exc:
                self.fail(observation, str(exc))
        elif self.stage == "approach_raise":
            if not hasattr(self, "approach_raise_target") or age == 0:
                self.approach_raise_target = observation.ee_position.copy()
                self.approach_raise_target[2] = max(observation.ee_position[2], self.travel_height)
            cmd, reached = self.motion.pose_move(
                observation, self.approach_raise_target, observation.ee_quaternion, 1.0, speed=0.18, joint_step=0.065
            )
            if reached:
                self.approach_target = self.grasp.pregrasp.copy()
                self.approach_target[2] = max(
                    self.grasp.clearance_height, self.grasp.pregrasp[2]
                )
                self.enter("approach_high", observation, "Translate above objects before descending")
                del self.approach_raise_target
        elif self.stage == "approach_high":
            cmd, reached = self.motion.pose_move(
                observation, self.approach_target, self.grasp.quaternion, 1.0, speed=0.20, joint_step=0.065
            )
            if reached:
                self.enter("pregrasp", observation, "Lower locally above the selected target")
        elif self.stage == "pregrasp":
            cmd, reached = self.motion.pose_move(
                observation, self.grasp.pregrasp, self.grasp.quaternion, 1.0, speed=0.10
            )
            if reached:
                self.enter("descend", observation, "Descend along approach axis with open jaws")
        elif self.stage == "descend":
            try:
                fruit = target in ("apple", "orange")
                if fruit:
                    center = self.perception.track_center(observation, target)
                    delta = center[:2] - self.grasp.position[:2]
                    self.grasp.position[:2] = center[:2]
                    self.grasp.pregrasp[:2] += delta
                    self.fruit_tracking_evidence = self.perception.last_tracking_evidence.copy()
                move = self.motion.tracked_descent if fruit else self.motion.pose_move
                cmd, reached = move(
                    observation, self.grasp.position, self.grasp.quaternion, 1.0, speed=0.08
                )
            except ModelServiceError as exc:
                self.fail(observation, str(exc))
                reached = False
            if target in ("banana", "potted_meat_can") and age > 1.2:
                # Curved-body registration can leave a few millimetres of surface
                # error. Accept settled contact inside that bound, not a long push.
                reached = reached or (
                    self.motion.last_error.get("position_error", 1.0) < (0.010 if target == "banana" else 0.004)
                    and self.motion.last_error.get("orientation_error", 1.0) < 0.045
                    and np.linalg.norm(observation.joint_velocity) < 0.05
                )
            if reached:
                if target == "banana":
                    self.grasp.position = observation.ee_position.copy()
                self.close_q = observation.joint_position.copy()
                self.enter("close", observation, "Close at the planned contact center")
        elif self.stage == "close":
            cmd = self.motion.command(self.close_q, observation, max(0.0, 1.0 - age / 0.6))
            if age >= 1.2:
                if observation.gripper_opening < 0.06:
                    self.fail(observation, "Empty close: no object between fingers")
                else:
                    self.lift_target = observation.ee_position.copy()
                    self.lift_target[2] = max(
                        getattr(self.grasp, "clearance_height", self.travel_height),
                        self.grasp.pregrasp[2],
                    )
                    self.enter("lift", observation, "Lift with fixed wrist attitude")
        elif self.stage == "lift":
            cmd, reached = self.motion.pose_move(
                observation,
                self.lift_target,
                self.grasp.quaternion,
                0.0,
                speed=0.14,
                joint_step=0.055,
            )
            if (
                getattr(self.grasp, "clearance_height", self.travel_height)
                < self.travel_height - 0.02
                and age > 1.2
            ):
                clearance_height = getattr(self.grasp, "clearance_height", self.travel_height)
                reached = reached or (
                    observation.ee_position[2] >= clearance_height - 0.003
                    and self.motion.last_error.get("position_error", 1.0) < 0.015
                    and self.motion.last_error.get("orientation_error", 1.0) < 0.045
                    and np.linalg.norm(observation.joint_velocity) < 0.05
                )
            if reached:
                if observation.ee_position[2] < self.travel_height - 0.02:
                    self.retract_target = observation.ee_position.copy()
                    radius = np.linalg.norm(self.retract_target[:2])
                    self.retract_target[:2] *= min(1.0, 0.53 / max(radius, 1e-9))
                    self.enter(
                        "retract",
                        observation,
                        "Low edge departure complete; retract inward before full raise",
                    )
                else:
                    self.carry_height = max(observation.ee_position[2], self.travel_height)
                    self.carry_target = np.r_[self.tray[:2], self.carry_height]
                    self.enter(
                        "transfer",
                        observation,
                        "Vertical departure complete; carry at clearance height",
                    )
        elif self.stage == "retract":
            cmd, reached = self.motion.pose_move(
                observation,
                self.retract_target,
                self.grasp.quaternion,
                0.0,
                speed=0.12,
            )
            if reached:
                self.clearance_target = observation.ee_position.copy()
                self.clearance_target[2] = self.travel_height
                self.enter(
                    "clearance",
                    observation,
                    "Retracted inward; raise the payload to full travel height",
                )
        elif self.stage == "clearance":
            cmd, reached = self.motion.pose_move(
                observation,
                self.clearance_target,
                self.grasp.quaternion,
                0.0,
                speed=0.14,
            )
            if reached:
                self.carry_height = max(observation.ee_position[2], self.travel_height)
                self.carry_target = np.r_[self.tray[:2], self.carry_height]
                self.enter(
                    "transfer",
                    observation,
                    "Full travel height reached in the inner workspace",
                )
        elif self.stage == "transfer":
            cmd, reached = self.motion.pose_move(
                observation,
                self.carry_target,
                self.carry_quat,
                0.0,
                speed=0.18,
                joint_step=0.055,
            )
            if reached:
                self.place_target = self.carry_target.copy()
                self.place_target[2] = self.grasp.position[2] + (
                    0.016 if target in ("apple", "orange") else 0.025
                )
                self.place_progress_position = observation.ee_position.copy()
                self.place_progress_time = observation.time
                self.enter(
                    "measure_payload" if target in ("apple", "orange") else "place",
                    observation,
                    "Measure the held object before lowering"
                    if target in ("apple", "orange")
                    else "Lower the object into the tray",
                )
        elif self.stage == "measure_payload":
            try:
                center = self.perception.held_center(observation, target)
                self.payload_offset = center - observation.ee_position
                self.place_target = (
                    np.r_[self.carry_target[:2], 0.416 + 0.036] - self.payload_offset
                )
                self.place_target[2] = max(self.place_target[2], 0.475)
                self.place_progress_position = observation.ee_position.copy()
                self.place_progress_time = observation.time
                self.enter("place", observation, "Lower using the measured object-to-hand offset")
            except (ModelServiceError, ValueError) as exc:
                self.fail(observation, str(exc))
        elif self.stage == "place":
            cmd, reached = self.motion.pose_move(
                observation, self.place_target, self.carry_quat, 0.0, speed=0.06
            )
            if np.linalg.norm(observation.ee_position - self.place_progress_position) > 0.001:
                self.place_progress_position = observation.ee_position.copy()
                self.place_progress_time = observation.time
            supported = (
                observation.time - self.place_progress_time > 0.6
                and np.linalg.norm(observation.ee_position[:2] - self.place_target[:2]) < 0.006
                and observation.ee_position[2]
                < self.place_target[2] + (0.16 if target == "banana" else 0.045)
            )
            if reached or supported:
                self.release_q = observation.joint_position.copy()
                self.enter(
                    "release",
                    observation,
                    "Release at support contact"
                    if supported
                    else "Open jaws and let the object settle",
                )
        elif self.stage == "release":
            release_command = 1.0
            if target in ("apple", "orange"):
                if age < 0.7:
                    release_command = self.release_opening * float(np.clip((age - 0.3) / 0.4, 0, 1))
                else:
                    release_command = self.release_opening + (1 - self.release_opening) * float(
                        np.clip((age - 0.7) / 1.5, 0, 1)
                    )
            cmd = self.motion.command(
                self.release_q,
                observation,
                release_command,
            )
            # The fruit is already supported before the jaws finish opening at
            # 2.2 s. Keep a short settling margin, then start the empty retreat.
            if age > (2.3 if target in ("apple", "orange") else 0.7):
                self.retreat_target = observation.ee_position.copy()
                self.retreat_target[2] += 0.07 if target in ("apple", "orange") else 0.10
                self.retreat_target[2] = max(self.retreat_target[2], self.travel_height)
                self.enter("retreat", observation, "Clear the released object")
        elif self.stage == "retreat":
            cmd, reached = self.motion.pose_move(
                observation,
                self.retreat_target,
                self.carry_quat,
                1.0,
                speed=0.10 if target in ("apple", "orange") else 0.08,
            )
            if reached:
                if target in ("apple", "orange"):
                    # Clear both camera rays with a short local translation.
                    # The previous opposite-side waypoint crossed the entire table.
                    self.observation_target = observation.ee_position.copy()
                    self.observation_target[0] += 0.10
                    self.observation_target[2] = self.travel_height
                    if len(self.actions) > 1:
                        # Multi-object continuation needs both tray masks visible.
                        destination = self.actions[self.index][1]
                        side = -1.0 if self.estimates[destination].center[1] >= 0 else 1.0
                        self.observation_target = np.array([0.50, side * 0.15, self.travel_height])
                    self.enter(
                        "observe_clear",
                        observation,
                        "Move the empty hand away from the camera view",
                    )
                else:
                    self.enter("verify", observation, "Re-observe object in its tray")
        elif self.stage == "observe_clear":
            cmd, reached = self.motion.pose_move(
                observation, self.observation_target, self.carry_quat, 1.0, speed=0.20, joint_step=0.065
            )
            if reached:
                self.enter("verify", observation, "Re-observe the tray with the hand out of view")
        elif self.stage == "verify" and age >= 0.5:
            try:
                target, destination = self.actions[self.index]
                if target in ("apple", "orange"):
                    # Trays are fixed; retain their calibrated visual estimate.
                    estimates = {destination: self.estimates[destination]}
                    center = self.perception.held_center(
                        observation, target, near=estimates[destination].center
                    )
                    estimates[target] = Estimate(
                        target,
                        center,
                        np.array([0.076, 0.076, 0.072]),
                        np.eye(3),
                        {"service": "sam3_two_view_surface"},
                    )
                else:
                    center, evidence = placed_center(
                        self.perception, observation, target, self.estimates[destination].center
                    )
                    prior = self.estimates[target]
                    estimates = {
                        destination: self.estimates[destination],
                        target: Estimate(
                            target, center, prior.size.copy(), prior.rotation.copy(), evidence
                        ),
                    }
                self.estimates.update(estimates)
                offset = estimates[target].center[:2] - estimates[destination].center[:2]
                self.validation = {
                    "target": target,
                    "offset_xy": offset.tolist(),
                    "center": estimates[target].center.tolist(),
                    "tray_center": estimates[destination].center.tolist(),
                    "cached_static_tray": True,
                    "evidence": estimates[target].evidence,
                }
                inside = (
                    np.linalg.norm(offset) < 0.10
                    if destination == "round_tray"
                    else max(abs(offset)) < 0.105
                )
                if not inside:
                    self.fail(observation, "Visual verification: object outside destination")
                else:
                    self.verification_centers.append(estimates[target].center.copy())
                    recent = self.verification_centers[-3:]
                    stable = (
                        len(recent) >= 3
                        and np.max(
                            np.linalg.norm(np.asarray(recent) - np.mean(recent, axis=0), axis=1)
                        )
                        < 0.0015
                    )
                    if (
                        target not in ("apple", "orange")
                        or stable
                        or (self.index < len(self.actions) - 1 and len(recent) >= 3)
                    ):
                        self.index += 1
                        self.retry = 0
                        self.enter(
                            "settle" if self.index == len(self.actions) else "perceive",
                            observation,
                            "Visual placement verified",
                        )
            except (ModelServiceError, ValueError) as exc:
                self.fail(observation, str(exc))
        elif self.stage == "recover":
            cmd, reached = self.motion.pose_move(
                observation, self.raise_target, self.raise_quat, 1.0, speed=0.18, joint_step=0.065
            )
            if reached:
                self.enter("park_high", observation, "Return to the raised observation pose")
        elif self.stage == "park_high":
            cmd, reached = self.motion.pose_move(
                observation, self.park_position, self.raise_quat, 1.0, speed=0.20, joint_step=0.065
            )
            if reached:
                self.enter("park_orient", observation, "Orient the empty hand in the inner workspace")
        elif self.stage == "park_orient":
            cmd, reached = self.motion.pose_move(
                observation, self.park_position, self.park_quat, 1.0, joint_step=0.065
            )
            if reached:
                self.enter("perceive", observation, "Parked high; acquire fresh observation")
        if (
            self.stage in ("lift", "retract", "clearance", "align", "transfer", "place")
            and observation.gripper_opening < 0.05
        ):
            destination = self.actions[self.index][1]
            over_destination = (
                np.linalg.norm(observation.ee_position[:2] - self.estimates[destination].center[:2])
                < 0.13
            )
            if self.stage == "place" or over_destination:
                self.release_q = observation.joint_position.copy()
                self.enter(
                    "release",
                    observation,
                    "Object left jaws over destination; clear and verify before retrying",
                )
            else:
                self.fail(observation, "Grasp lost during loaded motion")
        if (
            age > 10.0
            and self.stage not in ("settle", "safe_stop", "perceive", "verify")
            and self.entered != observation.time
        ):
            self.fail(observation, "Motion timeout; re-observe")
        # The benchmark remembers a verification stage for the whole episode.
        # Expose final verification only after the last object was released and
        # the hand retreated, so a wide fruit still held in the jaws cannot end
        # a multi-object episode early through the gripper-opening predicate.
        reported_stage = (
            "verify_item"
            if self.stage == "verify" and self.index < len(self.actions) - 1
            else "verify_place"
            if self.stage == "settle"
            else self.stage
        )
        return PolicyDecision(
            cmd,
            reported_stage,
            self.reason,
            target_id=target,
            # Keep the official runner alive for its ten-frame physical check.
            done=self.stage == "safe_stop",
            request_retry=self.stage == "recover",
            debug={
                "motion": self.motion.last_error,
                "fruit_tracking": getattr(self, "fruit_tracking_evidence", {}),
                "retry": self.retry,
                "task_plan": {"actions": [{"pick_id": p, "place_id": d} for p, d in self.actions]},
                "model_evidence": {k: v.evidence for k, v in self.estimates.items()},
                "geometry": {
                    k: {"center": v.center.tolist(), "size": v.size.tolist()}
                    for k, v in self.estimates.items()
                },
                "grasp_position": self.grasp.position.tolist() if self.grasp is not None else None,
                "visual_verification": self.validation,
            },
        )
