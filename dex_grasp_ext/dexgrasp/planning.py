"""Grasp planning: turn an object's perceived size into hand parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hand import GRASP_CENTRE_LOCAL, plan_closure
from .perception import grasp_radius


@dataclass(frozen=True)
class GraspPlan:
    closure: float
    yaw_deg: float
    centre_local: np.ndarray
    radius_m: float
    source: str

    def as_dict(self) -> dict:
        return {
            "closure": self.closure,
            "yaw_deg": self.yaw_deg,
            "centre_local": self.centre_local.tolist(),
            "radius_m": self.radius_m,
            "source": self.source,
        }


def plan_grasp(
    object_name: str,
    calibration: list[dict] | None = None,
    *,
    yaw_deg: float = 0.0,
    closure_override: float | None = None,
    radius_override: float | None = None,
) -> GraspPlan:
    """Choose the finger closure / cage centre for `object_name`.

    With a calibration table the plan is derived from the measured relation
    between closure and enclosed radius; without one it falls back to the
    hand-tuned constant so experiments can still run.
    """
    radius = radius_override if radius_override is not None else grasp_radius(object_name)
    if calibration:
        row = (
            plan_closure(calibration, radius)
            if closure_override is None
            else min(calibration, key=lambda item: abs(item["fraction"] - closure_override))
        )
        return GraspPlan(
            closure=float(row["fraction"]),
            yaw_deg=float(yaw_deg),
            centre_local=np.asarray(row["centre_local"], dtype=np.float64),
            radius_m=float(row["radius_m"]),
            source="calibrated",
        )
    return GraspPlan(
        closure=float(closure_override if closure_override is not None else 0.6),
        yaw_deg=float(yaw_deg),
        centre_local=np.asarray(GRASP_CENTRE_LOCAL, dtype=np.float64),
        radius_m=float(radius),
        source="constant",
    )


def rank_plans(
    object_name: str,
    calibration: list[dict] | None = None,
    grasp_table: dict | None = None,
    *,
    limit: int = 3,
) -> list[GraspPlan]:
    """Candidate plans in preference order, used for retry after a failed lift.

    The synthesis table's per-candidate lift heights are the ordering signal;
    when the table is missing we fall back to a small closure ladder around the
    calibrated plan so the retry behaviour still exists.
    """
    plans: list[GraspPlan] = []
    entry = (grasp_table or {}).get(object_name)
    if entry:
        ranked = sorted(
            (row for row in entry.get("trials", []) if row.get("lift_m", 0) > 0.04),
            key=lambda row: (-row["lift_m"], row.get("object_drift_after_close_m") or 0.0),
        )
        for row in ranked[:limit]:
            plans.append(
                plan_grasp(
                    object_name,
                    calibration,
                    yaw_deg=row["yaw_deg"],
                    closure_override=row["closure"],
                )
            )
    if not plans:
        for delta in (0.0, -0.05, 0.05, -0.1):
            plans.append(
                plan_grasp(object_name, calibration, closure_override=0.6 + delta)
            )
    seen: set[tuple[float, float]] = set()
    unique = []
    for plan in plans:
        key = (round(plan.closure, 3), round(plan.yaw_deg, 1))
        if key not in seen:
            seen.add(key)
            unique.append(plan)
    return unique[:limit]
