## 2026-09-16 current adaptive-clearance snapshot

- Ruff and all 54 automated tests pass.
- Task 3 seed 977989 passes all four goals in 2809/3000 control steps with zero retries and zero unsafe contacts.
- The planner first requests a 0.64 m departure. When that pose is unreachable, it searches for the highest verified height from 0.50 m to 0.64 m instead of using a fixed 0.50 m fallback.
- On Task 3 seed 874039, the edge potted-meat can departs at about 0.54 m, retracts inward, then rises to about 0.64 m before transfer; no other object is disturbed before its own pick and no unsafe contact is recorded.
- The full seed-874039 episode remains 3/4 because the edge orange can slip during its long loaded transfer. This seed is not counted as a full success.

The older sections below describe earlier frozen revisions and remain as historical evidence.

---

## 最新随机 seed 补测：12/15 通过

在冻结 V3 上新抽取 5 个 seed：[919900, 127945, 977989, 874694, 145455]。每个各测 Task 1、Task 2、Task 3。
不安全接触 0 次，策略源码未改。详细结果见 outputs/RANDOM_SEED_REPORT_20260915_1.md。

---

## V3 完整验证通过

- 同一冻结版本：公开 15/15，额外 15/15；Task 1 12/12、Task 2 12/12、Task 3 6/6。
- 真实 SAM3、原始实时异步评测、原来的 800/3000 步预算；不安全接触为 0。
- 47 项自动测试通过，两个验证集的控制器源码指纹与交付代码一致。
- 原始物理、机器人资产和评测源码哈希一致。
- 最紧的 800 步案例为苹果 seed 4093，774 步；最紧的 3000 步案例为 seed 92，2792 步。
- 已测种子全部通过，不代表对所有未测种子的数学保证；苹果时间余量仍有限。
- 报告、逐回合结果、补丁代码包、成功视频和校验表位于 outputs/。
- 本次恢复工作只在隔离的远端项目目录内写入文件。

最终证据：runs/rlx_final_public_v3、runs/rlx_final_extra_v3。
历史失败与开发记录保留如下，不混入最终成绩。

---

## Current revision: V3 integrated, full validation pending

V3 combines live RGB-D/CAD fruit tracking initialized by real SAM3, independent
lateral correction during descent, and segmentation-surface location checks
after release. Placement verification no longer requires an upright grasp pose
registration for tilted packages.

Isolated evidence:
- Coupled tracking failed Task 3 seed 91; decoupled tracking passed seed 91.
- Decoupled tracking passed apples 62/64/66 and Task 3 seed 92.
- Seed 93 failed after can CAD verification caused two unnecessary regrasp cycles.
- Surface-only placement verification plus tracking passed seed 93.
These are development cases, not a completed uniform-revision sweep.

The preceding V2 source/tests are preserved at
runtime/snapshots/frozen_payload_clearance_v2. V3 must pass all 47 tests and all
30 full-regression cases before packaging. Historical failures are preserved.
No scene/physics/evaluator changes; no hidden state or seed inputs to the policy.

---

## Current revision: live fruit tracking under investigation

V2 public regression finished 14/15 with zero unsafe contacts. Task 3 seed 91
failed physical stability after a tilted apple grasp. The extra V2 sweep was
interrupted and is incomplete; do not treat it as a full result.

Deterministic applied-command replays of both V1 success and V2 failure reproduced
every recorded robot joint state exactly (maximum error 0). The apple moved
several millimetres during descent; the stale target caused one-sided finger
contact before closing. RGB-D appearance/CAD tracking initialized by real SAM3
has submillimetre XY error before that contact, about 12 ms per snapshot.
It remains an experimental runtime candidate until real-model regression passes.

Evidence: runs/rlx_tracking_replay_failed_91, runs/rlx_tracking_replay_passed_91.
Current production policies retain the V2 geometry fixes; experiments are isolated.

---

## Current revision: payload clearance fixes under validation

The first frozen version passed all 15 public episodes, then 10/12 additional
episodes. Preserve that result: it is not a universal-seed claim.
Additional failures were banana seed 1234 (palm blocked low approach) and
Task 3 seed 2027 (tall bottle dropped during low retraction).

Both failure cases passed isolated real-model retests after:
- checking open-hand contact against the visually registered banana in the
  separate planning world, with both tilt directions available;
- raising tall payloads during retraction instead of moving horizontally low.

The current revision merges these fixes and is frozen for a new complete run:
36 tests, all 15 public episodes, and 15 additional episodes. Seed 4093 was
selected before viewing its results. Evidence is written to
runs/rlx_final_public_v2 and runs/rlx_final_extra_v2. These are in progress.
The preceding source/tests are preserved in runtime/snapshots/frozen_fast_clear_v1.
The rejected banana contact-tolerance candidate remains isolated in runtime.

---

# Clean grasp controller rebuild

Objective: reliable Task 1, Task 2 and Task 3 across randomized positions and yaw,
with language and real model grounding, unchanged course physics and evaluation.
Passing a finite seed set is evidence, not a proof of arbitrary-seed success.

## Fixed baseline
- Copied from the newly downloaded `mujoco-grasp-assignment-1`.
- Do not change scene.xml, panda.xml, runner.py, env.py or success predicates.
- Development-only oracle perception isolates control; never a submission fallback.
- Preserve failures and report oracle and real-model evaluations separately.

## Implementation order
1. Measured-state motion, gravity feed-forward through the position-command API,
   position AND orientation acceptance, geometric grasp candidates.
2. One shared pick/place controller with explicit recovery and visual verification.
3. SAM3 mask geometry and instruction parsing with validation.
4. Public async evaluation and additional independently selected seeds, videos.

## External references
- https://github.com/kevinzakka/mjctrl/blob/main/diffik_nullspace.py
  Damped Jacobian IK, velocity limits and gravity compensation. Conceptual reference;
  the course forbids changing physics, so use q_bias / actuator gain in commands,
  not the example's modification of body_gravcomp.
- https://robosuite.ai/docs/simulation/controller.html
  Separation of motion goals and low-level robot control. We retain the course
  joint-position API instead of replacing it with robosuite torque commands.

## Verified evidence (2026-09-15)

All results below use the untouched starter scene, Panda assets and evaluator.
Results are version-specific; ongoing changes require a fresh complete regression.

| Run | Input/control | Result |
| --- | --- | --- |
| `runs/real_task1_v3` | Real SAM3, default async + realtime | 6/6 public seeds 31–36, mean 617.5 steps, zero unsafe contacts |
| `runs/real_banana_v1` | Real SAM3, default async + realtime | Task 2 seed 61 passed in 585 steps; video saved |
| `runs/real_task2_v2` | Real SAM3, default async + realtime | 2/6; fruit rolling/time budgets and banana contact-height error remain |
| `runs/task3_oracle91_v15` | Oracle geometry, original physical simulation | All four objects released and stable, 1836 steps; NOT formal success |
| `runs/dev_task3_93_v13` | Oracle geometry, original physical simulation | All four objects released and stable, 2666 steps; NOT formal success |
| `runs/real_task3_v2` | Real SAM3, default async + realtime | 0/3; fruit placement remains unresolved |
| `runs/banana_candidate2_public` | Real SAM3, default async + realtime, source manifest recorded | 3/3 banana seeds 61, 63, 65; mean 583.3 steps, zero unsafe contacts |

The initial 25-test regression passed with macOS rendering available. Tests verify
API behavior, deterministic commands, static gravity compensation, language
relationships and unchanged official files. They do not prove grasping success.

## Current implementation

- `policies/motion.py`: measured-state position/orientation control, command-level
  gravity feed-forward, and collision screening in a sanitized planning scene.
- `policies/grounding.py`: SAM3 masks, calibrated depth, body-local CAD registration,
  robust tray-rim fitting, clause parsing and optional starter VLM fallback.
- `policies/grasp_geometry.py`: geometric grasp/approach search with IK feasibility.
- `policies/cad_geometry.py`: visible-surface registration and curved banana section.
- `policies/student_policy.py`: one pick/place state machine, limited recovery,
  payload clearance, contact-aware release, fruit support and visual verification.

No oracle fallback exists in the production policy. Diagnostic truth is confined
to explicitly named scripts and files. No seed or hidden target is read by policy.
Final verification is reported only after the last object has been released and
the hand has retreated; earlier per-object checks use a separate stage.

## Reproduce

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
export MUJOCO_GL=glfw  # macOS; use egl for a compatible headless Linux host
export GRASPBENCH_SAM3_URL=http://127.0.0.1:18766/infer
.venv/bin/python scripts/validate_seeds.py --tasks configs/task1_place_public.json --output runs/new_task1
.venv/bin/python scripts/validate_seeds.py --tasks configs/task2_ycb_public.json --output runs/new_task2
.venv/bin/python scripts/validate_seeds.py --tasks configs/task3_sort_public.json --output runs/new_task3
```

`--seeds 101 202` repeats each task instruction with additional independently chosen
layout seeds. `--video` records each rollout. A source-hash manifest and partial
aggregate are saved alongside the untouched runner's per-episode summaries.
The SAM3 endpoint must be available; no silent fallback is used. Ambiguous language
uses the starter VLM adapter only when its documented environment is configured.

## Frozen controller under full validation

The current production controller includes the fast direct-support candidate.
Its isolated real async/realtime validation passed apples 62/64/66 in
710/711/739 steps and Task 3 seed 91 in 2560 steps, all with zero unsafe contacts.
After integration, all 32 tests passed. The frozen full public regression passed 15/15 with zero unsafe contacts:
Task 1 6/6, Task 2 6/6, Task 3 3/3 (2546/2802/2616 steps).
The 12 preselected additional episodes (seeds 1234, 2027) are now running under
runs/rlx_final_extra; runs/rlx_final_public contains the completed public result.

Changes beyond the prior integrated-center version:
- Direct vertical placement at tray wall support; no floor-level lateral push.
- Faster fruit lift and transfer to reduce time-dependent slip in the jaws.
- Raise and move the empty hand out of the camera view before fruit verification.
- Allow remaining objects to progress after three inside-tray observations;
  final physical success still uses the untouched evaluator.
- If a fruit drops over its destination, clear and visually verify it first.
  Re-observed fruit already supported in the tray is not blindly regrasped.

The prior formal controller is preserved at
runtime/snapshots/integrated_center/policies. The exact successful experimental
source remains runtime/fast_corner_source.py; production formatting is equivalent.

## Remote execution (authorization restored)

The user explicitly approved sending synthetic MuJoCo JPEGs and object names to
the campus SAM3 service. All resumed writes were restricted to the isolated
remote project checkout. Local/Desktop copies were not synchronized.

Use runtime/run.sh for the remote Python environment. It confines caches/temp
files to this checkout, selects installed NVIDIA EGL via runtime/nvidia_egl.json,
and uses SAM3 at http://127.0.0.1:8766/infer. No system configuration or physics
files were changed. The network interruption has been resolved.

### Measured changes

- IK uses a model copy with contact solving disabled only for inverse kinematics.
  Original simulation and collision screening remain unchanged. Same-input
  planning benchmark: 6.640 s to 0.282 s with identical grasp joint vectors.
- Apples/oranges now use public CAD surface registration with two camera views.
  Initial apple center errors decreased in the saved same-frame audit.
- Fruit is placed directly at tray-wall support without a lateral push or wrist reorientation.
- Verification commands restore the original cleared joint pose, avoiding
  cumulative arm sag during repeated slow visual calls. Fixed tray estimates
  are cached; fruit observations remain live.
- After visual verification, the controller holds clearance in settle instead
  of terminating early. The unchanged evaluator judges ten-frame stability.

The previous production policies are preserved in runtime/snapshots/fast_ik_only.
Older runtime wrappers dynamically compile production files; use their saved
source manifests and this snapshot when reproducing historical experiments.

### Version-specific evidence

- Fast-IK-only real async/realtime: Task 1 6/6, banana 3/3, zero unsafe contacts.
- Anchored CAD experimental real async/realtime: apple 1/3.
- Anchored CAD Task 3 seed 91: four final goal predicates true at step 3000,
  but no ten-frame streak; FAILURE, not a successful sorting episode.
- Integrated production real Task 2: 3/6 (all bananas passed, all apples failed).
  The integrated-center regression finished Task 1 6/6 and Task 3 0/3.
  This is the superseded controller, preserved in its snapshot.
- Original 26 tests passed after integration. Seven focused tests, including
  cleared-pose retention and official-file hashes, also passed.
- Isolated lower-release-height diagnostics: floor-only 3/3 synchronous and
  3/3 two-frame reuse. Softer descent 3/3 synchronous but 2/3 two-frame reuse.
  These use oracle geometry and are NOT real-perception scores.

### Remaining work

Fruit can arrive tilted or be released above the floor and keep rocking.
Lower-release-height candidates failed 0/3 real apple tests and remain isolated.
The fast direct-support candidate has been merged after successful real apple
and Task 3 seed-91 tests. Complete the frozen public and additional regressions
before claiming broad reliability. The full objective is not yet verified.

Historical local reports above describe older versions and environments. Current
remote evidence and source fingerprints take precedence. A finite seed suite
does not prove arbitrary-seed success.
