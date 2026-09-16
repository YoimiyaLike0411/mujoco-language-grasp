# V3 新抽取随机 seed 补测

结果：12/15 通过；不安全接触 0 次。

在运行前使用操作系统随机源，从 100000–999999 中不放回抽取 5 个 seed，并保存抽样记录。
每个 seed 各运行一个 Task 1、一个 Task 2 和一个完整 Task 3。Task 1 轮换公开物体模板，Task 2 为三个苹果、两个香蕉案例。
控制器与此前通过 30 个案例的 V3 源码指纹完全一致。本轮仅测试，未修改策略。
使用真实 SAM3、原始物理、原始实时异步评测，以及 800/3000 步预算。

| Seed | Task 1 | Task 2 | Task 3 |
|---|---|---|---|
| 919900 | 通过 467/800 | 通过 665/800 | 通过 2624/3000 |
| 127945 | 通过 430/800 | 通过 599/800 | 通过 2479/3000 |
| 977989 | 通过 531/800 | 通过 570/800 | 失败 2203/3000 |
| 874694 | 通过 505/800 | 通过 590/800 | 失败 1384/3000 |
| 145455 | 通过 515/800 | 通过 703/800 | 失败 297/3000 |

## 性能余量

- 800 步预算下最紧的成功案例：random_task2_145455，703 步，余量 97 步。
- 3000 步预算下最紧的成功案例：random_task3_919900，2624 步，余量 376 步。

## 失败回合

- random_task3_145455：policy_done，最终阶段 safe_stop；目标状态 {'apple->round_tray': False, 'orange->round_tray': False, 'mustard_bottle->square_tray': False, 'potted_meat_can->square_tray': False}。
  - 触发原因：Empty close: no object between fingers；Unreliable CAD registration for mustard_bottle: 0.0094 m
  - 失败视频：outputs/random_task3_145455_failure.mp4
- random_task3_977989：policy_done，最终阶段 safe_stop；目标状态 {'apple->round_tray': True, 'orange->round_tray': False, 'mustard_bottle->square_tray': True, 'potted_meat_can->square_tray': True}。
  - 触发原因：Fruit tracking surface mismatch；Unreliable fruit tracking update
  - 失败视频：outputs/random_task3_977989_failure.mp4
- random_task3_874694：policy_done，最终阶段 safe_stop；目标状态 {'apple->round_tray': False, 'orange->round_tray': False, 'mustard_bottle->square_tray': True, 'potted_meat_can->square_tray': True}。
  - 触发原因：Unreliable fruit tracking update
  - 失败视频：outputs/random_task3_874694_failure.mp4

## 复现与证据

- 配置：configs/rlx_random_v3_20260915_1.json
- 抽样记录：runs/rlx_random_v3_20260915_1/sampling.json
- 每回合日志和视频：runs/rlx_random_v3_20260915_1/<episode_id>/

```sh
runtime/run.sh scripts/validate_seeds.py --tasks configs/rlx_random_v3_20260915_1.json --output runs/recheck_random_v3 --video
```

这是对新抽取的 5 个 seed、15 个案例的补测，不构成对所有未测 seed 的保证。
